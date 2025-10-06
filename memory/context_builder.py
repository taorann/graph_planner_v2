# memory/context_builder.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import ast
import io
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Tuple, Dict, Optional, Set

# -----------------------------
# Public data types & exception
# -----------------------------

ChunkKind = Literal["window", "skeleton", "dep_local", "dep_cross", "full_file"]

@dataclass
class DocChunk:
    path: str
    kind: ChunkKind
    start: int       # 1-based inclusive
    end: int         # 1-based inclusive
    text: str
    est_tokens: int
    why: str         # explanation / provenance

@dataclass
class ContextPack:
    chunks: List[DocChunk]
    total_tokens: int
    budget_tokens: int
    truncated: bool
    mode: str        # "wsd" or "full"

class ContextTooLarge(Exception):
    """Raised when FULL mode content exceeds sensible limits (no trimming here)."""
    pass


# -----------------------------
# Public entry
# -----------------------------

def build(
    plan,                      # memory.types.Plan (duck-typed)
    subgraph: List[str],       # candidate neighbor file paths (relative to repo_root)
    *,
    repo_root: str,
    token_budget: int,
    mode: str,                 # "wsd" | "full"
    radius_r: int = 12,
    deps_per_file: int = 3,
    cross_deps_limit: int = 8,
) -> ContextPack:
    """
    Build a context pack for CGM text modality.
    - mode = "full": include full text for all target files (no trimming; raises if too large).
    - mode = "wsd" : window + skeleton + 1-hop deps (local/cross) with a token budget & trimming.
    """
    if mode not in ("wsd", "full"):
        raise ValueError("context.mode must be 'wsd' or 'full'")

    root = Path(repo_root)
    if not root.exists():
        raise FileNotFoundError(f"repo_root not found: {repo_root}")

    if mode == "full":
        return build_full(plan, repo_root)
    else:
        return build_wsd(
            plan,
            subgraph,
            repo_root=repo_root,
            token_budget=token_budget,
            radius_r=radius_r,
            deps_per_file=deps_per_file,
            cross_deps_limit=cross_deps_limit,
        )


# -----------------------------
# FULL mode
# -----------------------------

def build_full(plan, repo_root: str) -> ContextPack:
    files = _sorted_unique_files_from_plan(plan)
    chunks: List[DocChunk] = []
    total = 0

    for rel in files:
        txt = _read_text(Path(repo_root) / rel)
        est = _est_tokens(txt)
        chunks.append(
            DocChunk(
                path=rel,
                kind="full_file",
                start=1,
                end=_line_count(txt),
                text=txt,
                est_tokens=est,
                why="full",
            )
        )
        total += est

    # 在 FULL 模式下不做裁剪；如需保护，可在此加入一个粗略上限
    # （例如禁止超过 120k tokens），这里保持“严格不裁剪”的约定：
    return ContextPack(
        chunks=chunks,
        total_tokens=total,
        budget_tokens=total,
        truncated=False,
        mode="full",
    )


# -----------------------------
# WSD mode
# -----------------------------

def build_wsd(
    plan,
    subgraph: List[str],
    *,
    repo_root: str,
    token_budget: int,
    radius_r: int,
    deps_per_file: int,
    cross_deps_limit: int,
) -> ContextPack:
    """
    Window + Skeleton + 1-hop Deps, with trimming order:
    dep_cross -> dep_local -> skeleton -> (finally) window tail
    """
    root = Path(repo_root)
    # 1) Collect windows per file
    windows_by_file: Dict[str, List[Tuple[int, int]]] = {}
    for t in getattr(plan, "targets", []):
        windows_by_file.setdefault(t.path, []).append((int(t.start), int(t.end)))

    # Normalize windows (merge overlaps)
    for p, wins in windows_by_file.items():
        windows_by_file[p] = _merge_windows(sorted(wins))

    chunks: List[DocChunk] = []
    # For local dep extraction
    symbol_cache_by_file: Dict[str, Set[str]] = {}

    # 2) Build window + skeleton + local deps per file
    for rel_path, win_list in windows_by_file.items():
        abs_path = root / rel_path
        file_text = _read_text(abs_path)
        line_total = _line_count(file_text)
        lines = _split_lines(file_text)

        # --- windows ---
        for (s, e) in win_list:
            ws, we = _expand_window(s, e, radius_r, line_total)
            text = _slice_lines(lines, ws, we)
            chunks.append(DocChunk(
                path=rel_path,
                kind="window",
                start=ws,
                end=we,
                text=text,
                est_tokens=_est_tokens(text),
                why=f"plan_window±{radius_r}",
            ))

        # --- skeleton ---
        skel_lines = _python_skeleton(abs_path, file_text)
        if skel_lines:
            skel_text = "\n".join(skel_lines) + "\n"
            chunks.append(DocChunk(
                path=rel_path,
                kind="skeleton",
                start=1,
                end=line_total,
                text=skel_text,
                est_tokens=_est_tokens(skel_text),
                why="skeleton",
            ))

        # --- local deps ---
        # Candidates: identifiers used in window regions but not defined within them.
        win_text_concat = "".join(
            _slice_lines(lines, _expand_window(s, e, radius_r, line_total)[0],
                         _expand_window(s, e, radius_r, line_total)[1])
            for (s, e) in win_list
        )
        used = _extract_identifiers(win_text_concat)
        defined_here = _extract_local_defs(file_text)  # defs/classes/constants in file
        undefined = [name for name in used if name not in defined_here]

        symbol_cache_by_file[rel_path] = set(defined_here)

        # Find definitions inside the same file for undefined symbols
        local_hits = _find_symbol_defs_in_file(lines, undefined, deps_per_file)
        for name, (ds, de) in local_hits:
            frag = _slice_lines(lines, max(1, ds - 5), min(line_total, de + 5))
            chunks.append(DocChunk(
                path=rel_path,
                kind="dep_local",
                start=max(1, ds - 5),
                end=min(line_total, de + 5),
                text=frag,
                est_tokens=_est_tokens(frag),
                why=f"local_def {name}",
            ))

    # 3) Cross-file deps (scan neighbor files suggested by subgraph)
    #    Heuristic: only search in .py files listed in subgraph (excluding current file)
    need_cross: Dict[str, Set[str]] = {}
    for rel_path, defs in symbol_cache_by_file.items():
        # Names used but not defined locally (again), reuse computed from used/defined?
        # Simpler: re-compute per file for clarity
        file_text = _read_text(root / rel_path)
        lines = _split_lines(file_text)
        line_total = len(lines)
        used = _extract_identifiers(file_text)
        undefined = set(n for n in used if n not in defs)
        if undefined:
            need_cross[rel_path] = undefined

    cross_added = 0
    # Build a map for neighbor candidate paths (only .py)
    neighbor_candidates = [p for p in subgraph if p.endswith(".py")]
    for rel_path, names in need_cross.items():
        if cross_added >= cross_deps_limit:
            break
        for nb in neighbor_candidates:
            if nb == rel_path:
                continue
            nb_text = _read_text(root / nb)
            nb_lines = _split_lines(nb_text)
            nb_total = len(nb_lines)
            # Find best matches for needed names in neighbor file
            matches = _search_defs_in_text(nb_text, names, limit=min(2, len(names)))
            for name, (s, e) in matches:
                frag = _slice_lines(nb_lines, max(1, s - 5), min(nb_total, e + 5))
                chunks.append(DocChunk(
                    path=nb,
                    kind="dep_cross",
                    start=max(1, s - 5),
                    end=min(nb_total, e + 5),
                    text=frag,
                    est_tokens=_est_tokens(frag),
                    why=f"cross_def {name}",
                ))
                cross_added += 1
                if cross_added >= cross_deps_limit:
                    break
            if cross_added >= cross_deps_limit:
                break

    # 4) Deduplicate chunks (by path + start + end + text hash)
    chunks = _dedup_chunks(chunks)

    # 5) Sort: window > skeleton > dep_local > dep_cross; then by path/start
    kind_order = {"window": 0, "skeleton": 1, "dep_local": 2, "dep_cross": 3, "full_file": 4}
    chunks.sort(key=lambda c: (kind_order.get(c.kind, 9), c.path, c.start))

    # 6) Trim to budget (dep_cross -> dep_local -> skeleton -> window tail)
    total = sum(c.est_tokens for c in chunks)
    truncated = False
    if total > token_budget:
        truncated = True
        total, chunks = _trim_to_budget(chunks, token_budget)

    return ContextPack(
        chunks=chunks,
        total_tokens=sum(c.est_tokens for c in chunks),
        budget_tokens=token_budget,
        truncated=truncated,
        mode="wsd",
    )


# -----------------------------
# Helpers (I/O, text, tokens)
# -----------------------------

def _read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        # As a fallback, try binary -> utf-8 decode ignoring errors
        with io.open(p, "rb") as f:
            data = f.read()
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return data.decode("latin-1", errors="ignore")

def _line_count(text: str) -> int:
    # 1-based lines: number of '\n' + 1 (empty file => 1 line)
    return text.count("\n") + 1

def _split_lines(text: str) -> List[str]:
    # 1-based convenience: we'll index with 1..N by padding
    # But here: return actual lines without trailing newline
    return text.splitlines()

def _slice_lines(lines: List[str], s: int, e: int) -> str:
    # s,e inclusive; 1-based
    s0 = max(1, s)
    e0 = max(s0, e)
    # Convert to 0-based slices:
    return "\n".join(lines[s0 - 1:e0]) + "\n"

def _est_tokens(text: str) -> int:
    # Simple heuristic: 1 token ~ 4 chars
    return max(1, len(text) // 4)

def _sorted_unique_files_from_plan(plan) -> List[str]:
    files = sorted({t.path for t in getattr(plan, "targets", [])})
    return files

def _merge_windows(wins: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not wins:
        return wins
    merged = []
    cur_s, cur_e = wins[0]
    for s, e in wins[1:]:
        if s <= cur_e + 1:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged

def _expand_window(s: int, e: int, r: int, line_total: int) -> Tuple[int, int]:
    return max(1, s - r), min(line_total, e + r)


# -----------------------------
# Skeleton (Python)
# -----------------------------

_PY_KEYWORDS = set((
    "False","None","True","and","as","assert","async","await","break","class","continue",
    "def","del","elif","else","except","finally","for","from","global","if","import",
    "in","is","lambda","nonlocal","not","or","pass","raise","return","try","while","with","yield"
))

def _python_skeleton(abs_path: Path, file_text: str) -> List[str]:
    # Only for Python files
    if abs_path.suffix != ".py":
        return []
    try:
        tree = ast.parse(file_text, filename=str(abs_path))
    except Exception:
        return []

    lines: List[str] = []
    class V(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef):
            bases = ", ".join(_safe_unparse(b) for b in node.bases) if node.bases else ""
            if bases:
                lines.append(f"L{node.lineno:>4}  class {node.name}({bases})")
            else:
                lines.append(f"L{node.lineno:>4}  class {node.name}")
            for b in node.body:
                if isinstance(b, ast.FunctionDef):
                    sig = _fmt_args(b.args)
                    lines.append(f"L{b.lineno:>4}      def {b.name}({sig})")
                elif isinstance(b, ast.AsyncFunctionDef):
                    sig = _fmt_args(b.args)
                    lines.append(f"L{b.lineno:>4}      async def {b.name}({sig})")

        def visit_FunctionDef(self, node: ast.FunctionDef):
            sig = _fmt_args(node.args)
            lines.append(f"L{node.lineno:>4}  def {node.name}({sig})")

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
            sig = _fmt_args(node.args)
            lines.append(f"L{node.lineno:>4}  async def {node.name}({sig})")

    V().visit(tree)
    return lines

def _safe_unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)  # py3.9+
    except Exception:
        return getattr(node, "id", "<expr>")

def _fmt_args(args: ast.arguments) -> str:
    parts: List[str] = []
    pos = [a.arg for a in getattr(args, "posonlyargs", [])] + [a.arg for a in args.args]
    defaults = list(args.defaults)
    n_no_default = len(pos) - len(defaults)
    for i, name in enumerate(pos):
        if i < n_no_default:
            parts.append(name)
        else:
            dv = _safe_unparse(defaults[i - n_no_default]) if defaults else "…"
            parts.append(f"{name}={dv}")
    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    for a, dv in zip(args.kwonlyargs, args.kw_defaults):
        parts.append(f"{a.arg}" if dv is None else f"{a.arg}={_safe_unparse(dv)}")
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")
    return ", ".join(parts)


# -----------------------------
# Identifier & Def finding
# -----------------------------

_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")

def _extract_identifiers(text: str) -> List[str]:
    # crude; filters out keywords and very short names
    names = _IDENT_RE.findall(text)
    out = []
    for n in names:
        if n in _PY_KEYWORDS:
            continue
        if len(n) < 3 and n not in ("os", "re", "np", "pd"):  # allow some common short aliases
            continue
        out.append(n)
    # de-dup keeping order
    seen = set()
    uniq = []
    for n in out:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq

# simplistic "defs in file": class/def and top-level constants NAME = ...
_CLASS_DEF_RE = re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\s*(\(|:)", re.M)
_FUNC_DEF_RE  = re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.M)
_CONST_DEF_RE = re.compile(r"^\s*([A-Z][A-Z0-9_]+)\s*=", re.M)

def _extract_local_defs(file_text: str) -> Set[str]:
    out = set(m.group(1) for m in _CLASS_DEF_RE.finditer(file_text))
    out.update(m.group(1) for m in _FUNC_DEF_RE.finditer(file_text))
    out.update(m.group(1) for m in _CONST_DEF_RE.finditer(file_text))
    return out

def _find_symbol_defs_in_file(lines: List[str], names: List[str], limit_per_file: int) -> List[Tuple[str, Tuple[int,int]]]:
    """
    Return list of (name, (start_line, end_line)) for simple defs in the same file.
    end_line is a heuristic: find the next blank line or next def/class.
    """
    text = "\n".join(lines) + "\n"
    hits: List[Tuple[str, Tuple[int, int]]] = []
    patterns = [
        (lambda n: re.compile(rf"^\s*def\s+{re.escape(n)}\s*\(", re.M)),
        (lambda n: re.compile(rf"^\s*class\s+{re.escape(n)}\s*(\(|:)", re.M)),
        (lambda n: re.compile(rf"^\s*{re.escape(n)}\s*=", re.M)),
    ]
    used = set()
    for n in names:
        if len(hits) >= limit_per_file:
            break
        for mk in patterns:
            m = mk(n).search(text)
            if m:
                start = _line_no_of_pos(text, m.start())
                end = _heuristic_block_end(lines, start)
                if (n, (start, end)) not in used:
                    used.add((n, (start, end)))
                    hits.append((n, (start, end)))
                    break
    return hits

def _line_no_of_pos(text: str, pos: int) -> int:
    # number of '\n' before pos + 1
    return text.count("\n", 0, pos) + 1

def _heuristic_block_end(lines: List[str], start_line: int) -> int:
    """
    Find a reasonable end for a def/class/const snippet:
    - stop at next def/class
    - or at next "obvious separator" blank line after a few lines
    - otherwise cap at start_line + 20
    """
    n = len(lines)
    limit = min(n, start_line + 20)
    pat_def_or_class = re.compile(r"^\s*(def|class)\s+")
    # search from next line
    for i in range(start_line, min(n, start_line + 200)):
        line = lines[i - 1]
        if i > start_line and pat_def_or_class.match(line):
            return i - 1
        if i > start_line + 3 and line.strip() == "":
            # stop at the first blank line after at least 3 lines of content
            return i
        if i >= limit:
            return i
    return min(n, start_line + 20)

def _search_defs_in_text(file_text: str, names: Set[str], limit: int = 2) -> List[Tuple[str, Tuple[int,int]]]:
    """
    Search in neighbor file for any of 'names'.
    Returns up to 'limit' hits total across names.
    """
    lines = file_text.splitlines()
    hits: List[Tuple[str, Tuple[int,int]]] = []
    used = set()
    for n in names:
        if len(hits) >= limit:
            break
        # prefer def/class; then constant
        m = re.search(rf"^\s*def\s+{re.escape(n)}\s*\(", file_text, flags=re.M)
        if not m:
            m = re.search(rf"^\s*class\s+{re.escape(n)}\s*(\(|:)", file_text, flags=re.M)
        if not m:
            m = re.search(rf"^\s*{re.escape(n)}\s*=", file_text, flags=re.M)
        if m:
            s = _line_no_of_pos(file_text, m.start())
            e = _heuristic_block_end(lines, s)
            if (n, (s, e)) not in used:
                used.add((n, (s, e)))
                hits.append((n, (s, e)))
    return hits


# -----------------------------
# Trimming to budget
# -----------------------------

def _dedup_chunks(chunks: List[DocChunk]) -> List[DocChunk]:
    seen = set()
    out: List[DocChunk] = []
    for c in chunks:
        key = (c.path, c.kind, c.start, c.end, hash(c.text))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out

def _trim_to_budget(chunks: List[DocChunk], budget: int) -> Tuple[int, List[DocChunk]]:
    """
    Trim chunks by category order: dep_cross -> dep_local -> skeleton -> window tail.
    Returns (new_total_tokens, new_chunks)
    """
    # Group indices by kind
    idx_by_kind: Dict[str, List[int]] = {"dep_cross": [], "dep_local": [], "skeleton": [], "window": []}
    for i, c in enumerate(chunks):
        if c.kind in idx_by_kind:
            idx_by_kind[c.kind].append(i)

    total = sum(c.est_tokens for c in chunks)
    if total <= budget:
        return total, chunks

    # Helper to drop whole chunks of a given kind from the end
    def drop_whole(kind: str):
        nonlocal total, chunks
        for i in reversed(idx_by_kind.get(kind, [])):
            if total <= budget:
                break
            total -= chunks[i].est_tokens
            chunks[i].text = ""           # mark dropped
            chunks[i].est_tokens = 0
        # purge dropped
        chunks = [c for c in chunks if c.text != ""]
        # rebuild indices
        _rebuild_idx()

    # For window tail trimming: reduce trailing lines while keeping [start,end] intact
    def trim_window_tails():
        nonlocal total, chunks
        windows = [c for c in chunks if c.kind == "window"]
        # Sort by longest first (more to trim)
        windows.sort(key=lambda c: c.est_tokens, reverse=True)
        for c in windows:
            if total <= budget:
                break
            # Keep at least the target core region (we don't know exact core; keep 2/3)
            lines = c.text.splitlines()
            if len(lines) <= 8:
                continue
            # trim 20% of the tail each iteration (at least 4 lines)
            trim_n = max(4, int(len(lines) * 0.2))
            new_lines = lines[: max(1, len(lines) - trim_n)]
            new_text = "\n".join(new_lines) + "\n"
            delta = _est_tokens(c.text) - _est_tokens(new_text)
            if delta <= 0:
                continue
            c.text = new_text
            c.end = c.start + len(new_lines) - 1
            c.est_tokens = _est_tokens(new_text)
            total -= delta

    def _rebuild_idx():
        idx_by_kind.clear()
        idx_by_kind.update({"dep_cross": [], "dep_local": [], "skeleton": [], "window": []})
        for i, cc in enumerate(chunks):
            if cc.kind in idx_by_kind:
                idx_by_kind[cc.kind].append(i)

    # 1) Drop dep_cross
    drop_whole("dep_cross")
    if total <= budget:
        return total, chunks

    # 2) Drop dep_local
    drop_whole("dep_local")
    if total <= budget:
        return total, chunks

    # 3) Drop skeleton
    drop_whole("skeleton")
    if total <= budget:
        return total, chunks

    # 4) Trim window tails softly
    # (iterate a few passes to avoid over-trimming a single window)
    for _ in range(6):
        if total <= budget:
            break
        trim_window_tails()

    return total, chunks
