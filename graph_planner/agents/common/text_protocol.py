"""Utilities for the text-trajectory protocol shared by planner agents."""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from difflib import unified_diff
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional

from ...memory import subgraph_store
from ...runtime.sandbox import PatchApplier
from ...core.patches import patch_id
from .contracts import (
    CGM_CONTRACT,
    CGMPatch,
    CGMPatchErrorCode,
    ProtocolError,
    normalize_newlines,
    parse_action_block,
    validate_cgm_patch,
    validate_planner_action,
)

__all__ = [
    "RepairRuntimeState",
    "format_action_block",
    "build_cgm_payload",
    "call_cgm",
    "pick_best_candidate",
    "validate_unified_diff",
    "try_apply_and_test",
    "emit_observation",
    "handle_planner_repair",
]


def format_action_block(name: str, params: Mapping[str, Any]) -> str:
    """Render a ``<function=...>`` block from structured parameters."""

    lines = [f"<function={name}>"]
    for key, value in params.items():
        lines.append(f"  <param name=\"{key}\">{_format_param_value(value)}</param>")
    lines.append("</function>")
    return "\n".join(lines)


def _format_param_value(value: Any) -> str:
    if isinstance(value, str):
        if not value:
            return ""
        if any(ch in value for ch in "<>\n"):
            return "<![CDATA[\n" + value + "\n]]>"
        return value
    return json.dumps(value, ensure_ascii=False)


def _canonicalise_step(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return ""
    return re.sub(r"^(?:\d+\s*[\).:-]\s*|[-*]\s*)", "", stripped)


def _truncate(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


@dataclass
class RepairRuntimeState:
    """Context required to execute a CGM repair cycle."""

    issue: Dict[str, Any]
    subgraph: subgraph_store.WorkingSubgraph
    sandbox: Any
    repo_root: str
    text_memory: Mapping[str, str]
    related_files: MutableMapping[str, str]
    default_focus_ids: List[str]
    cgm_generate: Callable[[Dict[str, Any], int], List[Dict[str, Any]]]
    snippets: List[Dict[str, Any]] = field(default_factory=list)
    token_budget: int = 4096
    max_graph_nodes: int = 64
    max_files: int = 4
    max_file_bytes: int = 20000
    cgm_top_k: int = 1
    constraints: Mapping[str, Any] = field(
        default_factory=lambda: {
            "patch_style": "unified_diff",
            "one_file_per_patch": True,
            "tests_required": False,
        }
    )

    def sorted_nodes(self) -> List[Dict[str, Any]]:
        nodes = list(self.subgraph.nodes.values())
        nodes.sort(key=lambda n: float(n.get("score", 0.0)), reverse=True)
        return nodes

    def focus_or_default(self, focus_ids: Iterable[str]) -> List[str]:
        focus = [fid for fid in focus_ids if fid]
        if focus:
            return focus
        return list(self.default_focus_ids)

    def compact_graph(self, focus_ids: Iterable[str]) -> Dict[str, Any]:
        nodes = []
        edges = list(self.subgraph.edges or [])
        for node in self.sorted_nodes()[: self.max_graph_nodes]:
            compact = {k: node.get(k) for k in ("id", "kind", "path", "span", "summary", "score")}
            nodes.append({k: v for k, v in compact.items() if v is not None})
        return {"nodes": nodes, "edges": edges, "focus_ids": self.focus_or_default(focus_ids)}

    def limited_files(self) -> Dict[str, str]:
        limited: Dict[str, str] = {}
        for idx, (path, content) in enumerate(self.related_files.items()):
            if idx >= self.max_files:
                break
            limited[path] = _truncate(content, self.max_file_bytes)
        return limited

    def generate_candidates(self, payload: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
        return self.cgm_generate(payload, k)


@dataclass(frozen=True)
class DiffAnalysis:
    """Summary of validated diff statistics for telemetry."""

    new_text: str
    n_hunks: int
    added_lines: int
    removed_lines: int


_STATE: ContextVar[Optional[RepairRuntimeState]] = ContextVar("gp_repair_state", default=None)


@contextmanager
def _bind_state(state: RepairRuntimeState):
    token = _STATE.set(state)
    try:
        yield
    finally:
        _STATE.reset(token)


def _require_state() -> RepairRuntimeState:
    state = _STATE.get()
    if state is None:
        raise RuntimeError("repair runtime state not initialised")
    return state


def build_cgm_payload(state: RepairRuntimeState, subplan: str, focus_ids: Iterable[str]) -> Dict[str, Any]:
    """Construct the payload sent to CodeFuse-CGM."""

    steps = [step for step in (_canonicalise_step(line) for line in subplan.splitlines()) if step]
    graph = state.compact_graph(focus_ids)
    memory = {
        "session_summary": state.text_memory.get("session_summary", ""),
        "turn_notes": state.text_memory.get("turn_notes", ""),
    }
    files = state.limited_files()
    issue = {
        "title": state.issue.get("title", ""),
        "body": state.issue.get("body") or state.issue.get("description", ""),
    }
    payload = {
        "graph": graph,
        "text_memory": memory,
        "files": files,
        "issue": issue,
        "plan": steps,
        "constraints": dict(state.constraints),
        "plan_text": subplan.strip(),
    }
    return payload


def call_cgm(payload: Dict[str, Any], k: int = 1) -> List[Dict[str, Any]]:
    """Request ``k`` patch candidates from the configured CGM backend."""

    state = _require_state()
    candidates = state.generate_candidates(payload, k)
    return candidates[:k]


def pick_best_candidate(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return the candidate with the highest confidence score."""

    if not candidates:
        raise ValueError("no CGM candidates available")
    return max(candidates, key=lambda c: float(c.get("confidence", 0.0)))


def _clean_diff_path(diff_path: str) -> str:
    diff_path = diff_path.strip()
    if diff_path.startswith("a/") or diff_path.startswith("b/"):
        return diff_path[2:]
    return diff_path


def _read_original_text(state: RepairRuntimeState, path: str) -> str:
    if path in state.related_files:
        return normalize_newlines(state.related_files[path])

    base = Path(state.repo_root) if state.repo_root else Path.cwd()
    target = base.joinpath(path)
    try:
        data = target.read_bytes()
    except FileNotFoundError as exc:
        raise ProtocolError(
            CGMPatchErrorCode.PATH_MISSING.value,
            f"original file '{path}' not found for patch application",
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise ProtocolError(
            CGMPatchErrorCode.INVALID_PATCH_SCHEMA.value,
            f"unable to read '{path}': {exc}",
        ) from exc

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        error = ProtocolError(
            CGMPatchErrorCode.ENCODING_UNSUPPORTED.value,
            f"file '{path}' is not UTF-8 encoded: {exc}",
        )
        error.__cause__ = exc
        raise error
    return normalize_newlines(text)


def _apply_patch_edits(original_text: str, edits: List[Mapping[str, Any]], path: str) -> str:
    if not edits:
        raise ProtocolError(CGMPatchErrorCode.INVALID_PATCH_SCHEMA.value, "no edits supplied")

    original_lines = original_text.splitlines(keepends=True)
    new_lines = list(original_lines)

    for edit in sorted(edits, key=lambda e: (int(e.get("start", 1)), int(e.get("end", 1)))):
        try:
            start = int(edit["start"])
            end = int(edit["end"])
        except Exception as exc:  # pragma: no cover - validated upstream
            raise ProtocolError(CGMPatchErrorCode.RANGE_INVALID.value, f"invalid range in edit for {path}") from exc
        if start < 1:
            raise ProtocolError(CGMPatchErrorCode.RANGE_INVALID.value, f"edit start must be >=1 for {path}")
        if end < start:
            raise ProtocolError(CGMPatchErrorCode.RANGE_INVALID.value, f"edit end must be >= start for {path}")
        if start - 1 > len(new_lines):
            raise ProtocolError(CGMPatchErrorCode.RANGE_INVALID.value, f"edit start {start} exceeds file length for {path}")
        if end > len(new_lines) + 1:
            raise ProtocolError(CGMPatchErrorCode.RANGE_INVALID.value, f"edit end {end} exceeds file length for {path}")
        replacement = normalize_newlines(str(edit["new_text"]))
        replacement_lines = replacement.splitlines(keepends=True)
        new_lines[start - 1 : end] = replacement_lines

    new_text = "".join(new_lines)
    if CGM_CONTRACT.constraints.get("newline_required") and not new_text.endswith("\n"):
        raise ProtocolError(CGMPatchErrorCode.NEWLINE_MISSING.value, f"resulting file '{path}' must end with newline")
    return new_text


def _build_unified_diff(original_text: str, new_text: str, path: str) -> str:
    original_lines = original_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    diff_lines = list(
        unified_diff(
            original_lines,
            new_lines,
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="",
        )
    )
    if not diff_lines:
        raise ProtocolError(
            CGMPatchErrorCode.INVALID_PATCH_SCHEMA.value,
            "CGM patch produced no diff",
        )
    return "\n".join(diff_lines) + "\n"


def _analyse_unified_diff(original_text: str, diff_text: str, path: str) -> DiffAnalysis:
    diff_text = normalize_newlines(diff_text)
    if not diff_text.strip():
        raise ProtocolError(CGMPatchErrorCode.INVALID_UNIFIED_DIFF.value, "diff content is empty")

    lines = diff_text.splitlines()
    header_old: Optional[str] = None
    header_new: Optional[str] = None
    seen_paths: set[str] = set()

    for line in lines:
        if line.startswith("diff --git"):
            parts = line.split()
            if len(parts) >= 4:
                seen_paths.add(_clean_diff_path(parts[2]))
                seen_paths.add(_clean_diff_path(parts[3]))
        elif line.startswith("--- "):
            header_old = _clean_diff_path(line[4:])
            seen_paths.add(header_old)
        elif line.startswith("+++ "):
            header_new = _clean_diff_path(line[4:])
            seen_paths.add(header_new)

    seen_paths.discard("/dev/null")
    if not seen_paths:
        raise ProtocolError(CGMPatchErrorCode.INVALID_UNIFIED_DIFF.value, "unable to determine target file from diff header")
    if len(seen_paths) > 1:
        raise ProtocolError(CGMPatchErrorCode.MULTI_FILE_DIFF.value, "diff spans multiple files")

    target = _clean_diff_path(path)
    header_target = header_new or header_old or next(iter(seen_paths))
    if target != header_target:
        raise ProtocolError(
            CGMPatchErrorCode.INVALID_UNIFIED_DIFF.value,
            f"diff header references '{header_target}' but repair target is '{target}'",
        )

    original_lines = normalize_newlines(original_text).splitlines()
    new_lines: List[str] = []
    current_orig = 0
    n_hunks = 0
    added = 0
    removed = 0
    idx = 0
    line_count = len(lines)

    while idx < line_count:
        line = lines[idx]
        if not line.strip():
            idx += 1
            continue
        if line.startswith("@@ "):
            n_hunks += 1
            hunk_header = line
            try:
                header_parts = line.split()
                old_range = header_parts[1]
                new_range = header_parts[2]
                old_start, old_len = old_range[1:].split(",") if "," in old_range else (old_range[1:], "1")
                new_start, new_len = new_range[1:].split(",") if "," in new_range else (new_range[1:], "1")
                old_start_i = int(old_start)
                old_len_i = int(old_len)
                new_start_i = int(new_start)
                new_len_i = int(new_len)
            except Exception as exc:
                raise ProtocolError(CGMPatchErrorCode.INVALID_UNIFIED_DIFF.value, f"invalid hunk header '{hunk_header}'") from exc

            if old_start_i < 1 or new_start_i < 1 or old_len_i < 0 or new_len_i < 0:
                raise ProtocolError(CGMPatchErrorCode.RANGE_INVALID.value, f"invalid ranges in hunk '{hunk_header}'")
            if old_start_i - 1 < current_orig:
                raise ProtocolError(CGMPatchErrorCode.RANGE_INVALID.value, "overlapping hunks detected")
            if old_start_i - 1 > len(original_lines):
                raise ProtocolError(CGMPatchErrorCode.RANGE_INVALID.value, "hunk starts beyond file length")

            # Append unchanged lines preceding the hunk
            while current_orig < old_start_i - 1 and current_orig < len(original_lines):
                new_lines.append(original_lines[current_orig])
                current_orig += 1

            idx += 1
            consumed_old = 0
            produced_new = 0
            while idx < line_count:
                segment = lines[idx]
                if segment.startswith("@@ ") or segment.startswith("diff --git") or segment.startswith("--- ") or segment.startswith("+++ "):
                    break
                if not segment.strip():
                    idx += 1
                    continue
                if segment.startswith(" "):
                    if current_orig >= len(original_lines):
                        raise ProtocolError(CGMPatchErrorCode.RANGE_INVALID.value, "context extends beyond original file")
                    expected = original_lines[current_orig]
                    if expected != segment[1:]:
                        raise ProtocolError(
                            CGMPatchErrorCode.HUNK_MISMATCH.value,
                            f"context mismatch at line {current_orig + 1}",
                        )
                    new_lines.append(expected)
                    current_orig += 1
                    consumed_old += 1
                    produced_new += 1
                elif segment.startswith("-"):
                    if current_orig >= len(original_lines):
                        raise ProtocolError(CGMPatchErrorCode.RANGE_INVALID.value, "deletion exceeds original file length")
                    expected = original_lines[current_orig]
                    if expected != segment[1:]:
                        raise ProtocolError(
                            CGMPatchErrorCode.HUNK_MISMATCH.value,
                            f"deletion mismatch at line {current_orig + 1}",
                        )
                    current_orig += 1
                    consumed_old += 1
                    removed += 1
                elif segment.startswith("+"):
                    new_lines.append(segment[1:])
                    produced_new += 1
                    added += 1
                elif segment.startswith("\\"):
                    # "\ No newline at end of file" -- ignore
                    pass
                else:
                    raise ProtocolError(CGMPatchErrorCode.INVALID_UNIFIED_DIFF.value, f"unexpected diff line '{segment}'")
                idx += 1

            if consumed_old != old_len_i:
                raise ProtocolError(
                    CGMPatchErrorCode.RANGE_INVALID.value,
                    f"hunk removed {consumed_old} lines but header expected {old_len_i}",
                )
            if produced_new != new_len_i:
                raise ProtocolError(
                    CGMPatchErrorCode.RANGE_INVALID.value,
                    f"hunk added {produced_new} lines but header expected {new_len_i}",
                )
            continue
        idx += 1

    # Append trailing lines untouched by hunks
    new_lines.extend(original_lines[current_orig:])
    new_text = "\n".join(new_lines) + "\n"
    if CGM_CONTRACT.constraints.get("newline_required") and not new_text.endswith("\n"):
        raise ProtocolError(CGMPatchErrorCode.NEWLINE_MISSING.value, f"resulting file '{path}' must end with newline")

    return DiffAnalysis(new_text=new_text, n_hunks=n_hunks, added_lines=added, removed_lines=removed)


def validate_unified_diff(patch_text: str, path: str) -> DiffAnalysis:
    """Validate a unified diff against the current working tree."""

    state = _require_state()
    original_text = _read_original_text(state, path)
    return _analyse_unified_diff(original_text, patch_text, path)


def _wrap_runner(obj: Any, attr: str) -> Optional[Callable[[Path], Any]]:
    fn = getattr(obj, attr, None)
    if not callable(fn):
        return None

    def _runner(temp_dir: Path) -> Any:
        try:
            return fn(temp_dir=temp_dir)
        except TypeError:
            try:
                return fn(temp_dir)
            except TypeError:
                return fn()

    return _runner


def _normalise_tests(result: Any) -> Dict[str, Any]:
    if isinstance(result, Mapping):
        data = dict(result)
        data.setdefault("passed", bool(data.get("passed") or data.get("ok")))
        return data
    return {"passed": bool(result), "stdout": ""}


def _normalise_lint(result: Any) -> Dict[str, Any]:
    if isinstance(result, Mapping):
        data = dict(result)
        data.setdefault("ok", bool(data.get("ok") or data.get("passed")))
        return data
    return {"ok": bool(result), "stdout": ""}


def try_apply_and_test(
    path: str,
    patch: str,
    *,
    new_content: Optional[str] = None,
    stats: Optional[Mapping[str, int]] = None,
    patch_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """Apply a validated diff via :class:`PatchApplier` and execute checks."""

    state = _require_state()
    repo_root = Path(state.repo_root or ".").resolve()
    if not getattr(state, "_patch_applier", None):
        state._patch_applier = PatchApplier()  # type: ignore[attr-defined]
    applier: PatchApplier = state._patch_applier  # type: ignore[attr-defined]

    run_tests = _wrap_runner(state.sandbox, "test")
    run_lint = _wrap_runner(state.sandbox, "lint")

    def _tests(temp_dir: Path) -> Dict[str, Any]:
        if run_tests is None:
            return {"passed": True, "stdout": ""}
        return _normalise_tests(run_tests(temp_dir))

    def _lint(temp_dir: Path) -> Dict[str, Any]:
        if run_lint is None:
            return {"ok": True, "stdout": ""}
        return _normalise_lint(run_lint(temp_dir))

    result = applier.apply_in_temp_then_commit(
        repo_root,
        patch,
        path,
        run_tests=_tests,
        run_lint=_lint,
        patch_id=patch_hash,
        new_content=new_content,
        stats=stats,
    )
    return result


def emit_observation(name: str, data: Mapping[str, Any]) -> str:
    """Wrap the observation payload in the trajectory protocol envelope."""

    serialised = json.dumps(data, ensure_ascii=False)
    return f'<observation for="{name}">{serialised}</observation>'


def _focus_ids_from_params(params: Mapping[str, Any] | None) -> List[str]:
    if not isinstance(params, Mapping):
        return []
    focus = params.get("focus_ids")
    if isinstance(focus, list):
        return [str(item) for item in focus if str(item)]
    if isinstance(focus, (str, int, float)):
        text = str(focus)
        return [text] if text else []
    return []


def _prepare_patch(state: RepairRuntimeState, patch: CGMPatch) -> tuple[str, DiffAnalysis, str]:
    original_text = _read_original_text(state, patch.path)
    new_text = _apply_patch_edits(original_text, patch.edits, patch.path)
    diff_text = _build_unified_diff(original_text, new_text, patch.path)
    analysis = _analyse_unified_diff(original_text, diff_text, patch.path)
    return diff_text, analysis, new_text


def handle_planner_repair(action: Mapping[str, Any] | str, state: RepairRuntimeState) -> Dict[str, Any]:
    """Execute a repair cycle given planner parameters and runtime state."""

    if isinstance(action, str):
        raw_block = action
        raw_params: Mapping[str, Any] = {}
    else:
        if "raw" in action:
            raw_block = str(action.get("raw") or "")
        else:
            params_obj = action.get("params") if isinstance(action, Mapping) else action
            raw_params = params_obj if isinstance(params_obj, Mapping) else {}
            raw_block = format_action_block("repair", raw_params)
        raw_params = action.get("params") if isinstance(action, Mapping) and isinstance(action.get("params"), Mapping) else {}
    parsed = parse_action_block(raw_block)
    repair_action = validate_planner_action(parsed)
    if repair_action.type != "repair":
        raise ProtocolError("unknown-action", "handle_planner_repair expected a repair action")

    subplan_text = (repair_action.plan or "").strip()
    focus_ids = _focus_ids_from_params(parsed.get("params"))
    if not focus_ids:
        focus_ids = [str(t.get("id")) for t in repair_action.plan_targets or [] if t.get("id")]

    payload = build_cgm_payload(state, subplan_text, focus_ids)
    with _bind_state(state):
        candidates = call_cgm(payload, k=state.cgm_top_k)
    if not candidates:
        return {"ok": False, "error": "cgm-empty", "msg": "CGM returned no candidates"}

    candidate = pick_best_candidate(candidates)
    confidence = float(candidate.get("confidence", 0.0))

    try:
        patch = validate_cgm_patch(candidate)
    except ProtocolError as exc:
        return {"ok": False, "error": exc.code, "msg": exc.detail, "confidence": confidence, "fallback_reason": exc.code}

    analysis: Optional[DiffAnalysis] = None
    diff_text = ""
    new_text = ""
    try:
        diff_text, analysis, new_text = _prepare_patch(state, patch)
    except ProtocolError as exc:
        return {
            "ok": False,
            "error": exc.code,
            "msg": exc.detail,
            "confidence": confidence,
            "fallback_reason": exc.code,
            "patch_id": patch_id(candidate),
            "n_hunks": analysis.n_hunks if analysis else 0,
            "added_lines": analysis.added_lines if analysis else 0,
            "removed_lines": analysis.removed_lines if analysis else 0,
        }

    patch_hash = patch_id(candidate)
    telemetry = {
        "patch_id": patch_hash,
        "n_hunks": analysis.n_hunks,
        "added_lines": analysis.added_lines,
        "removed_lines": analysis.removed_lines,
    }

    enhanced_candidate = dict(candidate)
    enhanced_candidate.setdefault("patch", {"edits": patch.edits, "summary": patch.summary})
    enhanced_candidate["diff"] = diff_text
    enhanced_candidate.setdefault("path", patch.path)

    if not repair_action.apply:
        return {
            "ok": True,
            "applied": False,
            "candidate": enhanced_candidate,
            "confidence": confidence,
            **telemetry,
        }

    try:
        with _bind_state(state):
            result = try_apply_and_test(
                patch.path,
                diff_text,
                new_content=new_text,
                stats={
                    "n_hunks": analysis.n_hunks,
                    "added_lines": analysis.added_lines,
                    "removed_lines": analysis.removed_lines,
                },
                patch_hash=patch_hash,
            )
    except ProtocolError as exc:
        failure = {
            "ok": False,
            "error": exc.code,
            "msg": exc.detail,
            "confidence": confidence,
            "fallback_reason": exc.code,
            **telemetry,
        }
        if hasattr(state.sandbox, "reset_soft"):
            try:
                state.sandbox.reset_soft()
            except Exception:  # pragma: no cover - defensive
                pass
        temp_path = getattr(exc, "temp_path", "")
        if temp_path:
            failure["temp_path"] = temp_path
        return failure

    result.setdefault("confidence", confidence)
    result.setdefault("candidate", enhanced_candidate)
    result.setdefault("msg", candidate.get("rationale", ""))
    result.update({k: v for k, v in telemetry.items() if k not in result})
    result.setdefault("temp_path", result.get("temp_path", ""))
    result["applied"] = bool(result.get("applied"))
    return result
