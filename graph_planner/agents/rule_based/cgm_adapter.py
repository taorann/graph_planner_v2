# -*- coding: utf-8 -*-
from __future__ import annotations
"""
agents/rule_based/cgm_adapter.py

规则驱动的补丁生成器：
- generate(subgraph_linearized, plan, constraints, snippets) -> Patch
  根据 PlannerEnv 返回的片段数据，对每个 PlanTarget 的末行附加注释标记，
  以保证补丁在 Guard/测试流程中的可见性，同时避免依赖容器外的文件系统。
"""

from typing import Any, Dict, List, Optional
import os

from aci.schema import Patch, PatchEdit, Plan, PlanTarget
from ...infra import telemetry
from ...infra.config import load as load_config
from ...integrations.codefuse_cgm import CodeFuseCGMClient

_MARKER = "CGM-LOCAL"


# ---------------- helpers ----------------

def _detect_comment_prefix(path: str) -> str:
    """根据扩展名返回单行注释前缀；未知类型用 '#'."""

    ext = os.path.splitext(path)[1].lower()
    if ext in {".py", ".rb", ".pl"}:
        return "#"
    if ext in {".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".h", ".hpp", ".cpp", ".cc", ".go", ".rs", ".kt", ".scala"}:
        return "//"
    if ext in {".sh", ".bash", ".zsh", ".fish", ".dockerfile"}:
        return "#"
    if ext in {".sql"}:
        return "--"
    if ext in {".toml", ".ini", ".cfg"}:
        return "#"
    if ext in {".yaml", ".yml"}:
        return "#"
    if ext in {".md", ".rst"}:
        return "<!--"
    return "#"


def _append_marker_line(line: str, path: str) -> str:
    """在一行尾部追加标记注释，保持原缩进。"""

    pref = _detect_comment_prefix(path)
    if pref == "<!--":
        suffix = f" <!-- {_MARKER} -->"
    else:
        suffix = f" {pref} {_MARKER}"
    return line.rstrip("\r\n") + suffix


def _parse_snippet_lines(snippet: Dict[str, Any]) -> Dict[int, str]:
    lines: Dict[int, str] = {}
    for raw in snippet.get("snippet") or []:
        if ":" not in raw:
            continue
        prefix, remainder = raw.split(":", 1)
        try:
            idx = int(prefix)
        except ValueError:
            continue
        text = remainder[1:] if remainder.startswith(" ") else remainder
        lines[idx] = text.rstrip("\r\n")
    return lines


def _index_snippets(snippets: Optional[List[Dict[str, Any]]]) -> Dict[str, Dict[int, str]]:
    index: Dict[str, Dict[int, str]] = {}
    for snip in snippets or []:
        path = snip.get("path")
        if not path:
            continue
        lines = _parse_snippet_lines(snip)
        if not lines:
            continue
        index.setdefault(path, {}).update(lines)
        abs_path = snip.get("abs_path")
        if abs_path:
            index.setdefault(abs_path, {}).update(lines)
    return index


def _read_line_from_fs(path: str, lineno: int) -> Optional[str]:
    abspath = path if os.path.isabs(path) else os.path.join(os.getcwd(), path)
    try:
        with open(abspath, "r", encoding="utf-8", errors="ignore") as f:
            for idx, line in enumerate(f, 1):
                if idx == lineno:
                    return line.rstrip("\r\n")
    except Exception:
        return None
    return None


def _build_single_edit_for_target(
    target: PlanTarget,
    snippet_index: Dict[str, Dict[int, str]],
) -> Optional[PatchEdit]:
    path = target.path
    line_no = int(getattr(target, "end", 1))
    line_map = snippet_index.get(path) or snippet_index.get(os.path.abspath(path))
    line_text: Optional[str] = None
    if line_map:
        line_text = line_map.get(line_no) or line_map.get(int(getattr(target, "start", line_no)))
    if not line_text:
        line_text = _read_line_from_fs(path, line_no)
    if not line_text:
        return None
    if _MARKER in line_text:
        return None
    new_line = _append_marker_line(line_text, path) + "\n"
    return PatchEdit(path=path, start=line_no, end=line_no, new_text=new_line)


def _summarize_patch(edits: List[PatchEdit]) -> str:
    if not edits:
        return "local-cgm: no-op (no eligible lines to edit)"
    paths = sorted({e["path"] for e in edits})
    return f"local-cgm: edits={len(edits)} files={len(paths)} marker={_MARKER}"


# ---------------- main API ----------------

_CLIENT_CACHE: Optional[CodeFuseCGMClient] = None
_CLIENT_FINGERPRINT: Optional[tuple] = None


def _get_client() -> Optional[CodeFuseCGMClient]:
    global _CLIENT_CACHE, _CLIENT_FINGERPRINT
    cfg = load_config()
    cgm_cfg = getattr(cfg, "cgm", None)
    if not cgm_cfg or not getattr(cgm_cfg, "enabled", False):
        return None
    endpoint = getattr(cgm_cfg, "endpoint", None)
    if not endpoint:
        return None
    api_key = None
    api_key_env = getattr(cgm_cfg, "api_key_env", None)
    if api_key_env:
        api_key = os.environ.get(api_key_env)
    fingerprint = (
        endpoint,
        api_key,
        getattr(cgm_cfg, "model", None),
        getattr(cgm_cfg, "temperature", None),
        getattr(cgm_cfg, "max_tokens", None),
        getattr(cgm_cfg, "timeout_s", 60),
    )
    if _CLIENT_CACHE and _CLIENT_FINGERPRINT == fingerprint:
        return _CLIENT_CACHE
    _CLIENT_CACHE = CodeFuseCGMClient(
        endpoint=endpoint,
        api_key=api_key,
        model=getattr(cgm_cfg, "model", None),
        temperature=getattr(cgm_cfg, "temperature", None),
        max_tokens=getattr(cgm_cfg, "max_tokens", None),
        timeout_s=int(getattr(cgm_cfg, "timeout_s", 60)),
    )
    _CLIENT_FINGERPRINT = fingerprint
    return _CLIENT_CACHE


def _generate_local_patch(
    plan: Plan,
    constraints: Optional[Dict[str, Any]],
    snippets: Optional[List[Dict[str, Any]]],
) -> Patch:
    max_edits = int((constraints or {}).get("max_edits", 3))
    snippet_index = _index_snippets(snippets)
    edits: List[PatchEdit] = []

    for target in plan.targets:
        if len(edits) >= max_edits:
            break
        edit = _build_single_edit_for_target(target, snippet_index)
        if edit:
            edits.append(edit)

    patch: Patch = {"edits": edits, "summary": _summarize_patch(edits)}
    return patch


def generate(
    subgraph_linearized: Optional[List[Dict[str, Any]]],
    plan: Plan,
    constraints: Optional[Dict[str, Any]] = None,
    snippets: Optional[List[Dict[str, Any]]] = None,
    plan_text: Optional[str] = None,
    issue: Optional[Dict[str, Any]] = None,
) -> Patch:
    """生成补丁：优先调用 CodeFuse CGM，失败时回退到本地标记方案。"""

    client = _get_client()
    if client:
        try:
            return client.generate_patch(
                issue=issue,
                plan=plan,
                plan_text=plan_text,
                subgraph_linearized=subgraph_linearized,
                snippets=snippets,
                metadata={"constraints": constraints or {}},
            )
        except Exception as exc:  # pragma: no cover - graceful fallback
            telemetry.log_event(
                {
                    "kind": "cgm",
                    "ok": False,
                    "error": str(exc),
                    "endpoint": getattr(client, "endpoint", ""),
                }
            )

    patch = _generate_local_patch(plan, constraints, snippets)
    if client:
        base_summary = patch.get("summary") or "local-cgm"
        patch["summary"] = f"{base_summary} | fallback"
    return patch
