"""Utilities for the text-trajectory protocol shared by planner agents.

The planner-side model responds with a ``<function=...>`` block for each
decision.  This module provides helpers to parse that block, construct the
payload consumed by CodeFuse-CGM, and drive the repair execution pipeline
including diff validation, application, and observation emission.
"""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional

from ...memory import subgraph_store

__all__ = [
    "ActionParseError",
    "RepairRuntimeState",
    "parse_action_block",
    "format_action_block",
    "build_cgm_payload",
    "call_cgm",
    "pick_best_candidate",
    "validate_unified_diff",
    "try_apply_and_test",
    "emit_observation",
    "handle_planner_repair",
]


class ActionParseError(ValueError):
    """Raised when the planner response does not match the protocol."""


_ACTION_RE = re.compile(r"^<function\s*=\s*([a-zA-Z0-9_.-]+)\s*>", re.IGNORECASE)
_END_RE = re.compile(r"</function>\s*$", re.IGNORECASE | re.DOTALL)
_PARAM_RE = re.compile(r"<param\s+name=\"([^\"]+)\">(.*?)</param>", re.DOTALL | re.IGNORECASE)


def _strip_cdata(value: str) -> str:
    if value.startswith("<![CDATA[") and value.endswith("]]>"):
        return value[9:-3]
    return value


def _normalise_json_value(raw: str) -> Any:
    text = raw.strip()
    if not text:
        return ""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return raw.strip("\r\n")


def parse_action_block(text: str, allowed: Iterable[str]) -> Dict[str, Any]:
    """Parse the planner ``<function=...>`` block into a structured dict.

    Parameters
    ----------
    text:
        Raw planner response containing exactly one ``<function=...>`` block.
    allowed:
        Iterable of valid function names.  A :class:`ValueError` is raised if
        the block name is not within this set.

    Returns
    -------
    dict
        Mapping with keys ``name`` and ``params``.
    """

    if not isinstance(text, str):
        raise ActionParseError("planner response must be a string")
    stripped = text.strip()
    match = _ACTION_RE.match(stripped)
    if not match:
        raise ActionParseError("no <function=...> block found in planner output")
    name = match.group(1).strip().lower()

    allowed_set = {entry.lower() for entry in allowed}
    if allowed_set and name not in allowed_set:
        raise ActionParseError(f"action '{name}' is not allowed; expected one of {sorted(allowed_set)}")

    inner = stripped[match.end() :]
    if not _END_RE.search(inner):
        raise ActionParseError("planner action block must terminate with </function>")
    inner = _END_RE.sub("", inner)

    params: Dict[str, Any] = {}
    last_end = 0
    for match in _PARAM_RE.finditer(inner):
        start, end = match.span()
        gap = inner[last_end:start]
        if gap.strip():
            raise ActionParseError("unexpected content between <param> elements")
        key = match.group(1).strip()
        raw_value = _strip_cdata(match.group(2).strip())
        if key in params:
            raise ActionParseError(f"duplicate parameter '{key}' in planner action")
        params[key] = _normalise_json_value(raw_value)
        last_end = end
    if inner[last_end:].strip():
        raise ActionParseError("unexpected trailing content after last <param>")

    return {"name": name, "params": params}


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


def validate_unified_diff(patch: str, path: str) -> None:
    """Best-effort validation of a unified diff for a single file."""

    if not patch.strip():
        raise ValueError("CGM returned an empty diff")
    lines = patch.splitlines()
    normalized_path = path.strip()
    if not normalized_path:
        raise ValueError("candidate path is empty")

    def _clean(diff_path: str) -> str:
        diff_path = diff_path.strip()
        if diff_path.startswith("a/") or diff_path.startswith("b/"):
            return diff_path[2:]
        return diff_path

    seen_paths: List[str] = []
    hunk_count = 0
    current_old: Optional[str] = None
    current_new: Optional[str] = None
    for line in lines:
        if line.startswith("diff --git"):
            parts = line.split()
            if len(parts) >= 4:
                seen_paths.append(_clean(parts[2]))
                seen_paths.append(_clean(parts[3]))
        elif line.startswith("--- "):
            current_old = _clean(line[4:])
            seen_paths.append(current_old)
        elif line.startswith("+++ "):
            current_new = _clean(line[4:])
            seen_paths.append(current_new)
            if current_old and current_new and current_old != current_new:
                raise ValueError("unified diff changes multiple files")
        elif line.startswith("@@"):
            hunk_count += 1

    unique_paths = {p for p in seen_paths if p}
    if not unique_paths:
        raise ValueError("unable to determine the target file in diff")
    if len(unique_paths) > 1:
        raise ValueError("diff spans multiple files; expected a single target")
    if _clean(normalized_path) not in unique_paths:
        raise ValueError(f"diff targets {unique_paths.pop()} but planner requested {normalized_path}")
    if hunk_count == 0:
        raise ValueError("diff does not contain any hunks")


def _count_hunks(patch: str) -> int:
    return sum(1 for line in patch.splitlines() if line.startswith("@@"))


def try_apply_and_test(path: str, patch: str) -> Dict[str, Any]:
    """Apply the diff, run lint/tests, and summarise the outcome."""

    state = _require_state()
    sandbox = state.sandbox
    applied = sandbox.apply_patch(patch)
    if not applied:
        sandbox.reset_soft()
        return {"ok": False, "error": "apply-failed", "msg": f"failed to apply diff for {path}"}

    lint_ok = bool(sandbox.lint())
    if not lint_ok:
        sandbox.reset_soft()
        return {"ok": False, "error": "lint-failed", "msg": "lint checks failed"}

    tests = sandbox.test()
    tests_passed = bool(tests.get("passed"))
    if not tests_passed:
        sandbox.reset_soft()
        msg = tests.get("stdout") or "tests failed"
        return {"ok": False, "error": "tests-failed", "msg": msg, "tests": tests}

    return {
        "ok": True,
        "applied": True,
        "path": path,
        "hunks": _count_hunks(patch),
        "tests_passed": tests_passed,
        "lint_ok": lint_ok,
        "tests": tests,
    }


def emit_observation(name: str, data: Mapping[str, Any]) -> str:
    """Wrap the observation payload in the trajectory protocol envelope."""

    serialised = json.dumps(data, ensure_ascii=False)
    return f'<observation for="{name}">{serialised}</observation>'


def handle_planner_repair(action: Mapping[str, Any], state: RepairRuntimeState) -> Dict[str, Any]:
    """Execute a repair cycle given planner parameters and runtime state."""

    params = action.get("params") if "params" in action else action
    name = action.get("name") if isinstance(action, Mapping) else None
    if name and str(name).lower() != "repair":
        raise ValueError("handle_planner_repair expects a repair action")

    subplan = params.get("subplan") if isinstance(params, Mapping) else None
    if not isinstance(subplan, str) or not subplan.strip():
        raise ValueError("repair action requires a non-empty 'subplan'")
    focus_ids = params.get("focus_ids") if isinstance(params, Mapping) else None
    if not isinstance(focus_ids, list):
        focus_ids = []
    apply_flag = params.get("apply", True) if isinstance(params, Mapping) else True
    apply_bool = bool(apply_flag)

    payload = build_cgm_payload(state, subplan, focus_ids)
    with _bind_state(state):
        candidates = call_cgm(payload, k=state.cgm_top_k)
    if not candidates:
        return {"ok": False, "error": "cgm-empty", "msg": "CGM returned no candidates"}
    candidate = pick_best_candidate(candidates)
    patch = str(candidate.get("patch") or "")
    patch_path = str(candidate.get("path") or "")
    confidence = float(candidate.get("confidence", 0.0))

    try:
        validate_unified_diff(patch, patch_path)
    except ValueError as exc:
        return {"ok": False, "error": "invalid-diff", "msg": str(exc), "confidence": confidence}

    if not apply_bool:
        return {
            "ok": True,
            "applied": False,
            "candidate": candidate,
            "confidence": confidence,
        }

    with _bind_state(state):
        result = try_apply_and_test(patch_path, patch)
    result.setdefault("confidence", confidence)
    if result.get("ok"):
        result.setdefault("msg", candidate.get("rationale", ""))
        result.setdefault("candidate", candidate)
    else:
        result.setdefault("candidate", candidate)
    return result

