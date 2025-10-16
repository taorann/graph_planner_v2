"""Shared helpers for model-driven planner agents."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

from core.actions import (
    ActionUnion,
    ExploreAction,
    MemoryAction,
    RepairAction,
    SubmitAction,
)

SYSTEM_PROMPT = (
    "You are the Graph Planner decision model.\n"
    "You operate on a code graph derived from a software repository.\n"
    "For every observation produce a JSON object with the keys 'thought' and 'action'.\n"
    "The 'action' object must contain a 'type' field (explore, memory, repair, submit).\n"
    "For explore you may also set 'op' (find|expand|read), 'anchors', 'nodes', 'hop', and 'limit'.\n"
    "For memory provide 'ops' describing memory operations.\n"
    "For repair include 'apply', 'plan', 'plan_targets', and optionally 'patch'.\n"
    "Always respond with valid JSON (optionally inside ```json fences)."
)

JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})```", re.DOTALL)
FALLBACK_REASON_KEY = "fallback_reason"


def summarise_observation(
    obs: Dict[str, Any],
    reward: float,
    done: bool,
    info: Dict[str, Any] | None,
) -> Tuple[str, Dict[str, Any]]:
    issue = obs.get("issue") or {}
    steps = obs.get("steps", 0)
    last_info = obs.get("last_info") or {}
    pack = obs.get("observation_pack") or {}

    lines = [
        f"Issue: {issue.get('id', 'unknown')} | step={steps} | reward={reward} | done={done}",
    ]
    if issue.get("title"):
        lines.append(f"Title: {issue['title']}")
    if issue.get("body"):
        lines.append(f"Body: {issue['body'][:240]}")
    if pack.get("failure_frame"):
        ff = pack["failure_frame"]
        file_hint = ff.get("path") or ff.get("file")
        if file_hint:
            lines.append(f"Failure frame: {file_hint}:{ff.get('lineno')}")
    if pack.get("subgraph_stats"):
        stats = pack["subgraph_stats"]
        lines.append(f"Subgraph stats: nodes={stats.get('nodes')} edges={stats.get('edges')}")

    kind = last_info.get("kind")
    if kind:
        lines.append(f"Last op: {kind}")
    if kind == "explore" and last_info.get("op") == "expand":
        cands = last_info.get("candidates") or []
        if cands:
            lines.append("Top candidates:\n" + _format_candidates(cands))
    if kind == "explore" and last_info.get("op") == "read":
        snippets = last_info.get("snippets") or []
        for snip in snippets[:2]:
            snippet_lines = " | ".join((snip.get("snippet") or [])[:2])
            lines.append(
                f"Snippet {snip.get('path')}@{snip.get('start')}->{snip.get('end')}: {snippet_lines}"
            )
    if kind == "repair":
        lines.append(f"Patch applied: {last_info.get('applied')}")
        if last_info.get("lint"):
            lines.append(f"Lint rc={last_info['lint'].get('rc')}")
        if last_info.get("tests"):
            lines.append(f"Tests passed={last_info['tests'].get('passed')}")

    summary = "\n".join(lines)
    metadata = {
        "issue": issue,
        "steps": steps,
        "last_info": last_info,
        "reward": reward,
        "done": done,
        "info": info or {},
    }
    return summary, metadata


def _format_candidates(candidates: List[Dict[str, Any]], *, limit: int = 3) -> str:
    rows = []
    for cand in candidates[:limit]:
        path = cand.get("path") or "?"
        span = cand.get("span") or {}
        start = span.get("start")
        end = span.get("end")
        score = cand.get("score")
        rows.append(f"{path}:{start}-{end} (score={score})")
    return "\n".join(rows)


def extract_json_payload(response: str) -> Dict[str, Any] | None:
    if not response:
        return None
    fence_matches = JSON_BLOCK_RE.findall(response)
    candidate = fence_matches[-1] if fence_matches else None
    if not candidate:
        stripped = response.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            candidate = stripped
    if not candidate:
        candidate = _first_brace_block(response)
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _first_brace_block(text: str) -> str | None:
    stack = []
    start = None
    for idx, ch in enumerate(text):
        if ch == "{":
            if start is None:
                start = idx
            stack.append(ch)
        elif ch == "}" and stack:
            stack.pop()
            if not stack and start is not None:
                return text[start : idx + 1]
    return None


def action_from_payload(payload: Dict[str, Any] | None) -> ActionUnion | None:
    if not isinstance(payload, dict):
        return None
    type_name = (payload.get("type") or payload.get("action") or payload.get("kind") or "").lower()
    if type_name == "explore":
        return ExploreAction(
            op=str(payload.get("op") or payload.get("operation") or "expand"),
            anchors=list(payload.get("anchors") or []),
            nodes=list(payload.get("nodes") or []),
            hop=int(payload.get("hop", 1)),
            limit=int(payload.get("limit", 50)),
        )
    if type_name == "memory":
        return MemoryAction(
            ops=list(payload.get("ops") or []),
            budget=int(payload.get("budget", 30)),
            diversify_by_dir=int(payload.get("diversify_by_dir", 3)),
        )
    if type_name == "repair":
        plan_targets = payload.get("plan_targets") or payload.get("targets") or []
        return RepairAction(
            apply=bool(payload.get("apply", True)),
            issue=dict(payload.get("issue") or {}),
            plan=payload.get("plan"),
            plan_targets=list(plan_targets),
            patch=payload.get("patch"),
        )
    if type_name == "submit":
        return SubmitAction()
    return None


def action_to_payload(action: ActionUnion) -> Dict[str, Any]:
    if isinstance(action, ExploreAction):
        return {
            "type": "explore",
            "op": action.op,
            "anchors": action.anchors,
            "nodes": action.nodes,
            "hop": action.hop,
            "limit": action.limit,
        }
    if isinstance(action, MemoryAction):
        return {
            "type": "memory",
            "ops": action.ops,
            "budget": action.budget,
            "diversify_by_dir": action.diversify_by_dir,
        }
    if isinstance(action, RepairAction):
        return {
            "type": "repair",
            "apply": action.apply,
            "issue": action.issue,
            "plan": action.plan,
            "plan_targets": action.plan_targets,
            "patch": action.patch,
        }
    return {"type": "submit"}


__all__ = [
    "SYSTEM_PROMPT",
    "FALLBACK_REASON_KEY",
    "summarise_observation",
    "extract_json_payload",
    "action_from_payload",
    "action_to_payload",
]
