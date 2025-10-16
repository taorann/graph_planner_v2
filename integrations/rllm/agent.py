"""rLLM agent wrapper that translates model responses into Planner actions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

try:
    from rllm.agents.agent import Action, BaseAgent, Step, Trajectory
except ImportError as _exc:  # pragma: no cover - optional dependency
    Action = None  # type: ignore[assignment]
    BaseAgent = None  # type: ignore[assignment]
    Step = None  # type: ignore[assignment]
    Trajectory = None  # type: ignore[assignment]
    _AGENT_IMPORT_ERROR = _exc
else:
    _AGENT_IMPORT_ERROR = None

from agents.rule_based.planner import PlannerAgent as RuleFallbackAgent
from core.actions import (
    ActionUnion,
    ExploreAction,
    MemoryAction,
    RepairAction,
    SubmitAction,
)

SYSTEM_PROMPT = (
    "You are the Graph Planner RL controller.\n"
    "You operate on a code graph derived from a software repository.\n"
    "For every observation produce a JSON object with the keys 'thought' and 'action'.\n"
    "The 'action' object must contain a 'type' field (explore, memory, repair, submit).\n"
    "For explore you may also set 'op' (find|expand|read), 'anchors', 'nodes', 'hop', and 'limit'.\n"
    "For memory provide 'ops' describing memory operations.\n"
    "For repair include 'apply', 'plan', 'plan_targets', and optionally 'patch'.\n"
    "Always respond with valid JSON (optionally inside ```json fences)."
)

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})```", re.DOTALL)
_FALLBACK_REASON_KEY = "fallback_reason"


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


def _summarise_observation(obs: Dict[str, Any], reward: float, done: bool, info: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
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
            lines.append(f"Snippet {snip.get('path')}@{snip.get('start')}->{snip.get('end')}: {snippet_lines}")
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
        "info": info,
    }
    return summary, metadata


def _extract_json_payload(response: str) -> Dict[str, Any] | None:
    if not response:
        return None
    fence_matches = _JSON_BLOCK_RE.findall(response)
    candidate = fence_matches[-1] if fence_matches else None
    if not candidate:
        stripped = response.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            candidate = stripped
    if not candidate:
        brace_match = _first_brace_block(response)
        candidate = brace_match
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


def _action_from_payload(payload: Dict[str, Any] | None) -> ActionUnion | None:
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


def _action_to_payload(action: ActionUnion) -> Dict[str, Any]:
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


if BaseAgent is None:

    class GraphPlannerRLLMAgent:  # type: ignore[misc]
        """Placeholder that surfaces an actionable import error when rLLM is missing."""

        def __init__(self, *args, **kwargs):
            raise ImportError("rLLM is required to use GraphPlannerRLLMAgent") from _AGENT_IMPORT_ERROR

else:

    @dataclass
    class GraphPlannerRLLMAgent(BaseAgent):
        """Agent facade that keeps the rLLM execution engine decoupled from Planner internals."""

        system_prompt: str = SYSTEM_PROMPT
        use_rule_fallback: bool = True

        def __post_init__(self) -> None:
            self._trajectory = Trajectory()
            self._messages: List[Dict[str, str]] = []
            self._rule_agent = RuleFallbackAgent() if self.use_rule_fallback else None
            self._last_env_observation: Dict[str, Any] | None = None
            self._step_index = 0
            self.reset()

        # ------------------------------------------------------------------
        # BaseAgent interface
        # ------------------------------------------------------------------
        def reset(self):
            self._trajectory = Trajectory()
            self._messages = [{"role": "system", "content": self.system_prompt}]
            self._cur_step: Step | None = None
            self._last_env_observation = None
            self._step_index = 0

        def update_from_env(self, observation: Any, reward: float, done: bool, info: Dict[str, Any] | None, **kwargs):
            info = info or {}
            text, metadata = _summarise_observation(observation, reward, done, info)
            if self._trajectory.steps:
                prior = self._trajectory.steps[-1]
                prior.next_observation = text
                prior.reward = reward
                prior.done = done
                prior.info = {**prior.info, **metadata}
            self._messages.append({"role": "user", "content": text})
            self._cur_step = Step(observation=text, info=metadata)
            self._cur_step.chat_completions = list(self._messages)
            self._last_env_observation = observation

        def update_from_model(self, response: str, **kwargs) -> Action:
            if self._cur_step is None:
                raise RuntimeError("update_from_env must be called before update_from_model")
            thought, action_obj, assistant_msg, parser_meta = self._parse_model_response(response)
            self._messages.append({"role": "assistant", "content": assistant_msg})

            self._cur_step.thought = thought
            self._cur_step.action = _action_to_payload(action_obj)
            self._cur_step.model_response = response
            self._cur_step.info.update(parser_meta)
            self._trajectory.steps.append(self._cur_step)
            self._step_index += 1
            return Action(action=action_obj)

        @property
        def trajectory(self) -> Trajectory:
            return self._trajectory

        @property
        def chat_completions(self) -> List[Dict[str, str]]:
            return list(self._messages)

        def get_current_state(self) -> Step | None:
            if not self._trajectory.steps:
                return None
            return self._trajectory.steps[-1]

        # ------------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------------
        def _parse_model_response(
            self, response: str
        ) -> Tuple[str, ActionUnion, str, Dict[str, Any]]:
            payload = _extract_json_payload(response)
            if payload is None:
                return self._fallback_action("no_json", response)
            thought = str(payload.get("thought") or payload.get("reasoning") or "").strip()
            raw_action = payload.get("action")
            if isinstance(raw_action, str):
                try:
                    raw_action = json.loads(raw_action)
                except json.JSONDecodeError:
                    raw_action = None
            action_obj = _action_from_payload(raw_action)
            if action_obj is None:
                return self._fallback_action("invalid_action", response, raw_action)
            return thought, action_obj, response or json.dumps(payload, ensure_ascii=False), {
                "used_fallback": False,
                "raw_action": raw_action,
            }

        def _fallback_action(
            self,
            reason: str,
            response: str,
            raw_action: Any | None = None,
        ) -> Tuple[str, ActionUnion, str, Dict[str, Any]]:
            if not self._rule_agent:
                raise ValueError(f"Model response could not be parsed and fallback is disabled: {reason}")
            observation = self._last_env_observation or {}
            fallback = self._rule_agent.step(observation)
            action = fallback.get("action_obj") or SubmitAction()
            thought = fallback.get("plan", fallback.get("prompt", ""))
            assistant_msg = json.dumps(
                {
                    "thought": thought,
                    "action": _action_to_payload(action),
                    _FALLBACK_REASON_KEY: reason,
                },
                ensure_ascii=False,
            )
            meta = {
                "used_fallback": True,
                _FALLBACK_REASON_KEY: reason,
                "raw_action": raw_action,
                "model_response": response,
            }
            return thought, action, assistant_msg, meta
