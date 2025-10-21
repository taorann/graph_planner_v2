"""Planner agent that delegates decision making to a locally hosted LLM."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol

from ..common import text_protocol
from ..common.chat import (
    FALLBACK_REASON_KEY,
    SYSTEM_PROMPT,
    action_to_payload,
    summarise_observation,
)
from ..rule_based.planner import PlannerAgent as RulePlannerAgent
from ...core.actions import (
    ActionUnion,
    ExploreAction,
    MemoryAction,
    NoopAction,
    RepairAction,
    SubmitAction,
)
from ...infra.config import Config, load as load_config
from ...integrations.local_llm import LocalLLMError, build_planner_client
ALLOWED_ACTIONS = {"explore", "memory", "repair", "submit", "noop"}


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"false", "0", "no", "off"}:
            return False
        if lowered in {"true", "1", "yes", "on"}:
            return True
    return bool(value)


@dataclass
class _AgentState:
    issue: Dict[str, Any] = field(default_factory=dict)
    phase: str = "expand"
    last_candidates: List[Dict[str, Any]] = field(default_factory=list)
    last_snippets: List[Dict[str, Any]] = field(default_factory=list)
    last_memory: Dict[str, Any] = field(default_factory=dict)
    last_repair: Dict[str, Any] = field(default_factory=dict)
    plan_targets: List[Dict[str, Any]] = field(default_factory=list)
    plan_text: str = ""


class _ChatClient(Protocol):
    def chat(
        self,
        messages: Iterable[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        ...


class LocalLLMPlannerAgent:
    """Agent that mirrors the rule-based flow but relies on a local chat model."""

    def __init__(
        self,
        *,
        client: Optional[_ChatClient] = None,
        system_prompt: Optional[str] = None,
        use_rule_fallback: bool = True,
    ) -> None:
        self.cfg: Config = load_config()
        pm_cfg = getattr(self.cfg, "planner_model", None)
        if client is None:
            if pm_cfg is None:
                raise RuntimeError("planner_model section missing in configuration")
            try:
                client = build_planner_client(pm_cfg)
            except Exception as exc:  # pragma: no cover - configuration error
                raise RuntimeError(
                    "planner model client could not be initialised; ensure local endpoint is configured"
                ) from exc
        self._client = client
        self.state = _AgentState()
        self._rule_agent = RulePlannerAgent() if use_rule_fallback else None
        prompt = system_prompt or getattr(pm_cfg, "system_prompt", None) or SYSTEM_PROMPT
        self._messages: List[Dict[str, str]] = [{"role": "system", "content": prompt}]
        self._last_reward: float = 0.0
        self._last_done: bool = False
        self._last_info: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------
    def step(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        if not self.state.issue or obs.get("reset") or obs.get("steps") == 0:
            self._on_reset(obs)
        self._update_state(obs)

        summary, metadata = summarise_observation(obs, self._last_reward, self._last_done, self._last_info)
        metadata = self._normalise_metadata(metadata)
        self._messages.append({"role": "user", "content": summary})

        try:
            response = self._client.chat(self._messages, extra={"metadata": metadata})
        except LocalLLMError as exc:
            return self._fallback_decision(obs, summary, "client_error", error=str(exc))

        thought, action_obj, assistant_msg, parser_meta = self._parse_model_response(response, obs)
        self._messages.append({"role": "assistant", "content": assistant_msg})

        self.state.phase = getattr(action_obj, "type", self.state.phase)
        result = {
            "prompt": summary,
            "response": assistant_msg,
            "thought": thought,
            "action_obj": action_obj,
            "metadata": parser_meta,
        }
        return result

    def observe_outcome(self, reward: float, done: bool, info: Optional[Dict[str, Any]]) -> None:
        self._last_reward = reward
        self._last_done = done
        self._last_info = info or {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _on_reset(self, obs: Dict[str, Any]) -> None:
        self.state = _AgentState(issue=obs.get("issue") or {})
        self._messages = [self._messages[0]]  # keep system prompt
        self._last_reward = 0.0
        self._last_done = False
        self._last_info = {}

    def _update_state(self, obs: Dict[str, Any]) -> None:
        info = obs.get("last_info") or {}
        kind = info.get("kind")
        if kind == "explore" and info.get("op") == "expand":
            self.state.last_candidates = info.get("candidates", [])
            self.state.phase = "memory"
        elif kind == "memory":
            self.state.last_memory = info
            self.state.phase = "read"
        elif kind == "explore" and info.get("op") == "read":
            self.state.last_snippets = info.get("snippets", [])
            self.state.phase = "plan"
        elif kind == "repair":
            self.state.last_repair = info
            if info.get("applied"):
                self.state.phase = "submit"
            else:
                self.state.phase = "expand"

    def _parse_model_response(
        self, response: str, obs: Dict[str, Any]
    ) -> tuple[str, ActionUnion, str, Dict[str, Any]]:
        try:
            block = text_protocol.parse_action_block(response, ALLOWED_ACTIONS)
        except text_protocol.ActionParseError as exc:
            return self._fallback_tuple(obs, "parse_error", response, error=str(exc))

        params = dict(block.get("params") or {})
        thought = str(params.pop("thought", "")).strip()
        try:
            action_obj = self._action_from_block(block.get("name"), params)
        except ValueError as exc:
            return self._fallback_tuple(obs, "invalid_action", response, raw_action=params, error=str(exc))

        meta = {
            "used_fallback": False,
            "raw_action": {"name": block.get("name"), "params": params},
        }
        assistant_msg = response or text_protocol.format_action_block(
            str(block.get("name") or "noop"),
            {"thought": thought, **params},
        )
        return thought, action_obj, assistant_msg, meta

    def _fallback_decision(
        self,
        obs: Dict[str, Any],
        summary: str,
        reason: str,
        *,
        error: Optional[str] = None,
        raw_response: Optional[str] = None,
        raw_action: Any | None = None,
    ) -> Dict[str, Any]:
        thought, action, assistant_msg, meta = self._fallback_tuple(
            obs, reason, raw_response or "", raw_action, error=error
        )
        self._messages.append({"role": "assistant", "content": assistant_msg})
        return {
            "prompt": summary,
            "response": assistant_msg,
            "thought": thought,
            "action_obj": action,
            "metadata": meta,
        }

    def _fallback_tuple(
        self,
        obs: Dict[str, Any],
        reason: str,
        response: str,
        raw_action: Any | None = None,
        *,
        error: Optional[str] = None,
    ) -> tuple[str, Any, str, Dict[str, Any]]:
        if not self._rule_agent:
            raise ValueError(f"Model response could not be parsed and fallback is disabled: {reason}")
        fallback = self._rule_agent.step(obs)
        action = fallback.get("action_obj") or SubmitAction()
        thought = fallback.get("plan", fallback.get("prompt", ""))
        payload = action_to_payload(action)
        params = {"thought": thought, **{k: v for k, v in payload.items() if k != "type"}}
        if error:
            params["error"] = error
        params[FALLBACK_REASON_KEY] = reason
        assistant_msg = text_protocol.format_action_block(payload.get("type", "noop"), params)
        meta = {
            "used_fallback": True,
            FALLBACK_REASON_KEY: reason,
            "raw_action": raw_action,
            "model_response": response,
        }
        if error:
            meta["error"] = error
        return thought, action, assistant_msg, meta

    def _action_from_block(self, name: Any, params: Dict[str, Any]) -> ActionUnion:
        action_name = str(name or "").lower()
        if action_name == "explore":
            op = str(params.get("op") or "expand").lower()
            anchors = self._ensure_dict_list(params.get("anchors"))
            nodes = self._ensure_str_list(params.get("nodes"))
            hop = self._safe_int(params.get("hop"), 1)
            limit = self._safe_int(params.get("limit"), 50)
            return ExploreAction(op=op, anchors=anchors, nodes=nodes, hop=hop, limit=limit)
        if action_name == "memory":
            target = str(params.get("target", "explore"))
            scope = str(params.get("scope", "turn"))
            intent = str(params.get("intent", "commit"))
            selector = params.get("selector")
            if isinstance(selector, (list, dict)):
                selector = json.dumps(selector, ensure_ascii=False)
            return MemoryAction(target=target, scope=scope, intent=intent, selector=selector)
        if action_name == "repair":
            subplan = params.get("subplan")
            if not isinstance(subplan, str) or not subplan.strip():
                raise ValueError("repair action requires non-empty subplan")
            focus_ids = self._ensure_str_list(params.get("focus_ids"))
            apply_flag = _safe_bool(params.get("apply", True))
            plan_targets = [
                {"id": fid, "why": "planner-focus"}
                for fid in focus_ids
                if fid
            ]
            return RepairAction(
                apply=apply_flag,
                issue=dict(self.state.issue or {}),
                plan=subplan.strip(),
                plan_targets=plan_targets,
                patch=None,
            )
        if action_name == "submit":
            return SubmitAction()
        if action_name == "noop":
            return NoopAction()
        raise ValueError(f"unsupported action '{action_name}'")

    def _ensure_str_list(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value else []
        if isinstance(value, Iterable):
            result: List[str] = []
            for item in value:
                if isinstance(item, str):
                    if item:
                        result.append(item)
                elif isinstance(item, (int, float)):
                    result.append(str(item))
            return result
        return []

    def _ensure_dict_list(self, value: Any) -> List[Dict[str, Any]]:
        if value is None:
            return []
        if isinstance(value, dict):
            return [dict(value)]
        if isinstance(value, Iterable):
            result: List[Dict[str, Any]] = []
            for item in value:
                if isinstance(item, dict):
                    result.append(dict(item))
            return result
        return []

    def _safe_int(self, value: Any, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return default

    def _normalise_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return json.loads(json.dumps(metadata, default=str))
        except Exception:
            return {k: str(v) for k, v in (metadata or {}).items()}


__all__ = ["LocalLLMPlannerAgent"]
