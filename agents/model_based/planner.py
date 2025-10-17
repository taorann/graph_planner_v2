"""Planner agent that delegates decision making to a locally hosted LLM."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from aci.schema import Plan, PlanTarget
from agents.common.chat import (
    FALLBACK_REASON_KEY,
    SYSTEM_PROMPT,
    action_from_payload,
    action_to_payload,
    extract_json_payload,
    summarise_observation,
)
from agents.rule_based import cgm_adapter
from agents.rule_based.planner import PlannerAgent as RulePlannerAgent
from core.actions import RepairAction, SubmitAction
from infra.config import Config, load as load_config
from integrations.local_llm import LocalLLMClient, LocalLLMError
from memory import subgraph_store


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


class LocalLLMPlannerAgent:
    """Agent that mirrors the rule-based flow but relies on a local chat model."""

    def __init__(
        self,
        *,
        client: Optional[LocalLLMClient] = None,
        system_prompt: Optional[str] = None,
        use_rule_fallback: bool = True,
    ) -> None:
        self.cfg: Config = load_config()
        pm_cfg = getattr(self.cfg, "planner_model", None)
        if client is None:
            if pm_cfg is None:
                raise RuntimeError("planner_model section missing in configuration")
            try:
                client = LocalLLMClient.from_config(pm_cfg)
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
    ) -> tuple[str, Any, str, Dict[str, Any]]:
        payload = extract_json_payload(response)
        if payload is None:
            return self._fallback_tuple(obs, "no_json", response)
        thought = str(payload.get("thought") or payload.get("reasoning") or "").strip()
        raw_action = payload.get("action")
        if isinstance(raw_action, str):
            try:
                raw_action = json.loads(raw_action)
            except json.JSONDecodeError:
                raw_action = None
        action_obj = action_from_payload(raw_action)
        if action_obj is None:
            return self._fallback_tuple(obs, "invalid_action", response, raw_action)

        if isinstance(action_obj, RepairAction):
            action_obj = self._ensure_patch(action_obj, obs)

        meta = {
            "used_fallback": False,
            "raw_action": raw_action,
        }
        return thought, action_obj, response or json.dumps(payload, ensure_ascii=False), meta

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
        assistant_msg = json.dumps(
            {
                "thought": thought,
                "action": action_to_payload(action),
                FALLBACK_REASON_KEY: reason,
                "error": error,
            },
            ensure_ascii=False,
        )
        meta = {
            "used_fallback": True,
            FALLBACK_REASON_KEY: reason,
            "raw_action": raw_action,
            "model_response": response,
        }
        if error:
            meta["error"] = error
        return thought, action, assistant_msg, meta

    # ------------------------------------------------------------------
    # Patch materialisation
    # ------------------------------------------------------------------
    def _ensure_patch(self, action: RepairAction, obs: Dict[str, Any]) -> RepairAction:
        plan_targets = self._normalise_plan_targets(action.plan_targets)
        if not plan_targets:
            plan_targets = self._plan_targets_from_snippets(self.state.last_snippets or [])
        if not plan_targets:
            return action

        plan_text = action.plan or self._build_plan_text(plan_targets)
        snippets = self.state.last_snippets or self._fallback_snippets()

        plan_obj = Plan(
            targets=[
                PlanTarget(
                    path=pt["path"],
                    start=int(pt["start"]),
                    end=int(pt["end"]),
                    id=str(pt.get("id") or f"{pt['path']}::{pt['start']}-{pt['end']}"),
                    why=pt.get("why", "graph_planner-local"),
                )
                for pt in plan_targets
            ],
            budget={"mode": self.cfg.mode},
            priority_tests=[],
        )

        subgraph = subgraph_store.wrap(obs.get("subgraph") or {})
        linearized = subgraph_store.linearize(subgraph, mode=self.cfg.collate.mode)
        patch = cgm_adapter.generate(
            subgraph_linearized=linearized,
            plan=plan_obj,
            constraints={"max_edits": max(1, len(plan_targets))},
            snippets=snippets,
            plan_text=plan_text,
            issue=self.state.issue,
        )
        patch.setdefault("summary", plan_text)

        action.plan = plan_text
        action.plan_targets = plan_targets
        action.patch = patch
        self.state.plan_targets = plan_targets
        self.state.plan_text = plan_text
        return action

    def _normalise_plan_targets(self, raw: Iterable[Any]) -> List[Dict[str, Any]]:
        targets: List[Dict[str, Any]] = []
        for entry in raw or []:
            if not isinstance(entry, dict):
                continue
            path = entry.get("path")
            start = entry.get("start", entry.get("line"))
            end = entry.get("end", start)
            if not path or start is None:
                continue
            try:
                start_i = int(start)
                end_i = int(end if end is not None else start_i)
            except Exception:
                continue
            targets.append(
                {
                    "path": str(path),
                    "start": start_i,
                    "end": end_i,
                    "id": entry.get("id"),
                    "why": entry.get("why", "graph_planner-llm"),
                }
            )
        return targets

    def _plan_targets_from_snippets(self, snippets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        plan_targets: List[Dict[str, Any]] = []
        for snip in snippets:
            path = snip.get("path")
            if not path:
                continue
            start = int(snip.get("start", 1))
            end = int(snip.get("end", start))
            node_id = snip.get("node_id") or f"{path}::{start}-{end}"
            plan_targets.append(
                {
                    "path": path,
                    "start": start,
                    "end": end,
                    "id": node_id,
                    "why": "graph_planner-snippet",
                }
            )
        return plan_targets

    def _build_plan_text(self, plan_targets: List[Dict[str, Any]]) -> str:
        lines = []
        snippet_index = {snip.get("path"): snip for snip in self.state.last_snippets or []}
        for target in plan_targets:
            path = target["path"]
            start = target["start"]
            end = target["end"]
            snippet = snippet_index.get(path) or {}
            preview = " | ".join(line.split(":", 1)[-1].strip() for line in (snippet.get("snippet") or [])[:3])
            lines.append(f"- {path} L{start}-{end}: {preview[:160]}")
        return "Plan to address the following locations:\n" + "\n".join(lines)

    def _fallback_snippets(self) -> List[Dict[str, Any]]:
        snippets: List[Dict[str, Any]] = []
        for cand in self.state.last_candidates[:1]:
            path = cand.get("path")
            if not path:
                continue
            span = cand.get("span") or {}
            start = int(span.get("start", 1))
            end = int(span.get("end", start))
            snippet_line = f"{start:04d}: {cand.get('name', '')}"
            snippets.append(
                {
                    "path": path,
                    "start": start,
                    "end": end,
                    "node_id": cand.get("id"),
                    "snippet": [snippet_line],
                }
            )
        return snippets


    def _normalise_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return json.loads(json.dumps(metadata, default=str))
        except Exception:
            return {k: str(v) for k, v in (metadata or {}).items()}


__all__ = ["LocalLLMPlannerAgent"]
