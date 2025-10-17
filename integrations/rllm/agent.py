"""rLLM agent wrapper that translates model responses into Planner actions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List

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

from agents.common.chat import (
    FALLBACK_REASON_KEY,
    SYSTEM_PROMPT,
    action_from_payload,
    action_to_payload,
    extract_json_payload,
    summarise_observation,
)
from agents.rule_based.planner import PlannerAgent as RuleFallbackAgent
from core.actions import ActionUnion, SubmitAction


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
            text, metadata = summarise_observation(observation, reward, done, info)
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
            self._cur_step.action = action_to_payload(action_obj)
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
        ) -> tuple[str, ActionUnion, str, Dict[str, Any]]:
            payload = extract_json_payload(response)
            if payload is None:
                return self._fallback_action("no_json", response)
            thought = str(payload.get("thought") or payload.get("reasoning") or "").strip()
            raw_action = payload.get("action")
            if isinstance(raw_action, str):
                try:
                    raw_action = json.loads(raw_action)
                except json.JSONDecodeError:
                    raw_action = None
            action_obj = action_from_payload(raw_action)
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
        ) -> tuple[str, ActionUnion, str, Dict[str, Any]]:
            if not self._rule_agent:
                raise ValueError(f"Model response could not be parsed and fallback is disabled: {reason}")
            observation = self._last_env_observation or {}
            fallback = self._rule_agent.step(observation)
            action = fallback.get("action_obj") or SubmitAction()
            thought = fallback.get("plan", fallback.get("prompt", ""))
            assistant_msg = json.dumps(
                {
                    "thought": thought,
                    "action": action_to_payload(action),
                    FALLBACK_REASON_KEY: reason,
                },
                ensure_ascii=False,
            )
            meta = {
                "used_fallback": True,
                FALLBACK_REASON_KEY: reason,
                "raw_action": raw_action,
                "model_response": response,
            }
            return thought, action, assistant_msg, meta


__all__ = ["GraphPlannerRLLMAgent"]
