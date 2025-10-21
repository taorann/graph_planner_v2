"""Graph Planner rLLM agent 封装。

English summary
    Provides a thin wrapper around the rLLM ``BaseAgent`` so PPO training can
    interact with Graph Planner while keeping JSON parsing, fallback logic and
    CGM patch synthesis encapsulated in Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

from ...infra.vendor import ensure_rllm_importable

ensure_rllm_importable()

try:
    from rllm.rllm.agents.agent import Action, BaseAgent, Step, Trajectory  # type: ignore[attr-defined]
except ModuleNotFoundError:
    try:
        from rllm.agents.agent import Action, BaseAgent, Step, Trajectory  # type: ignore[attr-defined]
    except ModuleNotFoundError as _exc:  # pragma: no cover - optional dependency
        Action = None  # type: ignore[assignment]
        BaseAgent = None  # type: ignore[assignment]
        Step = None  # type: ignore[assignment]
        Trajectory = None  # type: ignore[assignment]
        _AGENT_IMPORT_ERROR = _exc
    else:
        _AGENT_IMPORT_ERROR = None
else:
    _AGENT_IMPORT_ERROR = None

from ...agents.common import text_protocol
from ...agents.common.chat import (
    FALLBACK_REASON_KEY,
    SYSTEM_PROMPT,
    action_to_payload,
    summarise_observation,
)
from ...agents.rule_based.planner import PlannerAgent as RuleFallbackAgent
from ...core.actions import (
    ActionUnion,
    ExploreAction,
    MemoryAction,
    NoopAction,
    RepairAction,
    SubmitAction,
)
from ...infra.config import Config, load as load_config


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


if BaseAgent is None:

    class GraphPlannerRLLMAgent:  # type: ignore[misc]
        """Placeholder that surfaces an actionable import error when rLLM is missing."""

        def __init__(self, *args, **kwargs):
            raise ImportError("rLLM is required to use GraphPlannerRLLMAgent") from _AGENT_IMPORT_ERROR

else:

    @dataclass
    class _AgentState:
        """保存最近一次交互的上下文信息，供 fallback 与补丁生成复用。"""

        issue: Dict[str, Any] = field(default_factory=dict)
        phase: str = "expand"
        last_candidates: List[Dict[str, Any]] = field(default_factory=list)
        last_snippets: List[Dict[str, Any]] = field(default_factory=list)
        last_memory: Dict[str, Any] = field(default_factory=dict)
        last_repair: Dict[str, Any] = field(default_factory=list)
        plan_targets: List[Dict[str, Any]] = field(default_factory=list)
        plan_text: str = ""

    @dataclass
    class GraphPlannerRLLMAgent(BaseAgent):
        """面向 rLLM 的 Graph Planner 代理封装。"""

        system_prompt: str = SYSTEM_PROMPT
        use_rule_fallback: bool = True

        def __post_init__(self) -> None:
            """初始化轨迹、消息列表以及可选的规则后备代理。"""

            self._trajectory = Trajectory()
            self._messages: List[Dict[str, str]] = []
            self._rule_agent = RuleFallbackAgent() if self.use_rule_fallback else None
            self._last_env_observation: Dict[str, Any] | None = None
            self._step_index = 0
            self._state = _AgentState()
            self._config: Config = load_config()
            self.reset()

        # ------------------------------------------------------------------
        # BaseAgent interface
        # ------------------------------------------------------------------
        def reset(self):
            """重置内部状态与交互历史。"""

            self._trajectory = Trajectory()
            self._messages = [{"role": "system", "content": self.system_prompt}]
            self._cur_step: Step | None = None
            self._last_env_observation = None
            self._step_index = 0
            self._state = _AgentState()

        def update_from_env(self, observation: Any, reward: float, done: bool, info: Dict[str, Any] | None, **kwargs):
            """根据环境返回的观察值更新轨迹和内部状态。"""

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
            self._update_state(observation)

        def update_from_model(self, response: str, **kwargs) -> Action:
            """解析模型输出，更新当前步骤并返回 rLLM ``Action``。"""

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
            """训练过程中累计的步骤轨迹。"""

            return self._trajectory

        @property
        def chat_completions(self) -> List[Dict[str, str]]:
            """以聊天消息形式返回历史对话。"""

            return list(self._messages)

        def get_current_state(self) -> Step | None:
            """返回最近一次 ``Step``，如不存在则返回 ``None``。"""

            if not self._trajectory.steps:
                return None
            return self._trajectory.steps[-1]

        # ------------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------------
        def _parse_model_response(
            self, response: str
        ) -> tuple[str, ActionUnion, str, Dict[str, Any]]:
            """尝试从模型回复中解析 Thought 与 Action，失败时触发 fallback。"""

            try:
                block = text_protocol.parse_action_block(response, ALLOWED_ACTIONS)
            except text_protocol.ActionParseError as exc:
                return self._fallback_action("parse_error", response, error=str(exc))

            params = dict(block.get("params") or {})
            thought = str(params.pop("thought", "")).strip()
            try:
                action_obj = self._action_from_block(block.get("name"), params)
            except ValueError as exc:
                return self._fallback_action("invalid_action", response, raw_action=params, error=str(exc))

            meta = {
                "used_fallback": False,
                "raw_action": {"name": block.get("name"), "params": params},
            }
            assistant_msg = response or text_protocol.format_action_block(
                str(block.get("name") or "noop"),
                {"thought": thought, **params},
            )
            return thought, action_obj, assistant_msg, meta

        def _fallback_action(
            self,
            reason: str,
            response: str,
            raw_action: Any | None = None,
            *,
            error: str | None = None,
        ) -> tuple[str, ActionUnion, str, Dict[str, Any]]:
            """调用规则代理生成保底动作，并记录失败原因。"""

            if not self._rule_agent:
                raise ValueError(f"Model response could not be parsed and fallback is disabled: {reason}")
            observation = self._last_env_observation or {}
            fallback = self._rule_agent.step(observation)
            action = fallback.get("action_obj") or SubmitAction()
            thought = fallback.get("plan", fallback.get("prompt", ""))
            payload = action_to_payload(action)
            params = {"thought": thought, **{k: v for k, v in payload.items() if k != "type"}}
            params[FALLBACK_REASON_KEY] = reason
            if error:
                params["error"] = error
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

        # ------------------------------------------------------------------
        # Patch helpers
        # ------------------------------------------------------------------
        def _update_state(self, observation: Dict[str, Any]) -> None:
            """保存最近一次环境信息，供补丁生成器使用。"""

            issue = observation.get("issue") or {}
            if issue and not self._state.issue:
                self._state.issue = issue
            info = observation.get("last_info") or {}
            kind = info.get("kind")
            if kind == "explore" and info.get("op") == "expand":
                self._state.last_candidates = info.get("candidates", [])
                self._state.phase = "memory"
            elif kind == "memory":
                self._state.last_memory = info
                self._state.phase = "read"
            elif kind == "explore" and info.get("op") == "read":
                self._state.last_snippets = info.get("snippets", [])
                self._state.phase = "plan"
            elif kind == "repair":
                self._state.last_repair = info
                if info.get("applied"):
                    self._state.phase = "submit"
                else:
                    self._state.phase = "expand"

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
                    issue=dict(self._state.issue or {}),
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
                    if isinstance(item, str) and item:
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

__all__ = ["GraphPlannerRLLMAgent"]
