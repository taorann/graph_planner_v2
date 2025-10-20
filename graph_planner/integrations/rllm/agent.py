"""Graph Planner rLLM agent 封装。

English summary
    Provides a thin wrapper around the rLLM ``BaseAgent`` so PPO training can
    interact with Graph Planner while keeping JSON parsing, fallback logic and
    CGM patch synthesis encapsulated in Python.
"""

from __future__ import annotations

import json
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

from ...agents.common.chat import (
    FALLBACK_REASON_KEY,
    SYSTEM_PROMPT,
    action_from_payload,
    action_to_payload,
    extract_json_payload,
    summarise_observation,
)
from ...agents.rule_based import cgm_adapter
from ...agents.rule_based.planner import PlannerAgent as RuleFallbackAgent
from ...core.actions import ActionUnion, RepairAction, SubmitAction
from ...infra.config import Config, load as load_config
from ...memory import subgraph_store
from aci.schema import Plan, PlanTarget


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
            if isinstance(action_obj, RepairAction):
                try:
                    action_obj = self._ensure_patch(action_obj)
                except Exception:
                    return self._fallback_action("patch_error", response, raw_action)
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
            """调用规则代理生成保底动作，并记录失败原因。"""

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

        def _ensure_patch(self, action: RepairAction) -> RepairAction:
            """如果模型动作缺少补丁，则调用 CGM 生成并填充。"""

            plan_targets = self._normalise_plan_targets(action.plan_targets)
            if not plan_targets:
                plan_targets = self._plan_targets_from_snippets(self._state.last_snippets or [])
            if not plan_targets:
                return action

            plan_text = action.plan or self._build_plan_text(plan_targets)
            snippets = self._state.last_snippets or self._fallback_snippets()

            plan_obj = Plan(
                targets=[
                    PlanTarget(
                        path=pt["path"],
                        start=int(pt["start"]),
                        end=int(pt["end"]),
                        id=str(pt.get("id") or f"{pt['path']}::{pt['start']}-{pt['end']}") ,
                        why=pt.get("why", "graph_planner-rl"),
                    )
                    for pt in plan_targets
                ],
                budget={"mode": self._config.mode},
                priority_tests=[],
            )

            subgraph = subgraph_store.wrap((self._last_env_observation or {}).get("subgraph") or {})
            linearized = subgraph_store.linearize(subgraph, mode=self._config.collate.mode)
            patch = cgm_adapter.generate(
                subgraph_linearized=linearized,
                plan=plan_obj,
                constraints={"max_edits": max(1, len(plan_targets))},
                snippets=snippets,
                plan_text=plan_text,
                issue=self._state.issue,
            )
            patch.setdefault("summary", plan_text)

            action.plan = plan_text
            action.plan_targets = plan_targets
            action.patch = patch
            self._state.plan_targets = plan_targets
            self._state.plan_text = plan_text
            return action

        def _normalise_plan_targets(self, raw: Iterable[Any]) -> List[Dict[str, Any]]:
            """将模型输出的 plan_targets 规范化为路径/行号字典。"""

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
                        "why": entry.get("why", "graph_planner-rl"),
                    }
                )
            return targets

        def _plan_targets_from_snippets(self, snippets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """当缺少 plan_targets 时根据片段自动推断。"""

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
            """使用片段摘要生成自然语言计划描述。"""

            lines = []
            snippet_index = {snip.get("path"): snip for snip in self._state.last_snippets or []}
            for target in plan_targets:
                path = target["path"]
                start = target["start"]
                end = target["end"]
                snippet = snippet_index.get(path) or {}
                preview = " | ".join(line.split(":", 1)[-1].strip() for line in (snippet.get("snippet") or [])[:3])
                lines.append(f"- {path} L{start}-{end}: {preview[:160]}")
            return "Plan to address the following locations:\n" + "\n".join(lines)

        def _fallback_snippets(self) -> List[Dict[str, Any]]:
            """当模型/环境无片段可用时构造最小补丁上下文。"""

            snippets: List[Dict[str, Any]] = []
            for cand in self._state.last_candidates[:1]:
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


__all__ = ["GraphPlannerRLLMAgent"]
