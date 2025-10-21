"""CGM rLLM agent 封装。

English summary
    Mirrors the CGM-specific PPO agent shipped with rLLM by turning planner
    observations into structured chat messages and parsing JSON patch replies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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
        _IMPORT_ERROR = _exc
    else:
        _IMPORT_ERROR = None
else:
    _IMPORT_ERROR = None

from ...agents.common.chat import extract_json_payload
from ...integrations.codefuse_cgm.formatting import GraphLinearizer, SnippetFormatter


DEFAULT_SYSTEM_PROMPT = (
    "You generate precise JSON patches for software issues."
    " Reply with an object containing a `patch` field that mirrors the"
    " Graph Planner CGM schema."
)


if BaseAgent is None:

    class CGMRLLMAgent:  # type: ignore[misc]
        """Placeholder that surfaces an actionable import error when rLLM is missing."""

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("rLLM is required to use CGMRLLMAgent") from _IMPORT_ERROR

else:

    @dataclass
    class _CGMObservation:
        """结构化的 CGM 观察数据，便于渲染 Prompt。"""

        issue: Dict[str, Any]
        plan_text: str
        plan_targets: List[Dict[str, Any]]
        graph_text: str
        snippets_text: str
        instruction: str

    @dataclass
    class CGMRLLMAgent(BaseAgent):
        """将 CGM 上下文转为聊天消息的最小代理。"""

        system_prompt: str = DEFAULT_SYSTEM_PROMPT

        def __post_init__(self) -> None:
            """初始化轨迹、消息缓冲与格式化工具。"""

            self._trajectory = Trajectory()
            self._messages: List[Dict[str, str]] = []
            self._cur_step: Step | None = None
            self._linearizer = GraphLinearizer()
            self._snippet_formatter = SnippetFormatter()
            self.reset()

        # ------------------------------------------------------------------
        # BaseAgent interface
        # ------------------------------------------------------------------
        def reset(self) -> None:
            """恢复初始状态并注入系统提示。"""

            self._trajectory = Trajectory()
            self._messages = [{"role": "system", "content": self.system_prompt}]
            self._cur_step = None

        def update_from_env(
            self,
            observation: Dict[str, Any],
            reward: float,
            done: bool,
            info: Dict[str, Any] | None,
            **kwargs,
        ) -> None:
            """根据环境观察构造用户消息并缓存元数据。"""

            payload = self._normalise_observation(observation)
            user_message = self._build_user_message(payload)
            metadata = {
                "issue": payload.issue,
                "plan_targets": payload.plan_targets,
                "reward": reward,
                "done": done,
            }
            if info:
                metadata["info"] = info
            self._messages.append({"role": "user", "content": user_message})
            step = Step(observation=user_message, info=metadata)
            step.chat_completions = list(self._messages)
            self._cur_step = step

        def update_from_model(self, response: str, **kwargs) -> Action:
            """解析模型返回的补丁并更新轨迹。"""

            if self._cur_step is None:
                raise RuntimeError("update_from_env must be called before update_from_model")
            patch, thought = self._parse_patch(response)
            self._messages.append({"role": "assistant", "content": response})
            self._cur_step.thought = thought
            self._cur_step.model_response = response
            self._cur_step.action = patch or {}
            self._trajectory.steps.append(self._cur_step)
            return Action(action=patch)

        @property
        def trajectory(self) -> Trajectory:
            """返回当前累计的交互轨迹。"""

            return self._trajectory

        @property
        def chat_completions(self) -> List[Dict[str, str]]:
            """返回包含系统提示的完整对话历史。"""

            return list(self._messages)

        def get_current_state(self) -> Step | None:
            """获取最后一个 ``Step``，若尚未交互则返回 ``None``。"""

            return self._trajectory.steps[-1] if self._trajectory.steps else None

        # ------------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------------
        def _normalise_observation(self, raw: Dict[str, Any]) -> _CGMObservation:
            """统一观察字段命名，补齐图与片段文本。"""

            issue = raw.get("issue") or {}
            plan_text = str(raw.get("plan_text") or raw.get("plan") or "").strip()
            plan_targets = [dict(t) for t in raw.get("plan_targets") or [] if isinstance(t, dict)]
            instruction = str(raw.get("instruction") or "Generate a JSON patch.").strip()
            graph_text = raw.get("graph_text")
            if graph_text is None:
                graph_text = self._linearizer.linearize(raw.get("graph"))
            snippets_text = raw.get("snippets_text")
            if snippets_text is None:
                snippets_text = self._snippet_formatter.format(raw.get("snippets"))
            return _CGMObservation(
                issue=dict(issue),
                plan_text=plan_text,
                plan_targets=plan_targets,
                graph_text=graph_text or "",
                snippets_text=snippets_text or "",
                instruction=instruction,
            )

        def _build_user_message(self, obs: _CGMObservation) -> str:
            """将结构化观察渲染为多段提示文本。"""

            sections = [obs.instruction]
            if obs.issue.get("title") or obs.issue.get("body"):
                title = obs.issue.get("title")
                body = obs.issue.get("body")
                issue_block = ["[Issue]"]
                if title:
                    issue_block.append(str(title))
                if body:
                    issue_block.append(str(body))
                sections.append("\n".join(issue_block))
            if obs.plan_text:
                sections.append(f"[Plan]\n{obs.plan_text}")
            if obs.graph_text:
                sections.append(f"[Subgraph]\n{obs.graph_text}")
            if obs.snippets_text:
                sections.append(f"[Snippets]\n{obs.snippets_text}")
            return "\n\n".join(section for section in sections if section)

        def _parse_patch(self, response: str) -> tuple[Optional[Dict[str, Any]], str]:
            """解析模型回复中的 JSON patch，返回补丁与思考文本。"""

            payload = extract_json_payload(response)
            if payload is None:
                try:
                    payload = json.loads(response)
                except Exception:
                    return None, ""
            thought = str(payload.get("thought") or payload.get("reasoning") or "").strip()
            patch_obj = None
            candidate = payload.get("patch")
            if isinstance(candidate, dict):
                patch_obj = candidate
            elif isinstance(payload.get("edits"), list):
                patch_obj = payload
            if not isinstance(patch_obj, dict):
                return None, thought

            edits_raw = patch_obj.get("edits")
            if not isinstance(edits_raw, list):
                return None, thought

            edits: List[Dict[str, Any]] = []
            for entry in edits_raw:
                if not isinstance(entry, dict):
                    continue
                path = entry.get("path")
                start = entry.get("start")
                end = entry.get("end") if entry.get("end") is not None else start
                new_text = entry.get("new_text")
                if not path or start is None or new_text is None:
                    continue
                try:
                    start_i = int(start)
                    end_i = int(end)
                except Exception:
                    continue
                text = str(new_text)
                if not text.endswith("\n"):
                    text += "\n"
                edits.append({"path": str(path), "start": start_i, "end": end_i, "new_text": text})
            if not edits:
                return None, thought

            summary = patch_obj.get("summary")
            if not isinstance(summary, str):
                summary = thought or "CGM patch"
            patch = {"edits": edits, "summary": summary}
            return patch, thought


__all__ = ["CGMRLLMAgent"]

