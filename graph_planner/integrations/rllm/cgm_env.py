"""CGM rLLM 环境适配器。

English summary
    Exposes a ``BaseEnv`` implementation that drives ``PlannerEnv`` to gather
    graph/snippet context, forwards it to the CGM agent, and evaluates returned
    patches within the sandboxed environment.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
from uuid import uuid4

from ...infra.vendor import ensure_rllm_importable

ensure_rllm_importable()

try:
    from rllm.rllm.environments.base.base_env import BaseEnv  # type: ignore[attr-defined]
except ModuleNotFoundError:
    try:
        from rllm.environments.base.base_env import BaseEnv  # type: ignore[attr-defined]
    except ModuleNotFoundError as _exc:  # pragma: no cover - optional dependency
        BaseEnv = None  # type: ignore[assignment]
        _IMPORT_ERROR = _exc
    else:
        _IMPORT_ERROR = None
else:
    _IMPORT_ERROR = None

from ...core.actions import ExploreAction, MemoryAction, RepairAction, SubmitAction
from ...env.planner_env import PlannerEnv
from ...runtime.sandbox import SandboxConfig


DEFAULT_INSTRUCTION = (
    "Generate a JSON patch with `patch.edits` describing path/start/end/new_text."
    " Ensure every `new_text` ends with a newline."
)


if BaseEnv is None:

    class CGMRLLMEnv:  # type: ignore[misc]
        """Placeholder raising an informative error when rLLM is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("rLLM is required to use CGMRLLMEnv") from _IMPORT_ERROR

else:

    @dataclass
    class _Context:
        """缓存环境启动时整理出的计划、片段与图信息。"""

        plan_text: str
        plan_targets: List[Dict[str, Any]]
        snippets: List[Dict[str, Any]]
        graph: Dict[str, Any]
        graph_text: str
        snippets_text: str

    class CGMRLLMEnv(BaseEnv):
        """面向 CGM 的 rLLM 环境，封装提示构造与补丁评估流程。"""

        def __init__(
            self,
            entry: Dict[str, Any],
            *,
            max_steps: int = 1,
            instruction: str | None = None,
            reward_scale: float = 1.0,
            failure_penalty: float = 0.0,
            step_penalty: float = 0.0,
            timeout_penalty: float = 0.0,
            repo_operation_limit: int | None = None,
            synthesis_strategy: str | None = None,
        ) -> None:
            """复制任务条目并保存交互限制/指令配置。"""

            self.entry = json.loads(json.dumps(entry))
            self.max_steps = max(1, int(entry.get("max_steps", max_steps)))
            self.instruction = instruction or entry.get("instruction") or DEFAULT_INSTRUCTION
            self.reward_scale = float(entry.get("reward_scale", reward_scale))
            self.failure_penalty = float(entry.get("failure_penalty", failure_penalty))
            self.step_penalty = float(entry.get("step_penalty", step_penalty))
            self.timeout_penalty = float(entry.get("timeout_penalty", timeout_penalty))
            limit = entry.get("repo_operation_limit") or entry.get("repo_op_limit") or repo_operation_limit
            self.repo_operation_limit = int(limit) if limit else None
            self.synthesis_strategy = entry.get("synthesis_strategy") or synthesis_strategy
            self._planner: PlannerEnv | None = None
            self._context: _Context | None = None
            self._issue_uid = uuid4().hex
            issue = self.entry.get("issue") or {}
            self._source_issue_id = str(issue.get("id") or "")
            self._step_count = 0

        # ------------------------------------------------------------------
        # BaseEnv interface
        # ------------------------------------------------------------------
        def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            """启动 PlannerEnv，采集初始上下文并返回观察值。"""

            self.close()
            self._planner = self._spawn_planner()
            base_obs = self._planner.reset()
            self._context = self._prepare_context(base_obs)
            self._step_count = 0
            obs = {
                "issue": base_obs.get("issue", {}),
                "plan_text": self._context.plan_text,
                "plan_targets": self._context.plan_targets,
                "graph": self._context.graph,
                "graph_text": self._context.graph_text,
                "snippets": self._context.snippets,
                "snippets_text": self._context.snippets_text,
                "instruction": self.instruction,
                "env_config": {
                    "reward_scale": self.reward_scale,
                    "failure_penalty": self.failure_penalty,
                    "step_penalty": self.step_penalty,
                    "timeout_penalty": self.timeout_penalty,
                    "repo_operation_limit": self.repo_operation_limit,
                    "synthesis_strategy": self.synthesis_strategy,
                },
            }
            info = {
                "task_id": self.entry.get("task_id"),
                "max_steps": self.max_steps,
                "issue_uid": self._issue_uid,
                "source_issue_id": self._source_issue_id,
            }
            return obs, info

        def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
            """将代理生成的补丁传入 PlannerEnv 并返回测试结果。"""

            if self._planner is None or self._context is None:
                raise RuntimeError("Environment must be reset before calling step().")

            self._step_count += 1
            patch = action.action if hasattr(action, "action") else action
            if not isinstance(patch, dict):
                patch = {}

            repair = RepairAction(
                apply=True,
                issue=self.entry.get("issue", {}),
                plan=self._context.plan_text,
                plan_targets=self._context.plan_targets,
                patch=patch,
            )

            repair_obs, _, _, repair_info = self._planner.step(repair)
            submit_obs, reward, done, submit_info = self._planner.step(SubmitAction())

            info = {
                "repair": repair_info,
                "submit": submit_info,
            }
            obs = {
                "issue": submit_obs.get("issue", {}),
                "tests": submit_info.get("tests"),
                "plan_text": self._context.plan_text,
            }
            adjusted = float(reward) * self.reward_scale
            if not done:
                adjusted -= self.step_penalty

            limit_triggered = False
            if self.repo_operation_limit and self._step_count >= self.repo_operation_limit:
                done = True
                limit_triggered = True
                info.setdefault("termination_reason", "repo_operation_limit")
            if not done and self._step_count >= self.max_steps:
                done = True
                limit_triggered = True
                info.setdefault("termination_reason", "max_steps")

            if done and reward <= 0:
                adjusted -= self.failure_penalty
                if limit_triggered:
                    adjusted -= self.timeout_penalty

            return obs, adjusted, done, info

        def close(self) -> None:
            """关闭底层 PlannerEnv 并释放上下文缓存。"""

            if self._planner is not None:
                try:
                    self._planner.close()
                finally:
                    self._planner = None
                    self._context = None
            self._step_count = 0

        @staticmethod
        def from_dict(extra_info: Dict[str, Any] | str) -> "CGMRLLMEnv":
            """允许通过 JSON/字典描述快速构建环境实例。"""

            if isinstance(extra_info, str):
                extra_info = json.loads(extra_info)
            env_kwargs: Dict[str, Any] = {}
            if isinstance(extra_info, dict):
                for key in (
                    "reward_scale",
                    "failure_penalty",
                    "step_penalty",
                    "timeout_penalty",
                    "repo_operation_limit",
                    "repo_op_limit",
                    "synthesis_strategy",
                ):
                    if key in extra_info:
                        env_kwargs[key] = extra_info[key]
            raw_entry = extra_info.get("raw_entry_json") if isinstance(extra_info, dict) else None
            if raw_entry:
                entry_payload = json.loads(raw_entry)
            else:
                entry_payload = extra_info
            return CGMRLLMEnv(entry=entry_payload, **env_kwargs)

        # ------------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------------
        def _spawn_planner(self) -> PlannerEnv:
            """根据任务配置启动新的 ``PlannerEnv``。"""

            sandbox = dict(self.entry.get("sandbox") or {})
            if not sandbox:
                raise ValueError("Sandbox configuration is required for CGMRLLMEnv")
            sandbox.setdefault("mounts", {})
            sandbox.setdefault("env", {})
            ds_path = sandbox.get("r2e_ds_json")
            if ds_path:
                sandbox["r2e_ds_json"] = str(Path(ds_path).expanduser().resolve())
            cfg = SandboxConfig(**sandbox)
            issue = self._build_issue_payload()
            return PlannerEnv(issue=issue, sandbox_cfg=cfg)

        def _build_issue_payload(self) -> Dict[str, Any]:
            issue = dict(self.entry.get("issue") or {})
            original_id = str(issue.get("id") or "")
            issue.setdefault("metadata", {})
            issue["metadata"]["source_issue_id"] = original_id or issue.get("metadata", {}).get("source_issue_id") or ""

            parts = [
                original_id or None,
                str(self.entry.get("task_id") or ""),
                f"pid{os.getpid()}",
                self._issue_uid,
            ]
            issue["id"] = "__".join([p for p in parts if p]) or self._issue_uid
            return issue

        def _prepare_context(self, initial_obs: Dict[str, Any]) -> _Context:
            """整理计划、片段、图结构文本供代理提示使用。"""

            plan = self.entry.get("plan") or {}
            plan_text = str(plan.get("text") or plan.get("plan_text") or "").strip()
            plan_targets = [dict(t) for t in plan.get("targets") or [] if isinstance(t, dict)]
            snippets = [dict(s) for s in self.entry.get("snippets") or [] if isinstance(s, dict)]
            graph = dict(self.entry.get("graph") or initial_obs.get("subgraph") or {})

            if not (plan_text and plan_targets and snippets):
                gathered = self._gather_from_env()
                plan_text = plan_text or gathered.plan_text
                plan_targets = plan_targets or gathered.plan_targets
                snippets = snippets or gathered.snippets
                graph = graph or gathered.graph

            # Use dedicated helpers to serialise context
            from ...integrations.codefuse_cgm.formatting import GraphLinearizer, SnippetFormatter

            linearizer_helper = GraphLinearizer()
            snippet_helper = SnippetFormatter()
            graph_obj = graph or (self._planner.subgraph.to_json_obj() if self._planner else {})
            graph_text = linearizer_helper.linearize(graph_obj)
            snippets_text = snippet_helper.format(snippets)

            return _Context(
                plan_text=plan_text,
                plan_targets=plan_targets,
                snippets=snippets,
                graph=graph_obj,
                graph_text=graph_text,
                snippets_text=snippets_text,
            )

        def _gather_from_env(self) -> _Context:
            """在 PlannerEnv 内执行标准探索以获取上下文。"""

            assert self._planner is not None
            planner = self._planner

            issue = planner.issue or {}
            failure = issue.get("failure_frame") or {}
            anchors = list(self.entry.get("anchors") or [])
            path_hint = failure.get("path")
            if path_hint:
                anchors.append({"kind": "file", "text": path_hint})
            if not anchors:
                anchors.append({"kind": "repo", "text": issue.get("id", "")})

            expand = ExploreAction(op="expand", anchors=anchors, hop=1, limit=planner.config.max_nodes_per_anchor)
            obs, _, _, info = planner.step(expand)
            candidates = info.get("candidates") or []

            planner.step(MemoryAction())

            node_ids = [cand.get("id") for cand in candidates[:3] if cand.get("id")]
            read = ExploreAction(op="read", nodes=node_ids, limit=len(node_ids) or 1)
            obs, _, _, info = planner.step(read)
            snippets = [dict(s) for s in info.get("snippets") or [] if isinstance(s, dict)]

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
                        "why": "graph_planner-cgm",
                    }
                )

            plan_lines = []
            for target in plan_targets:
                path = target["path"]
                snippet = next((s for s in snippets if s.get("path") == path), {})
                preview = " | ".join(line.split(":", 1)[-1].strip() for line in (snippet.get("snippet") or [])[:2])
                plan_lines.append(f"- {path} L{target['start']}-{target['end']}: {preview[:160]}")
            plan_text = "Plan to address the following locations:\n" + "\n".join(plan_lines)

            graph_obj = planner.subgraph.to_json_obj()
            from ...integrations.codefuse_cgm.formatting import GraphLinearizer, SnippetFormatter

            linearizer_helper = GraphLinearizer()
            snippet_helper = SnippetFormatter()
            graph_text = linearizer_helper.linearize(graph_obj)
            snippets_text = snippet_helper.format(snippets)

            return _Context(
                plan_text=plan_text,
                plan_targets=plan_targets,
                snippets=snippets,
                graph=graph_obj,
                graph_text=graph_text,
                snippets_text=snippets_text,
            )


__all__ = ["CGMRLLMEnv"]

