"""CGM rLLM 环境适配器。

English summary
    Exposes a ``BaseEnv`` implementation that drives ``PlannerEnv`` to gather
    graph/snippet context, forwards it to the CGM agent, and evaluates returned
    patches within the sandboxed environment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from ...infra.vendor import ensure_rllm_importable

ensure_rllm_importable()

try:
    from rllm.environments.base.base_env import BaseEnv  # type: ignore[attr-defined]
except ModuleNotFoundError as _exc:  # pragma: no cover - optional dependency
    BaseEnv = None  # type: ignore[assignment]
    _IMPORT_ERROR = _exc
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
        ) -> None:
            """复制任务条目并保存交互限制/指令配置。"""

            self.entry = json.loads(json.dumps(entry))
            self.max_steps = max(1, int(entry.get("max_steps", max_steps)))
            self.instruction = instruction or entry.get("instruction") or DEFAULT_INSTRUCTION
            self._planner: PlannerEnv | None = None
            self._context: _Context | None = None

        # ------------------------------------------------------------------
        # BaseEnv interface
        # ------------------------------------------------------------------
        def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            """启动 PlannerEnv，采集初始上下文并返回观察值。"""

            self.close()
            self._planner = self._spawn_planner()
            base_obs = self._planner.reset()
            self._context = self._prepare_context(base_obs)
            obs = {
                "issue": base_obs.get("issue", {}),
                "plan_text": self._context.plan_text,
                "plan_targets": self._context.plan_targets,
                "graph": self._context.graph,
                "graph_text": self._context.graph_text,
                "snippets": self._context.snippets,
                "snippets_text": self._context.snippets_text,
                "instruction": self.instruction,
            }
            info = {
                "task_id": self.entry.get("task_id"),
                "max_steps": self.max_steps,
            }
            return obs, info

        def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
            """将代理生成的补丁传入 PlannerEnv 并返回测试结果。"""

            if self._planner is None or self._context is None:
                raise RuntimeError("Environment must be reset before calling step().")

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
            return obs, reward, done, info

        def close(self) -> None:
            """关闭底层 PlannerEnv 并释放上下文缓存。"""

            if self._planner is not None:
                try:
                    self._planner.close()
                finally:
                    self._planner = None
                    self._context = None

        @staticmethod
        def from_dict(extra_info: Dict[str, Any] | str) -> "CGMRLLMEnv":
            """允许通过 JSON/字典描述快速构建环境实例。"""

            if isinstance(extra_info, str):
                extra_info = json.loads(extra_info)
            return CGMRLLMEnv(entry=extra_info)

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
                sandbox["r2e_ds_json"] = str(ds_path)
            cfg = SandboxConfig(**sandbox)
            issue = dict(self.entry.get("issue") or {})
            return PlannerEnv(issue=issue, sandbox_cfg=cfg)

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

