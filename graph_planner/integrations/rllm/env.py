"""rLLM-compatible environment wrapper around :mod:`graph_planner.env.planner_env`."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Tuple
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
        _ENV_IMPORT_ERROR = _exc
    else:
        _ENV_IMPORT_ERROR = None
else:
    _ENV_IMPORT_ERROR = None

from ...env.planner_env import PlannerEnv
from ...runtime.sandbox import SandboxConfig


if BaseEnv is None:

    class GraphPlannerRLLMEnv:  # type: ignore[misc]
        """Placeholder that surfaces an actionable import error when rLLM is missing."""

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("rLLM is required to use GraphPlannerRLLMEnv") from _ENV_IMPORT_ERROR

else:

    class GraphPlannerRLLMEnv(BaseEnv):
        """Adapter that exposes :class:`PlannerEnv` through the rLLM ``BaseEnv`` API."""

        def __init__(
            self,
            entry: Dict[str, Any],
            *,
            max_steps: int = 8,
            reward_scale: float | None = None,
            failure_penalty: float | None = None,
            step_penalty: float | None = None,
            timeout_penalty: float | None = None,
            repo_operation_limit: int | None = None,
            enable_cgm_synthesis: bool | None = None,
            synthesis_strategy: str | None = None,
        ) -> None:
            self.entry = deepcopy(entry)
            self.max_steps = int(entry.get("max_steps", max_steps))
            base_reward_scale = float(entry.get("reward_scale", 1.0))
            self.reward_scale = float(
                reward_scale if reward_scale is not None else base_reward_scale
            )
            base_failure_penalty = float(entry.get("failure_penalty", 0.0))
            self.failure_penalty = float(
                failure_penalty
                if failure_penalty is not None
                else base_failure_penalty
            )
            base_step_penalty = float(entry.get("step_penalty", 0.0))
            self.step_penalty = float(
                step_penalty if step_penalty is not None else base_step_penalty
            )
            base_timeout_penalty = float(entry.get("timeout_penalty", 0.0))
            self.timeout_penalty = float(
                timeout_penalty
                if timeout_penalty is not None
                else base_timeout_penalty
            )
            entry_repo_limit = entry.get("repo_operation_limit") or entry.get(
                "repo_op_limit"
            )
            limit_value = (
                repo_operation_limit
                if repo_operation_limit is not None
                else entry_repo_limit
            )
            self.repo_operation_limit = int(limit_value) if limit_value else None
            entry_enable_cgm = bool(entry.get("enable_cgm_synthesis", True))
            if enable_cgm_synthesis is None:
                self.enable_cgm_synthesis = entry_enable_cgm
            else:
                self.enable_cgm_synthesis = bool(enable_cgm_synthesis)
            self.synthesis_strategy = (
                synthesis_strategy
                if synthesis_strategy is not None
                else entry.get("synthesis_strategy")
            )
            self._planner: PlannerEnv | None = None
            self._last_observation: Dict[str, Any] | None = None
            self._issue_uid = uuid4().hex
            issue = self.entry.get("issue") or {}
            self._source_issue_id = str(issue.get("id") or "")
            self._step_count = 0

        # ------------------------------------------------------------------
        # BaseEnv interface
        # ------------------------------------------------------------------
        def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            self.close()
            self._planner = self._spawn_planner()
            observation = self._planner.reset()
            self._last_observation = observation
            self._step_count = 0
            info = {
                "task_id": self.entry.get("task_id"),
                "issue": self.entry.get("issue", {}),
                "max_steps": self.max_steps,
                "issue_uid": self._issue_uid,
                "source_issue_id": self._source_issue_id,
                "env_config": {
                    "reward_scale": self.reward_scale,
                    "failure_penalty": self.failure_penalty,
                    "step_penalty": self.step_penalty,
                    "timeout_penalty": self.timeout_penalty,
                    "repo_operation_limit": self.repo_operation_limit,
                    "enable_cgm_synthesis": self.enable_cgm_synthesis,
                    "synthesis_strategy": self.synthesis_strategy,
                },
            }
            return observation, info

        def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
            if self._planner is None:
                raise RuntimeError("Environment has not been reset before step().")
            observation, reward, done, info = self._planner.step(action)
            info = info or {}
            info.setdefault("max_steps", self.max_steps)
            self._last_observation = observation
            self._step_count += 1

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

            if done:
                if reward <= 0:
                    adjusted -= self.failure_penalty
                if limit_triggered and adjusted <= 0:
                    adjusted -= self.timeout_penalty

            reward = adjusted
            return observation, reward, done, info

        def close(self) -> None:
            if self._planner is not None:
                try:
                    self._planner.close()
                finally:
                    self._planner = None
                    self._last_observation = None
            self._step_count = 0

        def compute_final_reward(self) -> float:
            if self._planner is None:
                return 0.0
            info = self._planner.last_info or {}
            tests = info.get("tests") or info.get("submit", {})
            if isinstance(tests, dict) and tests.get("passed"):
                base = 1.0 * self.reward_scale
                return base
            penalty = self.failure_penalty
            if self._step_count >= self.max_steps:
                penalty += self.timeout_penalty
            return -(penalty)

        @staticmethod
        def from_dict(extra_info: Dict[str, Any] | str) -> "GraphPlannerRLLMEnv":
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
                    "enable_cgm_synthesis",
                    "synthesis_strategy",
                ):
                    if key in extra_info:
                        env_kwargs[key] = extra_info[key]
                if "repo_op_limit" in env_kwargs and "repo_operation_limit" not in env_kwargs:
                    env_kwargs["repo_operation_limit"] = env_kwargs.pop("repo_op_limit")
                else:
                    env_kwargs.pop("repo_op_limit", None)
            raw_entry = extra_info.get("raw_entry_json") if isinstance(extra_info, dict) else None
            if raw_entry:
                extra_payload = json.loads(raw_entry)
            else:
                extra_payload = extra_info
            max_steps = int(extra_payload.get("max_steps", 8))
            return GraphPlannerRLLMEnv(entry=extra_payload, max_steps=max_steps, **env_kwargs)

        # ------------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------------
        def _spawn_planner(self) -> PlannerEnv:
            sandbox_dict = dict(self.entry.get("sandbox") or {})
            if not sandbox_dict:
                raise ValueError("Sandbox configuration is required for GraphPlannerRLLMEnv")
            sandbox_dict.setdefault("mounts", {})
            sandbox_dict.setdefault("env", {})
            ds_path = sandbox_dict.get("r2e_ds_json")
            if ds_path:
                sandbox_dict["r2e_ds_json"] = str(Path(ds_path).expanduser().resolve())
            sandbox_cfg = SandboxConfig(**sandbox_dict)
            issue = self._build_issue_payload()
            return PlannerEnv(issue=issue, sandbox_cfg=sandbox_cfg)

        def _build_issue_payload(self) -> Dict[str, Any]:
            issue = deepcopy(self.entry.get("issue") or {})
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

        @property
        def planner(self) -> PlannerEnv | None:
            return self._planner

        @property
        def last_observation(self) -> Dict[str, Any] | None:
            return self._last_observation
