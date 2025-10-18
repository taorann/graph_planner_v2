"""rLLM-compatible environment wrapper around :mod:`graph_planner.env.planner_env`."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Tuple

from ...infra.vendor import ensure_rllm_importable

ensure_rllm_importable()

try:
    from rllm.environments.base.base_env import BaseEnv
except ImportError as _exc:  # pragma: no cover - optional dependency
    BaseEnv = None  # type: ignore[assignment]
    _ENV_IMPORT_ERROR = _exc
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

        def __init__(self, entry: Dict[str, Any], *, max_steps: int = 8) -> None:
            self.entry = deepcopy(entry)
            self.max_steps = int(entry.get("max_steps", max_steps))
            self._planner: PlannerEnv | None = None
            self._last_observation: Dict[str, Any] | None = None

        # ------------------------------------------------------------------
        # BaseEnv interface
        # ------------------------------------------------------------------
        def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            self.close()
            self._planner = self._spawn_planner()
            observation = self._planner.reset()
            self._last_observation = observation
            info = {
                "task_id": self.entry.get("task_id"),
                "issue": self.entry.get("issue", {}),
                "max_steps": self.max_steps,
            }
            return observation, info

        def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
            if self._planner is None:
                raise RuntimeError("Environment has not been reset before step().")
            observation, reward, done, info = self._planner.step(action)
            info = info or {}
            info.setdefault("max_steps", self.max_steps)
            self._last_observation = observation
            return observation, reward, done, info

        def close(self) -> None:
            if self._planner is not None:
                try:
                    self._planner.close()
                finally:
                    self._planner = None
                    self._last_observation = None

        def compute_final_reward(self) -> float:
            if self._planner is None:
                return 0.0
            info = self._planner.last_info or {}
            tests = info.get("tests") or info.get("submit", {})
            if isinstance(tests, dict) and tests.get("passed"):
                return 1.0
            return 0.0

        @staticmethod
        def from_dict(extra_info: Dict[str, Any] | str) -> "GraphPlannerRLLMEnv":
            if isinstance(extra_info, str):
                extra_info = json.loads(extra_info)
            max_steps = int(extra_info.get("max_steps", 8))
            return GraphPlannerRLLMEnv(entry=extra_info, max_steps=max_steps)

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
            issue = deepcopy(self.entry.get("issue") or {})
            return PlannerEnv(issue=issue, sandbox_cfg=sandbox_cfg)

        @property
        def planner(self) -> PlannerEnv | None:
            return self._planner

        @property
        def last_observation(self) -> Dict[str, Any] | None:
            return self._last_observation
