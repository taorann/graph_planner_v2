"""RLLM integration helpers with lazy imports to avoid circular dependencies."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from ...infra.vendor import ensure_rllm_importable
from .agent import GraphPlannerRLLMAgent
from .cgm_agent import CGMRLLMAgent
from .registry import register_rllm_components

ensure_rllm_importable()

try:  # Best-effort eager registration for Hydra-driven entrypoints
    from .env import GraphPlannerRLLMEnv as _GraphPlannerRLLMEnv  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    _GraphPlannerRLLMEnv = None
    GraphPlannerRLLMEnv = None  # type: ignore[assignment]
else:
    register_rllm_components(GraphPlannerRLLMAgent, _GraphPlannerRLLMEnv, name="graph_planner_repoenv")
    GraphPlannerRLLMEnv = _GraphPlannerRLLMEnv

try:
    from .cgm_env import CGMRLLMEnv as _CGMRLLMEnv  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    _CGMRLLMEnv = None
    CGMRLLMEnv = None  # type: ignore[assignment]
else:
    register_rllm_components(CGMRLLMAgent, _CGMRLLMEnv, name="graph_planner_cgm")
    CGMRLLMEnv = _CGMRLLMEnv

__all__ = [
    "GraphPlannerRLLMAgent",
    "GraphPlannerRLLMEnv",
    "CGMRLLMAgent",
    "CGMRLLMEnv",
    "GRAPH_PLANNER_DATASET_NAME",
    "GRAPH_PLANNER_CGM_DATASET_NAME",
    "load_task_entries",
    "register_dataset_from_file",
    "ensure_dataset_registered",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - trivial dispatcher
    """延迟导入 rLLM 组件，避免未安装依赖时报错。"""

    if name == "GraphPlannerRLLMEnv":
        module = import_module("graph_planner.integrations.rllm.env")
        env_cls = module.GraphPlannerRLLMEnv
        register_rllm_components(GraphPlannerRLLMAgent, env_cls, name="graph_planner_repoenv")
        return env_cls
    if name == "CGMRLLMEnv":
        module = import_module("graph_planner.integrations.rllm.cgm_env")
        env_cls = module.CGMRLLMEnv
        register_rllm_components(CGMRLLMAgent, env_cls, name="graph_planner_cgm")
        return env_cls
    if name in {
        "GRAPH_PLANNER_DATASET_NAME",
        "GRAPH_PLANNER_CGM_DATASET_NAME",
        "load_task_entries",
        "register_dataset_from_file",
        "ensure_dataset_registered",
    }:
        module = import_module("graph_planner.integrations.rllm.dataset")
        return getattr(module, name)
    raise AttributeError(name)
