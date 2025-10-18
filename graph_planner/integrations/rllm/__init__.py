"""RLLM integration helpers with lazy imports to avoid circular dependencies."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from ...infra.vendor import ensure_rllm_importable
from .agent import GraphPlannerRLLMAgent
from .registry import register_rllm_components

ensure_rllm_importable()

try:  # Best-effort eager registration for Hydra-driven entrypoints
    from .env import GraphPlannerRLLMEnv as _GraphPlannerRLLMEnv  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    _GraphPlannerRLLMEnv = None
else:
    register_rllm_components(GraphPlannerRLLMAgent, _GraphPlannerRLLMEnv)
    GraphPlannerRLLMEnv = _GraphPlannerRLLMEnv

__all__ = [
    "GraphPlannerRLLMAgent",
    "GraphPlannerRLLMEnv",
    "GRAPH_PLANNER_DATASET_NAME",
    "load_task_entries",
    "register_dataset_from_file",
    "ensure_dataset_registered",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - trivial dispatcher
    if name == "GraphPlannerRLLMEnv":
        module = import_module("graph_planner.integrations.rllm.env")
        env_cls = module.GraphPlannerRLLMEnv
        register_rllm_components(GraphPlannerRLLMAgent, env_cls)
        return env_cls
    if name in {
        "GRAPH_PLANNER_DATASET_NAME",
        "load_task_entries",
        "register_dataset_from_file",
        "ensure_dataset_registered",
    }:
        module = import_module("graph_planner.integrations.rllm.dataset")
        return getattr(module, name)
    raise AttributeError(name)
