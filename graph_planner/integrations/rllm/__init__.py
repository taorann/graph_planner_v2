"""RLLM integration helpers with lazy imports to avoid circular dependencies."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .agent import GraphPlannerRLLMAgent

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
        return module.GraphPlannerRLLMEnv
    if name in {
        "GRAPH_PLANNER_DATASET_NAME",
        "load_task_entries",
        "register_dataset_from_file",
        "ensure_dataset_registered",
    }:
        module = import_module("graph_planner.integrations.rllm.dataset")
        return getattr(module, name)
    raise AttributeError(name)
