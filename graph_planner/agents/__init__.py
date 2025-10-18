"""Agent package aggregating available controllers."""

from typing import Any

from .model_based import LocalLLMPlannerAgent
from .rule_based.planner import PlannerAgent

__all__ = ["PlannerAgent", "LocalLLMPlannerAgent", "GraphPlannerRLLMAgent"]


def __getattr__(name: str) -> Any:  # pragma: no cover - trivial lazy import
    if name == "GraphPlannerRLLMAgent":
        from ..integrations.rllm.agent import GraphPlannerRLLMAgent as _Agent

        return _Agent
    raise AttributeError(name)
