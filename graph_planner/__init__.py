"""Graph Planner core package consolidating agents, envs, and integrations."""

from importlib import import_module
from typing import Any

__all__ = ["agents", "core", "env", "integrations", "runtime"]


def __getattr__(name: str) -> Any:  # pragma: no cover - trivial lazy import
    if name in __all__:
        module = import_module(f"graph_planner.{name}")
        globals()[name] = module
        return module
    raise AttributeError(name)
