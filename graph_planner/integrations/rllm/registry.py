"""Runtime helpers for registering Graph Planner components with rLLM."""

from __future__ import annotations

from functools import lru_cache
from typing import Type

from ...infra.vendor import ensure_rllm_importable


@lru_cache(maxsize=None)
def register_rllm_components(
    agent_cls: Type[object],
    env_cls: Type[object],
    *,
    name: str = "graph_planner_repoenv",
) -> bool:
    """Inject Graph Planner agent/env classes into rLLM's registries."""

    if not ensure_rllm_importable():
        return False
    try:
        from rllm.trainer import env_agent_mappings as mapping
    except ImportError:  # pragma: no cover - optional dependency
        return False

    for container in (mapping.AGENT_CLASSES, mapping.AGENT_CLASS_MAPPING):
        if container.get(name) is agent_cls:
            continue
        container[name] = agent_cls
    for container in (mapping.ENV_CLASSES, mapping.ENV_CLASS_MAPPING):
        if container.get(name) is env_cls:
            continue
        container[name] = env_cls
    return True


__all__ = ["register_rllm_components"]
