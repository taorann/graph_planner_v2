"""Parallel configuration helpers for Graph Planner rLLM launches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass
class ParallelConfig:
    """Resolved parallel settings for planner/CGM + Ray resources."""

    tensor_parallel_planner: int
    tensor_parallel_cgm: int
    replicas: int
    parallel_agents: int
    rollout_workers: int
    workflow_parallel: int
    num_gpus: int
    num_nodes: int
    ray_num_gpus: int
    ray_num_cpus: int


def _lookup(mapping: Mapping[str, Any], *keys: str, default: int = 0) -> int:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return default
        current = current.get(key)
    if current is None:
        return default
    return int(current)


def resolve_parallel(config: Mapping[str, Any]) -> ParallelConfig:
    """Create a :class:`ParallelConfig` from merged run configuration."""

    parallel = config.get("parallel", {})
    resources = config.get("resources", {})
    return ParallelConfig(
        tensor_parallel_planner=_lookup(parallel, "tensor_parallel_planner", default=1),
        tensor_parallel_cgm=_lookup(parallel, "tensor_parallel_cgm", default=1),
        replicas=_lookup(parallel, "replicas", default=1),
        parallel_agents=_lookup(parallel, "parallel_agents", default=1),
        rollout_workers=_lookup(parallel, "rollout_workers", default=1),
        workflow_parallel=_lookup(parallel, "workflow_parallel", default=1),
        num_gpus=_lookup(resources, "num_gpus", default=1),
        num_nodes=_lookup(resources, "num_nodes", default=1),
        ray_num_gpus=_lookup(resources, "ray_num_gpus", default=1),
        ray_num_cpus=_lookup(resources, "ray_num_cpus", default=4),
    )


def preflight_check(pcfg: ParallelConfig) -> None:
    """Validate GPU/Ray resource assignments before launching training."""

    issues = []

    if pcfg.tensor_parallel_planner < 1 or pcfg.tensor_parallel_cgm < 1:
        issues.append("tensor parallel factors must be >= 1 for planner and CGM")

    total_gpus = pcfg.num_gpus * max(pcfg.num_nodes, 1)
    max_tp = max(pcfg.tensor_parallel_planner, pcfg.tensor_parallel_cgm)

    if pcfg.tensor_parallel_planner + pcfg.tensor_parallel_cgm > total_gpus:
        issues.append(
            "planner+CGM tensor parallel exceeds total GPUs; lower TP values or add GPUs/nodes"
        )

    if pcfg.replicas < 1:
        issues.append("replicas must be >= 1")

    if pcfg.replicas * max_tp > total_gpus:
        issues.append(
            "replicas * max(tensor_parallel) must fit within available GPUs; reduce replicas or TP"
        )

    if pcfg.workflow_parallel < max(pcfg.parallel_agents, pcfg.rollout_workers):
        issues.append(
            "workflow_parallel must be >= max(parallel_agents, rollout_workers); increase workflow_parallel"
        )

    if pcfg.ray_num_gpus < pcfg.replicas:
        issues.append(
            "Ray GPU budget smaller than replicas; raise ray_num_gpus or reduce replicas"
        )

    required_cpus = max(1, pcfg.rollout_workers) * 4
    if pcfg.ray_num_cpus < required_cpus:
        issues.append(
            "Ray CPU budget too small for rollout workers; increase ray_num_cpus or reduce workers"
        )

    if issues:
        formatted = "\n - ".join([""] + issues)
        raise ValueError(
            "Parallel resource configuration invalid:" + formatted +
            "\nSuggestions: decrease tensor parallel or replicas, reduce workflow concurrency, or allocate more Ray resources."
        )
