"""Utility helpers for Weights & Biases logging and runtime telemetry."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional


def init_wandb(
    *,
    enabled: bool,
    offline: bool,
    project: str,
    entity: Optional[str],
    run_name: str,
    config: Dict[str, Any],
) -> Optional[Any]:  # pragma: no cover - runtime side-effect
    """Initialise a W&B run if logging is enabled."""

    if not enabled:
        return None

    try:
        import wandb
    except Exception:  # pragma: no cover - wandb optional in tests
        return None

    mode = "offline" if offline else "online"
    run = wandb.init(project=project, entity=entity, name=run_name, mode=mode)
    try:
        wandb.config.update(config, allow_val_change=True)
    except Exception:  # pragma: no cover - guard against invalid config types
        pass

    watch_cfg = config.get("logging", {}).get("wandb", {}).get("watch", {})
    if watch_cfg and watch_cfg.get("enabled"):
        try:
            wandb.watch(
                None,
                log=watch_cfg.get("log", "all"),
                log_freq=int(watch_cfg.get("log_freq", 200)),
            )
        except Exception:  # pragma: no cover - optional
            pass
    return run


def make_gpu_snapshot() -> Callable[[], Dict[str, float]]:
    try:  # pragma: no cover - optional dependency
        import pynvml as nvml
        import torch

        nvml.nvmlInit()
        handles = [nvml.nvmlDeviceGetHandleByIndex(i) for i in range(torch.cuda.device_count())]

        def snapshot() -> Dict[str, float]:
            payload: Dict[str, float] = {}
            for idx, handle in enumerate(handles):
                util = nvml.nvmlDeviceGetUtilizationRates(handle).gpu
                mem = nvml.nvmlDeviceGetMemoryInfo(handle).used / 1e9
                payload[f"gpu/{idx}/util"] = float(util)
                payload[f"gpu/{idx}/mem_GB"] = round(float(mem), 2)
            return payload

        return snapshot
    except Exception:
        return lambda: {}


def make_ray_snapshot() -> Callable[[], Dict[str, float]]:
    try:  # pragma: no cover - requires ray runtime
        import ray

        def snapshot() -> Dict[str, float]:
            resources = ray.available_resources()
            return {
                "ray/cpus_avail": float(resources.get("CPU", 0.0)),
                "ray/gpus_avail": float(resources.get("GPU", 0.0)),
            }

        return snapshot
    except Exception:
        return lambda: {}


def log_metrics(step: int, metrics: Dict[str, Any]) -> None:  # pragma: no cover - telemetry helper
    try:
        import wandb

        wandb.log(metrics, step=step)
    except Exception:
        pass
