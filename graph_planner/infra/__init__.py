"""Infrastructure helpers for configuration, parallel orchestration, and telemetry."""

from . import telemetry
from .config import (
    DEFAULT_TRAIN_DATASET,
    DEFAULT_VAL_DATASET,
    TrainingRunConfig,
    build_cli_overrides,
    default_training_run_config,
    load,
    load_run_config_file,
    merge_run_config,
    serialise_resolved_config,
    update_args_from_config,
)
from .metrics import init_wandb, log_metrics, make_gpu_snapshot, make_ray_snapshot
from .parallel import preflight_check, resolve_parallel

__all__ = [
    "telemetry",
    "DEFAULT_TRAIN_DATASET",
    "DEFAULT_VAL_DATASET",
    "TrainingRunConfig",
    "build_cli_overrides",
    "default_training_run_config",
    "load",
    "load_run_config_file",
    "merge_run_config",
    "serialise_resolved_config",
    "update_args_from_config",
    "preflight_check",
    "resolve_parallel",
    "init_wandb",
    "log_metrics",
    "make_gpu_snapshot",
    "make_ray_snapshot",
]
