"""Train the Graph Planner exclusively with GRPO while treating CGM as a tool."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

# ---------------------------------------------------------------------------
# Ensure vLLM v1 behaviour is enabled before importing any engine wrappers.
# ---------------------------------------------------------------------------
_VLLM_ENV_DEFAULTS = {
    "VLLM_USE_V1": "1",
    "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
    "VLLM_ENGINE_ITERATION_TIMEOUT_S": "100000000000",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
}

for _key, _value in _VLLM_ENV_DEFAULTS.items():
    os.environ.setdefault(_key, _value)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import ray
from omegaconf import DictConfig, OmegaConf

from graph_planner.infra.vendor import ensure_rllm_importable

ensure_rllm_importable()

from graph_planner.integrations.rllm import (  # noqa: E402
    GRAPH_PLANNER_DATASET_NAME,
    GraphPlannerRLLMAgent,
    GraphPlannerRLLMEnv,
    ensure_dataset_registered,
)

LOGGER = logging.getLogger("planner_grpo.train")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to configs/experiments/planner_grpo_4gpu.yaml",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=None,
        help="Optional OmegaConf dotlist overrides (e.g. trainer.total_epochs=10)",
    )
    parser.add_argument(
        "--ray-address",
        default=None,
        help="Optional Ray address for connecting to an existing cluster.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved configuration and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the configuration without launching training.",
    )
    return parser.parse_args()


def _listify(value: Any | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [str(value)]
    if isinstance(value, dict):
        # When loading from YAML a dict may appear due to OmegaConf interpolation.
        return [str(v) for v in value.values()]
    if isinstance(value, Iterable):
        return [str(item) for item in value]
    return [str(value)]


def _load_config(path: Path, overrides: Sequence[str] | None) -> DictConfig:
    cfg = OmegaConf.load(path)
    OmegaConf.set_struct(cfg, False)
    if overrides:
        override_cfg = OmegaConf.from_dotlist(list(overrides))
        cfg = OmegaConf.merge(cfg, override_cfg)
    return cfg


def _ensure_runtime_env(cfg: DictConfig) -> dict[str, Any]:
    runtime_env = OmegaConf.to_container(
        OmegaConf.select(cfg, "ray.runtime_env"), resolve=True
    )
    runtime_env = dict(runtime_env or {})
    env_vars = dict(runtime_env.get("env_vars") or {})
    for key, value in _VLLM_ENV_DEFAULTS.items():
        env_vars.setdefault(key, value)
    runtime_env["env_vars"] = env_vars
    OmegaConf.update(cfg, "ray.runtime_env", runtime_env, merge=False)
    return runtime_env


def _register_datasets(train_files: Sequence[str], val_files: Sequence[str]) -> None:
    seen: set[tuple[str, str]] = set()
    for split, files in (("train", train_files), ("val", val_files)):
        for file_path in files:
            key = (split, file_path)
            if key in seen:
                continue
            ensure_dataset_registered(
                name=GRAPH_PLANNER_DATASET_NAME,
                split=split,
                path=file_path,
            )
            seen.add(key)


def _assert_fsdp(cfg: DictConfig) -> None:
    actor_strategy = str(OmegaConf.select(cfg, "actor_rollout_ref.actor.strategy") or "").lower()
    if actor_strategy != "fsdp":
        raise ValueError(
            "actor_rollout_ref.actor.strategy must be 'fsdp' for planner-only GRPO"
        )
    ref_strategy = OmegaConf.select(cfg, "actor_rollout_ref.ref.strategy")
    if ref_strategy is not None and str(ref_strategy).lower() != "fsdp":
        raise ValueError("Reference policy must use FSDP when defined")
    tensor_parallel = int(OmegaConf.select(cfg, "parallel.tensor_parallel_planner", default=1))
    if tensor_parallel != 1:
        raise ValueError("tensor_parallel_planner must be 1 when training with FSDP")
    rollout_tp = int(OmegaConf.select(cfg, "actor_rollout_ref.rollout.tensor_model_parallel_size", default=1))
    if rollout_tp != 1:
        LOGGER.warning(
            "Overriding rollout tensor model parallel size to 1 for compatibility with FSDP"
        )
        OmegaConf.update(
            cfg,
            "actor_rollout_ref.rollout.tensor_model_parallel_size",
            1,
            merge=False,
        )


def _log_batch(cfg: DictConfig) -> None:
    dp_world = int(OmegaConf.select(cfg, "trainer.n_gpus_per_node", default=1))
    dp_world *= int(OmegaConf.select(cfg, "trainer.nnodes", default=1))
    per_gpu_micro = OmegaConf.select(
        cfg,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu",
        default=None,
    )
    if per_gpu_micro is None:
        per_gpu_micro = OmegaConf.select(cfg, "actor_rollout_ref.actor.ppo_micro_batch_size", default=1)
    per_gpu_micro = int(per_gpu_micro or 1)
    grad_accum = int(OmegaConf.select(cfg, "trainer.gradient_accumulation_steps", default=1))
    real_batch = per_gpu_micro * dp_world * grad_accum
    configured_batch = int(OmegaConf.select(cfg, "data.train_batch_size", default=real_batch))
    LOGGER.info(
        "Batch configuration -> dp_world=%s micro=%s grad_accum=%s => global=%s",
        dp_world,
        per_gpu_micro,
        grad_accum,
        real_batch,
    )
    if configured_batch != real_batch:
        LOGGER.warning(
            "data.train_batch_size=%s differs from computed global batch size %s",
            configured_batch,
            real_batch,
        )


def _log_paths(train_files: Sequence[str], val_files: Sequence[str], model_path: str) -> None:
    LOGGER.info("Planner model path: %s", model_path)
    LOGGER.info("Training files (%d): %s", len(train_files), ", ".join(train_files))
    LOGGER.info("Validation files (%d): %s", len(val_files), ", ".join(val_files))


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    cfg = _load_config(args.config, args.overrides)

    train_files = _listify(OmegaConf.select(cfg, "data.train_files"))
    val_files = _listify(OmegaConf.select(cfg, "data.val_files"))
    if not train_files:
        raise ValueError("data.train_files must not be empty")
    if not val_files:
        raise ValueError("data.val_files must not be empty")

    model_path = str(OmegaConf.select(cfg, "actor_rollout_ref.model.path") or "")
    if not model_path:
        raise ValueError("actor_rollout_ref.model.path must be specified")

    _assert_fsdp(cfg)
    _log_batch(cfg)
    _log_paths(train_files, val_files, model_path)

    runtime_env = _ensure_runtime_env(cfg)
    LOGGER.info("Ray runtime env vars: %s", runtime_env.get("env_vars", {}))

    if args.print_config:
        print(OmegaConf.to_yaml(cfg))
        return

    if args.dry_run:
        LOGGER.info("Dry run requested; skipping Ray initialisation and training launch.")
        return

    address = args.ray_address or OmegaConf.select(cfg, "ray.address", default=None)
    ray.init(address=address, runtime_env=runtime_env or None, ignore_reinit_error=True)

    try:
        _register_datasets(train_files, val_files)

        from rllm.trainer.agent_trainer import AgentTrainer  # noqa: WPS433

        trainer = AgentTrainer(
            agent_class=GraphPlannerRLLMAgent,
            env_class=GraphPlannerRLLMEnv,
            config=cfg,
        )

        LOGGER.info("Starting GRPO training loop ...")
        trainer.train()
        LOGGER.info("Training finished successfully.")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
