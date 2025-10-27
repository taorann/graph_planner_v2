"""Evaluate a planner-only GRPO checkpoint using the Graph Planner environment."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

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

LOGGER = logging.getLogger("planner_grpo.eval")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to the checkpoint directory")
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=None,
        help="Optional OmegaConf dotlist overrides",
    )
    parser.add_argument("--ray-address", default=None)
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for evaluation artifacts",
    )
    return parser.parse_args()


def _listify(value: Any | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [str(value)]
    if isinstance(value, dict):
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


def _register_datasets(val_files: Sequence[str]) -> None:
    seen: set[str] = set()
    for path in val_files:
        if path in seen:
            continue
        ensure_dataset_registered(
            name=GRAPH_PLANNER_DATASET_NAME,
            split="val",
            path=path,
        )
        seen.add(path)


def _prepare_eval_config(cfg: DictConfig, ckpt: Path, output_dir: Path | None) -> None:
    OmegaConf.update(cfg, "trainer.val_only", True, merge=False)
    OmegaConf.update(cfg, "trainer.val_before_train", True, merge=False)
    OmegaConf.update(cfg, "trainer.save_freq", -1, merge=False)
    OmegaConf.update(cfg, "trainer.test_freq", -1, merge=False)
    OmegaConf.update(cfg, "trainer.total_epochs", 0, merge=False)
    OmegaConf.update(cfg, "trainer.total_training_steps", 0, merge=False)
    OmegaConf.update(cfg, "trainer.resume_from_path", str(ckpt), merge=False)
    OmegaConf.update(cfg, "trainer.resume_mode", "resume_path", merge=False)
    val_batch = OmegaConf.select(cfg, "data.val_batch_size", default=None)
    if val_batch is not None:
        OmegaConf.update(cfg, "data.train_batch_size", val_batch, merge=False)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.update(cfg, "trainer.default_local_dir", str(output_dir), merge=False)
        OmegaConf.update(cfg, "trainer.output_dir", str(output_dir), merge=False)


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


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    cfg = _load_config(args.config, args.overrides)

    val_files = _listify(OmegaConf.select(cfg, "data.val_files"))
    if not val_files:
        raise ValueError("data.val_files must not be empty for evaluation")
    if not args.ckpt.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {args.ckpt}")
    LOGGER.info("Validation files (%d): %s", len(val_files), ", ".join(val_files))
    LOGGER.info("Checkpoint path: %s", args.ckpt)

    runtime_env = _ensure_runtime_env(cfg)
    LOGGER.info("Ray runtime env vars: %s", runtime_env.get("env_vars", {}))

    _prepare_eval_config(cfg, args.ckpt.resolve(), args.output_dir.resolve() if args.output_dir else None)
    _log_batch(cfg)

    if args.print_config:
        print(OmegaConf.to_yaml(cfg))
        return

    if args.dry_run:
        LOGGER.info("Dry run requested; skipping evaluation launch.")
        return

    address = args.ray_address or OmegaConf.select(cfg, "ray.address", default=None)
    ray.init(address=address, runtime_env=runtime_env or None, ignore_reinit_error=True)

    try:
        _register_datasets(val_files)

        from rllm.trainer.agent_trainer import AgentTrainer  # noqa: WPS433

        trainer = AgentTrainer(
            agent_class=GraphPlannerRLLMAgent,
            env_class=GraphPlannerRLLMEnv,
            config=cfg,
        )

        LOGGER.info("Running validation for checkpoint: %s", args.ckpt)
        trainer.train()
        LOGGER.info("Evaluation completed successfully.")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
