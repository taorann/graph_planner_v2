"""Train the Graph Planner exclusively with GRPO while treating CGM as a tool."""

from __future__ import annotations

import argparse
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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

_REPO_ROOT = Path(__file__).resolve().parents[1]


os.environ.setdefault("PYTHONPATH", str(_REPO_ROOT))

from omegaconf import DictConfig, OmegaConf

import ray

from graph_planner.infra.config import resolve_repo_path
from graph_planner.infra.vendor import ensure_rllm_importable
from graph_planner.integrations.rllm.shared_actors import CGMTool, PlannerEngine

LOGGER = logging.getLogger("planner_grpo.train")


@lru_cache(maxsize=1)
def _load_rllm_bindings() -> tuple[Any, Any, Any, str]:
    """Import rLLM bindings lazily to avoid heavy startup cost."""

    if not ensure_rllm_importable():  # pragma: no cover - defensive guard
        raise RuntimeError("Unable to import vendored rLLM modules")

    from graph_planner.integrations.rllm import (
        GRAPH_PLANNER_DATASET_NAME,
        GraphPlannerRLLMAgent,
        GraphPlannerRLLMEnv,
        ensure_dataset_registered,
    )

    return GraphPlannerRLLMAgent, GraphPlannerRLLMEnv, ensure_dataset_registered, GRAPH_PLANNER_DATASET_NAME


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


def _to_str_dict(payload: Mapping[str, Any] | None) -> dict[str, str]:
    if not payload:
        return {}
    result: dict[str, str] = {}
    for key, value in payload.items():
        if value is None:
            continue
        result[str(key)] = str(value)
    return result


def _collect_env_section(cfg: DictConfig, section: str) -> tuple[dict[str, str], bool]:
    node = OmegaConf.select(cfg, section, default=None)
    env: dict[str, str] = {}
    propagate = False
    if node is None:
        return env, propagate
    if isinstance(node, DictConfig):
        node = OmegaConf.to_container(node, resolve=True)
    if isinstance(node, Mapping):
        env.update(_to_str_dict(node.get("env")))
        propagate = bool(node.get("propagate_via_ray", False))
    return env, propagate


def _resolve_planner_env(cfg: DictConfig) -> tuple[dict[str, str], bool]:
    env, propagate = _collect_env_section(cfg, "graph_planner.planner")
    planner_path = resolve_repo_path(OmegaConf.select(cfg, "paths.planner_model", default=None))
    tokenizer_path = resolve_repo_path(
        OmegaConf.select(cfg, "paths.planner_tokenizer", default=None)
    )
    if planner_path:
        OmegaConf.update(cfg, "paths.planner_model", planner_path, merge=False)
    if planner_path:
        env.setdefault("PLANNER_MODEL_PATH", str(planner_path))
    if tokenizer_path:
        OmegaConf.update(cfg, "paths.planner_tokenizer", tokenizer_path, merge=False)
        env.setdefault("PLANNER_MODEL_TOKENIZER_PATH", str(tokenizer_path))
    return env, propagate


def _ensure_runtime_env(
    cfg: DictConfig, propagate_env: dict[str, str] | None = None
) -> dict[str, Any]:
    runtime_env = OmegaConf.to_container(
        OmegaConf.select(cfg, "ray.runtime_env"), resolve=True
    )
    runtime_env = dict(runtime_env or {})
    env_vars = dict(runtime_env.get("env_vars") or {})
    for key, value in _VLLM_ENV_DEFAULTS.items():
        env_vars.setdefault(key, value)
    env_vars.setdefault("PYTHONPATH", os.environ.get("PYTHONPATH", str(_REPO_ROOT)))
    if propagate_env:
        env_vars.update(propagate_env)
    runtime_env["env_vars"] = env_vars
    OmegaConf.update(cfg, "ray.runtime_env", runtime_env, merge=False)
    return runtime_env


def _get_or_create_actor(name: str, cls):
    try:
        return ray.get_actor(name)
    except Exception:
        return cls.options(name=name, lifetime="detached").remote()


def _maybe_wrap_save_with_reload(trainer) -> None:
    """Wrap trainer save to refresh the shared planner actor."""
    if trainer is None:
        return

    try:
        planner_actor = ray.get_actor("planner_engine")
    except Exception:
        return

    save_attr = None
    for candidate in ("save", "save_checkpoint"):
        method = getattr(trainer, candidate, None)
        if callable(method):
            save_attr = candidate
            break
    if save_attr is None:
        return

    original = getattr(trainer, save_attr)

    def _wrapped_save(*args, **kwargs):
        checkpoint_path = original(*args, **kwargs)
        target_path = checkpoint_path if isinstance(checkpoint_path, str) else os.environ.get("PLANNER_MODEL_PATH")
        if target_path:
            try:
                ray.get(planner_actor.reload_from.remote(target_path))
                print(f"[SYNC] planner_engine reloaded from: {target_path}")
            except Exception as exc:  # pragma: no cover - best effort logging
                print(f"[SYNC] planner_engine reload failed: {exc}")
        return checkpoint_path

    setattr(trainer, save_attr, _wrapped_save)


def _resolve_cgm_env(cfg: DictConfig) -> tuple[dict[str, str], bool]:
    env, propagate = _collect_env_section(cfg, "graph_planner.cgm")
    cgm_model = resolve_repo_path(OmegaConf.select(cfg, "paths.cgm_model"))
    if cgm_model:
        OmegaConf.update(cfg, "paths.cgm_model", cgm_model, merge=False)
    if not cgm_model and "CGM_MODEL_PATH" not in env:
        raise ValueError("paths.cgm_model must be specified for CGM inference")
    cgm_tokenizer = resolve_repo_path(
        OmegaConf.select(cfg, "paths.cgm_tokenizer", default=cgm_model)
    )
    if cgm_tokenizer:
        OmegaConf.update(cfg, "paths.cgm_tokenizer", cgm_tokenizer, merge=False)
    if cgm_model:
        env.setdefault("CGM_MODEL_PATH", str(cgm_model))
    if cgm_tokenizer:
        env.setdefault("CGM_TOKENIZER_PATH", str(cgm_tokenizer))
    env.setdefault("CGM_ENABLED", "1")
    return env, propagate


def _register_datasets(train_files: Sequence[str], val_files: Sequence[str]) -> None:
    (_, _, ensure_dataset_registered, dataset_name) = _load_rllm_bindings()

    seen: set[tuple[str, str]] = set()
    for split, files in (("train", train_files), ("val", val_files)):
        for file_path in files:
            abs_path = resolve_repo_path(file_path) if file_path else file_path
            file_path = abs_path or file_path
            key = (split, file_path)
            if key in seen:
                continue
            ensure_dataset_registered(
                name=dataset_name,
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


def _log_paths(
    train_files: Sequence[str],
    val_files: Sequence[str],
    model_path: str,
    cgm_model: str,
) -> None:
    LOGGER.info("Planner model path: %s", model_path)
    LOGGER.info("CGM model path: %s", cgm_model)
    LOGGER.info("Training files (%d): %s", len(train_files), ", ".join(train_files))
    LOGGER.info("Validation files (%d): %s", len(val_files), ", ".join(val_files))


def _resolve_ray_address(args: argparse.Namespace, cfg: DictConfig) -> str | None:
    for value in (
        getattr(args, "ray_address", None),
        os.environ.get("RAY_ADDRESS"),
        OmegaConf.select(cfg, "ray.address", default=None),
    ):
        if value is None:
            continue
        normalised = str(value).strip()
        if not normalised:
            continue
        lowered = normalised.lower()
        if lowered == "local":
            return None
        if lowered == "auto":
            return "auto"
        return normalised
    return None


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

    planner_env, planner_propagate = _resolve_planner_env(cfg)
    cgm_env, cgm_propagate = _resolve_cgm_env(cfg)
    for env_map in (planner_env, cgm_env):
        for key, value in env_map.items():
            os.environ.setdefault(key, value)
    _log_paths(
        train_files,
        val_files,
        model_path,
        cgm_model=cgm_env.get("CGM_MODEL_PATH", "<missing>"),
    )

    propagate_env: dict[str, str] = {}
    if planner_propagate:
        propagate_env.update(planner_env)
    if cgm_propagate:
        propagate_env.update(cgm_env)

    runtime_env = _ensure_runtime_env(cfg, propagate_env=propagate_env or None)
    LOGGER.info(
        "Ray runtime env vars: %s",
        runtime_env.get("env_vars", {}),
    )
    LOGGER.info(
        "Local planner env vars: %s",
        {k: planner_env[k] for k in sorted(planner_env)},
    )
    LOGGER.info(
        "Local CGM env vars: %s",
        {k: cgm_env[k] for k in sorted(cgm_env)},
    )

    if args.print_config:
        print(OmegaConf.to_yaml(cfg))
        return

    if args.dry_run:
        LOGGER.info("Dry run requested; skipping Ray initialisation and training launch.")
        return

    address = _resolve_ray_address(args, cfg)

    if address is None:
        LOGGER.info(
            "No Ray address provided via CLI/env/config; starting a fresh local Ray runtime."
        )
    else:
        LOGGER.info("Connecting to Ray using address=%s", address)

    ray.init(address=address, runtime_env=runtime_env or None, ignore_reinit_error=False)

    try:
        planner_engine = _get_or_create_actor("planner_engine", PlannerEngine)
        cgm_tool = _get_or_create_actor("cgm_tool", CGMTool)
        _ = (planner_engine, cgm_tool)
        print("[INIT] Shared actors ready: planner_engine & cgm_tool")

        _register_datasets(train_files, val_files)

        from rllm.trainer.agent_trainer import AgentTrainer  # noqa: WPS433

        (agent_cls, env_cls, _, _) = _load_rllm_bindings()

        trainer = AgentTrainer(
            agent_class=agent_cls,
            env_class=env_cls,
            config=cfg,
        )
        _maybe_wrap_save_with_reload(trainer)

        LOGGER.info("Starting GRPO training loop ...")
        trainer.train()
        LOGGER.info("Training finished successfully.")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
