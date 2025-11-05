"""Train the Graph Planner exclusively with GRPO while treating CGM as a tool."""

from __future__ import annotations

import argparse
import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

# ---------------------------------------------------------------------------
# Ensure vLLM v1 behaviour is enabled before importing any engine wrappers.
# ---------------------------------------------------------------------------
_VLLM_ENV_DEFAULTS = {
    "VLLM_USE_V1": "1",
    "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
    "VLLM_ENGINE_ITERATION_TIMEOUT_S": "100000000000",
}

for _key, _value in _VLLM_ENV_DEFAULTS.items():
    os.environ.setdefault(_key, _value)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

_REPO_ROOT = Path(__file__).resolve().parents[1]


os.environ.setdefault("PYTHONPATH", str(_REPO_ROOT))

from omegaconf import DictConfig, OmegaConf

import ray

from graph_planner.infra.config import resolve_repo_path
from graph_planner.infra.vendor import ensure_rllm_importable
from graph_planner.integrations.rllm.shared_actors import CGMTool, PlannerEngine
from graph_planner.integrations.rllm.dataset import ensure_dataset_registered as ensure_verl_parquet

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


def _iter_gpu_ids(raw: Any) -> Iterable[int]:
    if raw is None:
        return []
    if isinstance(raw, str):
        tokens = [tok.strip() for tok in raw.split(",") if tok.strip()]
    else:
        try:
            tokens = list(raw)
        except TypeError:
            tokens = [raw]
    for tok in tokens:
        try:
            yield int(tok)
        except (TypeError, ValueError):
            continue


def _count_topology_gpus(cfg: DictConfig) -> int:
    groups = OmegaConf.select(cfg, "system.topology.groups", default=None)
    if isinstance(groups, DictConfig):
        groups = OmegaConf.to_container(groups, resolve=True)
    gpu_ids: set[int] = set()
    if isinstance(groups, Mapping):
        for group in groups.values():
            if isinstance(group, Mapping):
                for gid in _iter_gpu_ids(group.get("gpus")):
                    gpu_ids.add(gid)
    return len(gpu_ids)


def _should_spawn_shared_actors(cfg: DictConfig) -> bool:
    env_flag = os.environ.get("GRAPH_PLANNER_SPAWN_SHARED_ACTORS")
    if env_flag is not None:
        return env_flag.strip().lower() in {"1", "true", "yes", "on"}
    return bool(OmegaConf.select(cfg, "graph_planner.shared_actors.spawn", default=False))


def _ensure_shared_actor(name: str, cls, *, spawn_if_missing: bool):
    if not ray.is_initialized():
        return None
    try:
        actor = ray.get_actor(name)
    except Exception:
        actor = None

    if actor is not None:
        return actor

    if not spawn_if_missing:
        print(f"[shared-actors] {name} not running; skipping spawn (disabled)")
        return None

    try:
        actor = cls.options(name=name, lifetime="detached").remote()
        print(f"[shared-actors] spawned {name}")
        return actor
    except Exception as exc:
        print(f"[shared-actors] failed to spawn {name}: {exc}")
        return None


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


def _maybe_materialize_json_to_verl_parquet(config):
    """
    For each file in data.train_files / val_files:
      - if suffix in {json,jsonl} -> normalize and write *_verl.parquet
      - replace cfg paths with the materialized parquet path
    Also prints a sample of container runtime config (docker image, workdir, mounts) from the parquet,
    so users can confirm the container will launch as intended.
    """

    def _process_list(name: str, lst):
        new_paths = []
        for i, p in enumerate(lst or []):
            p = str(p)
            low = p.lower()
            if low.endswith(".json") or low.endswith(".jsonl") or low.endswith(".jl"):
                ds = ensure_verl_parquet(
                    name="graph_planner_repoenv",
                    split=None,  # infer from file name
                    path=p,
                )
                print(
                    f"[DATA] {name}[{i}] JSON→Parquet: {p} -> {ds.get_verl_data_path()} (rows={ds.num_rows})"
                )
                new_paths.append(ds.get_verl_data_path())
            else:
                new_paths.append(p)
        return new_paths

    # mutate config in-place
    tf = OmegaConf.select(config, "data.train_files") or []
    vf = OmegaConf.select(config, "data.val_files") or []

    tf2 = _process_list("train_files", tf)
    vf2 = _process_list("val_files", vf)

    # write back
    if "data" not in config:
        config["data"] = {}
    config["data"]["train_files"] = tf2
    config["data"]["val_files"] = vf2

    # DEBUG: peek one row and print container args
    try:
        from datasets import load_dataset

        sample_path = (tf2 or vf2)[0]
        ds = load_dataset("parquet", data_files=sample_path, split="train")
        if len(ds) > 0 and "extra_info" in ds.column_names:
            ex = ds[0]
            extra = ex["extra_info"]
            if isinstance(extra, str):
                extra = json.loads(extra)
            sb = (extra or {}).get("sandbox", {})
            print(
                "[DATA→ENV] sample sandbox:",
                {
                    "backend": sb.get("backend"),
                    "docker_image": sb.get("docker_image"),
                    "workdir": sb.get("workdir"),
                    "mounts": sb.get("mounts"),
                    "env.size": len((sb.get("env") or {})),
                },
            )
    except Exception as e:  # pragma: no cover - best effort logging
        print("[WARN] Failed to peek sample sandbox info:", repr(e))


def _maybe_prepull_docker_images(config):
    import os as _os
    import subprocess

    if _os.environ.get("PREPULL_DOCKER", "0") != "1":
        return
    from datasets import load_dataset

    train_paths = OmegaConf.select(config, "data.train_files") or []
    val_paths = OmegaConf.select(config, "data.val_files") or []
    paths = list(train_paths) + list(val_paths)
    images = set()
    for p in paths:
        try:
            ds = load_dataset("parquet", data_files=p, split="train")
            for ex in ds.select(range(min(1000, len(ds)))):
                extra = ex.get("extra_info")
                if isinstance(extra, str):
                    extra = json.loads(extra)
                img = (extra or {}).get("sandbox", {}).get("docker_image")
                if img:
                    images.add(img)
        except Exception:
            pass
    if not images:
        print("[PREPULL] No docker images found in dataset.")
        return
    print("[PREPULL] pulling:", images)
    for img in sorted(images):
        try:
            subprocess.run(["docker", "pull", img], check=False)
        except Exception as e:  # pragma: no cover - best effort logging
            print("[PREPULL] pull failed:", img, repr(e))


def _register_datasets(train_files: Sequence[str], val_files: Sequence[str]) -> None:
    (_, _, ensure_dataset_registered, dataset_name) = _load_rllm_bindings()

    seen: set[tuple[str, str]] = set()
    for split, files in (("train", train_files), ("val", val_files)):
        for file_path in files:
            abs_path = resolve_repo_path(file_path) if file_path else file_path
            file_path = abs_path or file_path
            if file_path and "://" not in str(file_path):
                suffix = Path(str(file_path)).suffix.lower()
                if suffix in {".json", ".jsonl", ".parquet"} and not Path(str(file_path)).expanduser().exists():
                    raise FileNotFoundError(
                        f"Dataset file not found: {file_path}. Run dataset preparation before training."
                    )
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
    rollout_tp_raw = OmegaConf.select(
        cfg, "actor_rollout_ref.rollout.tensor_model_parallel_size", default=None
    )
    if rollout_tp_raw in (None, "", 0):
        return

    try:
        rollout_tp = int(rollout_tp_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "actor_rollout_ref.rollout.tensor_model_parallel_size must be integer-compatible"
        ) from exc

    if rollout_tp <= 0:
        raise ValueError("rollout.tensor_model_parallel_size must be >= 1")

    planner_gpus = OmegaConf.select(cfg, "system.topology.groups.planner.gpus", default=None)
    planner_gpu_count = 0
    if planner_gpus:
        try:
            planner_gpu_count = len(list(planner_gpus))
        except TypeError:
            planner_gpu_count = 0
    if planner_gpu_count <= 0:
        planner_gpu_count = int(OmegaConf.select(cfg, "actor_rollout_ref.actor.fsdp_config.fsdp_size", default=0) or 0)

    if planner_gpu_count and (planner_gpu_count % rollout_tp != 0):
        raise ValueError(
            "rollout.tensor_model_parallel_size must divide planner GPU count; "
            f"got tp={rollout_tp}, planner_gpus={planner_gpu_count}"
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

    # materialize JSON/JSONL → *_verl.parquet and rewrite cfg paths
    _maybe_materialize_json_to_verl_parquet(cfg)
    _maybe_prepull_docker_images(cfg)

    train_files = _listify(OmegaConf.select(cfg, "data.train_files"))
    val_files = _listify(OmegaConf.select(cfg, "data.val_files"))

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

    spawn_shared_requested = _should_spawn_shared_actors(cfg)

    if args.print_config:
        print(OmegaConf.to_yaml(cfg))
        return

    if args.dry_run:
        LOGGER.info("Dry run requested; skipping Ray initialisation and training launch.")
        return

    address = _resolve_ray_address(args, cfg)
    
    if address is None:
        LOGGER.info("No Ray address provided via CLI/env/config; starting a fresh local Ray runtime.")
        # 强制本进程本地，不要自动发现
        ray.init(
            address="local",                 # ←← 这里以前是 None
            runtime_env=runtime_env or None,
            ignore_reinit_error=False,
            namespace="graph-planner",
        )
    else:
        LOGGER.info("Connecting to Ray using address=%s", address)
        ray.init(
            address=address,
            runtime_env=runtime_env or None,
            ignore_reinit_error=False,
            namespace="graph-planner",
        )


    try:
        spawn_shared = spawn_shared_requested
        if spawn_shared:
            topology_gpu_budget = _count_topology_gpus(cfg)
            try:
                cluster_gpus = int(ray.cluster_resources().get("GPU", 0))
            except Exception:
                cluster_gpus = 0
            if cluster_gpus and cluster_gpus <= topology_gpu_budget:
                print("[shared-actors] skipping spawn due to insufficient GPU resources")
                spawn_shared = False

        planner_engine = _ensure_shared_actor(
            "planner_engine",
            PlannerEngine,
            spawn_if_missing=spawn_shared,
        )
        cgm_tool = _ensure_shared_actor(
            "cgm_tool",
            CGMTool,
            spawn_if_missing=spawn_shared,
        )
        if planner_engine or cgm_tool:
            print("[INIT] Shared actors ready: planner_engine & cgm_tool")
        else:
            print("[INIT] Shared actors not attached (disabled or unavailable)")

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
