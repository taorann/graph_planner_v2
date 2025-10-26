"""使用 rLLM 训练 Graph Planner/CGM 代理的命令行脚本。

English summary
    Provides a thin CLI that prepares datasets, config overrides and agent/env
    wiring before delegating to rLLM's GRPO-capable trainer.
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np
from omegaconf import OmegaConf, open_dict
from omegaconf.errors import ConfigKeyError

try:  # pragma: no cover - torch is an optional dependency for docs CI
    import torch
except Exception:  # pragma: no cover - torch missing on docs builds
    torch = None  # type: ignore[assignment]

from graph_planner.infra.config import (
    build_cli_overrides,
    default_training_run_config,
    load_run_config_file,
    merge_run_config,
    serialise_resolved_config,
    update_args_from_config,
)
from graph_planner.infra.metrics import (
    init_wandb,
    log_metrics,
    make_gpu_snapshot,
    make_ray_snapshot,
)
from graph_planner.infra.parallel import preflight_check, resolve_parallel
from graph_planner.infra.vendor import ensure_rllm_importable, find_in_rllm

ensure_rllm_importable()

from graph_planner.integrations.rllm import (  # noqa: E402
    CGMRLLMAgent,
    CGMRLLMEnv,
    GRAPH_PLANNER_CGM_DATASET_NAME,
    GRAPH_PLANNER_DATASET_NAME,
    GraphPlannerRLLMAgent,
    GraphPlannerRLLMEnv,
    ensure_dataset_registered,
    load_task_entries,
    resolve_task_file,
)
from graph_planner.runtime.containers import (  # noqa: E402
    collect_docker_images,
    load_docker_manifest,
    prepull_docker_images,
    write_docker_manifest,
)


LOGGER = logging.getLogger(__name__)

AgentTrainer = None  # type: ignore[assignment]

REPO_ROOT = Path(
    os.environ.get("GRAPH_PLANNER_ROOT") or Path(__file__).resolve().parent.parent
).expanduser().resolve()

DEFAULT_PLANNER_MODEL_PATH = (REPO_ROOT / "models" / "Qwen3-14B").resolve()
DEFAULT_CGM_MODEL_PATH = (REPO_ROOT / "models" / "CodeFuse-CGM").resolve()
DEFAULT_TRAIN_DATASET_PATH = (REPO_ROOT / "datasets" / "r2e_gym" / "train.jsonl").resolve()

SWEBENCH_RULE_REWARD_PATH = find_in_rllm("rllm", "rewards", "code_utils", "swebench.py")
SWEBENCH_RULE_REWARD_FN = "swebench_check_correctness"


_MODEL_PATH_FIELDS = {
    "model_path",
    "tokenizer_path",
    "critic_model_path",
    "critic_tokenizer_path",
    "cgm_model_path",
    "cgm_tokenizer_path",
}


def _resolve_optional_path(value: Path | str | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def _resolve_model_path(value: Path | str | None) -> str | None:
    """Resolve model-related paths supporting absolute or repo-relative values."""

    if value is None:
        return None

    raw_path = Path(value).expanduser()

    if raw_path.is_absolute():
        # ``Path.resolve`` canonicalises symlinks when the target exists but, on
        # some shared filesystems, strict resolution can unexpectedly fail if a
        # mounted path is only visible to remote workers.  We therefore only
        # collapse the path when it can be confirmed locally, otherwise we keep
        # the user's absolute string verbatim so it can still be interpreted on
        # Ray workers.
        return str(raw_path.resolve()) if raw_path.exists() else str(raw_path)

    for base in (Path.cwd(), REPO_ROOT):
        candidate = (base / raw_path).expanduser()
        if candidate.exists():
            return str(candidate.resolve())

    # If we cannot resolve the path relative to known roots we return the raw
    # string.  This allows callers to provide Hugging Face repository IDs or
    # other identifiers that remote workers know how to resolve.
    return str(value)


def _absolutise_args(args: argparse.Namespace) -> None:
    for field in [
        "dataset",
        "val_dataset",
        "docker_manifest",
        "model_path",
        "tokenizer_path",
        "critic_model_path",
        "critic_tokenizer_path",
        "cgm_model_path",
        "cgm_tokenizer_path",
        "output_dir",
        "resume",
        "config_file",
        "config",
    ]:
        if hasattr(args, field):
            resolver = _resolve_model_path if field in _MODEL_PATH_FIELDS else _resolve_optional_path
            resolved = resolver(getattr(args, field))
            if resolved is not None:
                setattr(args, field, resolved)


def _default_config_path() -> Path:
    """返回 rLLM 默认 PPO 配置文件路径。"""

    return find_in_rllm("trainer", "config", "agent_ppo_trainer.yaml")


def _collect_specified_cli_args(
    parser: argparse.ArgumentParser, argv: list[str] | None = None
) -> set[str]:
    tokens = list(argv if argv is not None else sys.argv[1:])
    specified: set[str] = set()
    for action in parser._actions:
        if action.dest in {"help", argparse.SUPPRESS}:
            continue
        if not action.option_strings:
            specified.add(action.dest)
            continue
        for opt in action.option_strings:
            if opt in tokens or any(token.startswith(f"{opt}=") for token in tokens):
                specified.add(action.dest)
                break
    return specified


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数，支持模型/数据集/调度配置。"""

    parser = argparse.ArgumentParser(description="Train Graph Planner agents with rLLM GRPO")
    parser.add_argument(
        "--agent",
        choices=["planner", "cgm", "both"],
        default="planner",
        help=(
            "Select which agent to train. Use 'planner' for the Graph Planner, "
            "'cgm' for the Code Graph Modifier, or 'both' to launch planner "
            "and CGM runs sequentially within a single invocation."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for Python/NumPy/Torch.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_TRAIN_DATASET_PATH,
        help="Training dataset in JSON/JSONL format.",
    )
    parser.add_argument("--dataset-name", default=None, help="Override dataset registry name.")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--val-dataset", type=Path, default=None)
    parser.add_argument("--val-split", default="val")
    parser.add_argument(
        "--docker-manifest",
        type=Path,
        default=None,
        help="Optional docker manifest path (defaults to <dataset dir>/docker_images.txt).",
    )
    parser.add_argument(
        "--prepull-containers",
        action="store_true",
        help="Pre-pull docker images referenced by the dataset before training.",
    )
    parser.add_argument("--prepull-max-workers", type=int, default=None)
    parser.add_argument("--prepull-retries", type=int, default=None)
    parser.add_argument("--prepull-delay", type=int, default=None)
    parser.add_argument("--prepull-timeout", type=int, default=None)
    parser.add_argument("--config-file", type=Path, default=None, help="High-level YAML configuration file.")
    parser.add_argument(
        "--yaml-only",
        action="store_true",
        help="Use YAML configuration exclusively (no CLI overrides besides path flags).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Target policy checkpoint path (defaults to models/Qwen3-14B or models/CodeFuse-CGM).",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for checkpoints and logs (defaults to model path).")
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument(
        "--critic-model-path",
        type=Path,
        default=None,
        help="Optional critic path; defaults to the policy checkpoint.",
    )
    parser.add_argument(
        "--critic-tokenizer-path",
        type=Path,
        default=None,
        help="Optional critic tokenizer path; defaults to the critic model path.",
    )
    parser.add_argument(
        "--cgm-model-path",
        type=Path,
        default=DEFAULT_CGM_MODEL_PATH,
        help="Optional CGM model used for planner patch synthesis.",
    )
    parser.add_argument("--cgm-tokenizer-path", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--stop", nargs="*", default=None, help="Optional list of stop strings for generation.")
    parser.add_argument("--stop-ids", nargs="*", type=int, default=None, help="Stop token ids for generation.")
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=None)
    parser.add_argument("--total-epochs", type=int, default=1)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None, help="Checkpoint save interval in steps.")
    parser.add_argument("--eval-interval", type=int, default=None, help="Evaluation interval in steps.")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from a previous run directory.")
    parser.add_argument("--early-stop-metric", default=None, help="Metric name used for early stopping.")
    parser.add_argument(
        "--early-stop-mode",
        choices=["min", "max"],
        default=None,
        help="Direction for early stopping metric.",
    )
    parser.add_argument("--early-stop-patience", type=int, default=None, help="Number of evals without improvement before stop.")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--rollout-replicas", type=int, default=1)
    parser.add_argument("--parallel-agents", type=int, default=None, help="Number of parallel agent-environment pairs.")
    parser.add_argument("--engine-max-workers", type=int, default=None, help="Thread pool size for async env operations.")
    parser.add_argument(
        "--workflow-parallel",
        type=int,
        default=None,
        help="Override rLLM workflow parallel task count.",
    )
    parser.add_argument(
        "--rollout-workers",
        type=int,
        default=None,
        help="Number of Verl rollout workers (defaults to parallel agent count).",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override.")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay override.")
    parser.add_argument("--warmup-steps", type=int, default=None, help="Scheduler warmup steps override.")
    parser.add_argument("--ray-address", default=None)
    parser.add_argument("--ray-num-cpus", type=int, default=None)
    parser.add_argument("--ray-num-gpus", type=int, default=None)
    parser.add_argument("--ray-memory", type=int, default=None)
    parser.add_argument("--ray-object-store-memory", type=int, default=None)
    parser.add_argument(
        "--config",
        type=Path,
        default=_default_config_path(),
        help="Base trainer config YAML.",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=None,
        help="Optional OmegaConf-style dotlist overrides (space separated).",
    )
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument(
        "--print-config-only",
        action="store_true",
        help="Print the resolved Hydra configuration and exit without launching training.",
    )
    parser.add_argument("--use-fallback", action="store_true", help="Enable rule fallback when training planner agent.")
    parser.add_argument(
        "--cgm-instruction",
        default=None,
        help="Instruction prompt passed to the CGM environment (cgm agent only).",
    )
    parser.add_argument("--reward-scale", type=float, default=1.0, help="Scale factor applied to environment rewards.")
    parser.add_argument("--failure-penalty", type=float, default=0.0, help="Penalty applied when episodes end without success.")
    parser.add_argument("--step-penalty", type=float, default=0.0, help="Penalty applied to each intermediate step.")
    parser.add_argument("--timeout-penalty", type=float, default=0.0, help="Penalty applied when max steps reached without success.")
    parser.add_argument("--repo-op-limit", type=int, default=None, help="Maximum repository operations before forcing termination.")
    parser.add_argument("--disable-cgm-synthesis", action="store_true", help="Disable CGM synthesis assistance inside the planner env.")
    parser.add_argument("--cgm-synthesis-strategy", default=None, help="Label for the CGM synthesis strategy exposed via env info.")
    parser.add_argument(
        "--strict-planner-io",
        action="store_true",
        help="Enable strict planner I/O re-validation and capping in env.",
    )
    parser.add_argument("--project-name", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--log-to-wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-offline", action="store_true", help="Force W&B offline mode when logging is enabled.")
    parser.add_argument(
        "--log-backend",
        choices=["tensorboard", "none"],
        default=None,
        help="Alternative logging backend when W&B is disabled.",
    )
    args, unknown = parser.parse_known_args(argv)
    if getattr(args, "print_config_only", False):
        args.print_config = True
    args._specified_cli_args = _collect_specified_cli_args(parser, argv)
    args.overrides = list(args.overrides or [])
    args._unknown_overrides = [token for token in unknown if token]
    return args


def _load_config(
    path: Path,
    overrides: list[str] | None = None,
    unknown: list[str] | None = None,
) -> OmegaConf:
    """从 YAML 文件加载 ``OmegaConf`` 配置并应用 dotlist 覆盖。"""

    resolved = path.resolve()
    cfg = OmegaConf.load(str(resolved))
    OmegaConf.set_struct(cfg, False)

    merged: list[str] = []
    for values in (overrides or [], unknown or []):
        for item in values:
            if not item:
                continue
            cleaned = item.lstrip("+")
            if cleaned.startswith("--"):
                cleaned = cleaned[2:]
            if cleaned:
                merged.append(cleaned)

    if merged:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(merged))
    print(f"[CFG] Applied {len(merged)} override(s)")
    return cfg


def _key_exists(cfg: OmegaConf, key: str) -> bool:
    """检测配置中是否已存在指定键。"""

    try:
        OmegaConf.select(cfg, key, throw_on_missing=True)
    except (ConfigKeyError, AttributeError, ValueError):
        return False
    return True


def _normalise_value(value: Any) -> Any:
    """将 ``Path`` 等类型转换成 YAML 友好的值。"""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        return [ _normalise_value(v) for v in value ]
    return value


def _set_if_exists(cfg: OmegaConf, key: str, value: Any) -> None:
    """仅在键已存在时更新配置值。"""

    if value is None:
        return
    if _key_exists(cfg, key):
        with open_dict(cfg):
            OmegaConf.update(cfg, key, _normalise_value(value), merge=False)


def _set(cfg: OmegaConf, key: str, value: Any) -> None:
    """在 OmegaConf 中写入（或创建）指定键。"""

    if value is None:
        return
    with open_dict(cfg):
        OmegaConf.update(cfg, key, _normalise_value(value), merge=True)


def _ensure_mapping(cfg: OmegaConf, key: str) -> None:
    """Ensure a nested mapping exists in the configuration."""

    if _key_exists(cfg, key):
        return
    _set(cfg, key, {})


def _ensure_default(cfg: OmegaConf, key: str, value: Any) -> None:
    """Set ``key`` to ``value`` when the configuration does not provide it."""

    try:
        current = OmegaConf.select(cfg, key, throw_on_missing=True)
    except (ConfigKeyError, AttributeError, ValueError):
        current = None

    if current is None:
        _set(cfg, key, value)


def _ensure_required_verl_flags(cfg: OmegaConf) -> None:
    """Backfill Verl config so runs default to GRPO with rule-based rewards."""

    required_defaults: dict[str, Any] = {
        "algorithm.use_kl_in_reward": False,
    }

    for dotted, default in required_defaults.items():
        _ensure_default(cfg, dotted, default)

    # Force GRPO toggles regardless of their values in the base PPO template.
    _set(cfg, "algorithm.adv_estimator", "grpo")
    _set(cfg, "actor_rollout_ref.actor.use_kl_loss", True)

    _ensure_mapping(cfg, "reward_model")
    _ensure_mapping(cfg, "reward_model.sandbox_fusion")
    _ensure_mapping(cfg, "reward_model.reward_kwargs")

    _ensure_default(cfg, "reward_model.enable", False)
    _ensure_default(cfg, "reward_model.reward_manager", "naive")
    _ensure_default(cfg, "reward_model.sandbox_fusion.url", None)
    _ensure_default(cfg, "reward_model.sandbox_fusion.max_concurrent", 64)
    _ensure_default(cfg, "reward_model.sandbox_fusion.memory_limit_mb", 1024)

    # Force a rule-based reward path so Verl never attempts to load an HF RM.
    _ensure_mapping(cfg, "custom_reward_function")
    _set(cfg, "custom_reward_function.path", str(SWEBENCH_RULE_REWARD_PATH))
    _set(cfg, "custom_reward_function.name", SWEBENCH_RULE_REWARD_FN)


def _iter_override_items(
    data: Mapping[str, Any], prefix: str = ""
) -> Iterable[tuple[str, Any]]:
    for key, value in data.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            yield from _iter_override_items(value, dotted)
        else:
            yield dotted, value


def _apply_verl_overrides(cfg: OmegaConf, overrides: Mapping[str, Any] | None) -> None:
    """将 YAML 中的 ``verl_overrides`` 展平并写入 Hydra 配置。"""

    if not overrides:
        return

    for dotted, value in _iter_override_items(overrides):
        if value is None:
            with open_dict(cfg):
                OmegaConf.update(cfg, dotted, None, merge=True)
        else:
            _set(cfg, dotted, value)


def _seed_everything(seed: int | None) -> None:
    """统一设置 Python/Numpy/PyTorch 随机种子。"""

    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - GPU specific branch
            torch.cuda.manual_seed_all(seed)
        try:
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
        except AttributeError:
            pass


def _resolve_dataset(
    *,
    dataset_path: Path,
    dataset_name: str,
    split: str,
    val_dataset: Path | None,
    val_split: str,
) -> Tuple[str, str | None, int, int | None]:
    """注册训练/验证集并返回 rLLM 生成的 verl parquet 路径及样本数。"""

    train_rows = load_task_entries(dataset_path)
    if not train_rows:
        raise RuntimeError(f"Dataset {dataset_path} did not contain any rows")
    train_ds = ensure_dataset_registered(name=dataset_name, split=split, path=dataset_path)
    train_path = train_ds.get_verl_data_path()
    if not train_path:
        raise RuntimeError("Training dataset registration did not produce a Verl parquet file")

    if val_dataset is None:
        return train_path, None, len(train_rows), None

    val_rows = load_task_entries(val_dataset)
    if not val_rows:
        raise RuntimeError(f"Validation dataset {val_dataset} did not contain any rows")
    val_ds = ensure_dataset_registered(name=f"{dataset_name}_val", split=val_split, path=val_dataset)
    val_path = val_ds.get_verl_data_path()
    if not val_path:
        raise RuntimeError("Validation dataset registration did not produce a Verl parquet file")
    return train_path, val_path, len(train_rows), len(val_rows)


def _apply_training_hyperparameters(
    cfg: OmegaConf, training_cfg: Mapping[str, Any] | None
) -> None:
    """Map high-level training config into Verl/rLLM optimizer knobs."""

    if not training_cfg:
        return

    def _set_cast(path: str, value: Any, cast: type | None = None) -> None:
        if value is None:
            return
        _set(cfg, path, cast(value) if cast else value)

    _set_cast("trainer.gradient_accumulation_steps", training_cfg.get("grad_accum_steps"), int)

    lr = training_cfg.get("lr")
    _set_cast("actor_rollout_ref.actor.optim.lr", lr, float)
    _set_cast("critic.optim.lr", lr, float)

    weight_decay = training_cfg.get("weight_decay")
    _set_cast("actor_rollout_ref.actor.optim.weight_decay", weight_decay, float)
    _set_cast("critic.optim.weight_decay", weight_decay, float)

    warmup_steps = training_cfg.get("warmup_steps")
    _set_cast("actor_rollout_ref.actor.optim.lr_warmup_steps", warmup_steps, int)
    _set_cast("critic.optim.lr_warmup_steps", warmup_steps, int)

    total_steps = training_cfg.get("total_steps")
    _set_cast("actor_rollout_ref.actor.optim.total_training_steps", total_steps, int)
    _set_cast("critic.optim.total_training_steps", total_steps, int)

    clip_norm = training_cfg.get("clip_grad_norm")
    _set_cast("actor_rollout_ref.actor.grad_clip", clip_norm, float)
    _set_cast("critic.grad_clip", clip_norm, float)

    grad_ckpt = training_cfg.get("gradient_checkpointing")
    if grad_ckpt is not None:
        flag = bool(grad_ckpt)
        _set(cfg, "actor_rollout_ref.model.enable_gradient_checkpointing", flag)
        _set(cfg, "critic.model.enable_gradient_checkpointing", flag)

    entropy = training_cfg.get("entropy_coef")
    _set_cast("actor_rollout_ref.actor.entropy_coeff", entropy, float)

    clip_coef = training_cfg.get("clip_coef")
    _set_cast("actor_rollout_ref.actor.clip_ratio", clip_coef, float)
    _set_cast("actor_rollout_ref.actor.clip_ratio_low", clip_coef, float)
    _set_cast("actor_rollout_ref.actor.clip_ratio_high", clip_coef, float)

    kl_coef = training_cfg.get("kl_coef")
    _set_cast("algorithm.kl_ctrl.kl_coef", kl_coef, float)
    _set_cast("actor_rollout_ref.actor.kl_loss_coef", kl_coef, float)
    _set_cast("actor_rollout_ref.actor.policy_loss.ppo_kl_coef", kl_coef, float)

    target_kl = training_cfg.get("target_kl")
    _set_cast("algorithm.kl_ctrl.target_kl", target_kl, float)


def _apply_model_overrides(cfg: OmegaConf, args: argparse.Namespace) -> None:
    """根据命令行参数注入模型/采样相关配置。"""

    model_path = str(args.model_path)
    tokenizer_path = str(args.tokenizer_path or args.model_path)
    critic_path = str(args.critic_model_path or args.model_path)
    critic_tok = str(args.critic_tokenizer_path or args.critic_model_path or args.model_path)

    _set_if_exists(cfg, "actor_rollout_ref.model.path", model_path)
    _set_if_exists(cfg, "actor_rollout_ref.model.tokenizer_path", tokenizer_path)
    _set_if_exists(cfg, "critic.model.path", critic_path)
    _set_if_exists(cfg, "critic.model.tokenizer_path", critic_tok)

    if args.temperature is not None:
        _set_if_exists(cfg, "actor_rollout_ref.rollout.temperature", float(args.temperature))
    if args.top_p is not None:
        _set_if_exists(cfg, "actor_rollout_ref.rollout.top_p", float(args.top_p))

    _set(cfg, "actor_rollout_ref.rollout.tensor_model_parallel_size", int(args.tensor_parallel))
    _set(cfg, "actor_rollout_ref.rollout.n", int(args.rollout_replicas))

    if args.max_input_tokens is not None:
        _set(cfg, "actor_rollout_ref.rollout.max_prompt_length", int(args.max_input_tokens))
        _set_if_exists(cfg, "data.max_prompt_length", int(args.max_input_tokens))
    if args.max_output_tokens is not None:
        _set(cfg, "actor_rollout_ref.rollout.max_response_length", int(args.max_output_tokens))
        _set_if_exists(cfg, "data.max_response_length", int(args.max_output_tokens))
    if args.stop:
        _set(cfg, "actor_rollout_ref.rollout.stop", list(args.stop))
    if args.stop_ids:
        _set(cfg, "actor_rollout_ref.rollout.stop_token_ids", [int(s) for s in args.stop_ids])


def _apply_parallel_overrides(cfg: OmegaConf, args: argparse.Namespace) -> None:
    """写入并行度与 Ray 资源相关的配置。"""

    _set(cfg, "trainer.nnodes", int(args.num_nodes))

    rollout_workers = args.rollout_workers if args.rollout_workers is not None else args.parallel_agents
    if rollout_workers:
        _set(cfg, "actor_rollout_ref.rollout.agent.num_workers", int(rollout_workers))

    if args.parallel_agents:
        _set(cfg, "rllm.agent.engine_args.n_parallel_agents", int(args.parallel_agents))
        if args.engine_max_workers is None:
            # Provide a generous default when not explicitly set.
            suggested_workers = max(64, int(args.parallel_agents) * 2)
            _set(cfg, "rllm.agent.engine_args.max_workers", suggested_workers)
    if args.engine_max_workers is not None:
        _set(cfg, "rllm.agent.engine_args.max_workers", int(args.engine_max_workers))

    if args.workflow_parallel is not None:
        _set(cfg, "rllm.workflow.n_parallel_tasks", int(args.workflow_parallel))

    if args.ray_num_cpus is not None:
        _set(cfg, "ray_init.num_cpus", int(args.ray_num_cpus))
    if args.ray_num_gpus is not None:
        _set(cfg, "ray_init.num_gpus", int(args.ray_num_gpus))
    else:
        _set(cfg, "ray_init.num_gpus", int(args.num_gpus))
    if args.ray_memory is not None:
        _set(cfg, "ray_init.memory", int(args.ray_memory))
    if args.ray_object_store_memory is not None:
        _set(cfg, "ray_init.object_store_memory", int(args.ray_object_store_memory))
    if args.ray_address:
        _set(cfg, "ray_init.address", args.ray_address)


def _validate_parallel_config(args: argparse.Namespace) -> None:
    """在启动训练前校验 GPU/并行度配置是否合理。"""

    total_gpus = max(1, int(args.num_gpus)) * max(1, int(args.num_nodes))
    issues = []
    if args.num_gpus <= 0:
        issues.append("--num-gpus must be positive")
    if args.tensor_parallel > args.num_gpus:
        issues.append(
            "tensor parallelism requires at least as many GPUs per node as --tensor-parallel; "
            "increase --num-gpus or reduce --tensor-parallel"
        )
    if args.tensor_parallel > total_gpus:
        issues.append(
            "total available GPUs across nodes is smaller than tensor parallel size; "
            "increase --num-gpus/--num-nodes or lower --tensor-parallel"
        )
    if args.ray_num_gpus is not None and args.ray_num_gpus < total_gpus:
        issues.append(
            "Ray GPU budget is smaller than requested trainer GPUs; "
            "set --ray-num-gpus >= --num-gpus * --num-nodes"
        )
    if args.parallel_agents and args.rollout_workers and args.rollout_workers < args.parallel_agents:
        issues.append(
            "--rollout-workers is smaller than --parallel-agents; "
            "increase rollout workers or reduce parallel agents"
        )
    if issues:
        formatted = "\n - ".join([""] + issues)
        raise ValueError(f"Parallel resource configuration invalid:{formatted}")


def _configure_agent_env(cfg: OmegaConf, args: argparse.Namespace) -> Tuple[type, Dict[str, Any], type, Dict[str, Any]]:
    """返回需注册的 Agent/Env 类及其构造参数，并同步写入配置。"""

    if args.agent == "planner":
        agent_cls = GraphPlannerRLLMAgent
        env_cls = GraphPlannerRLLMEnv
        agent_args: Dict[str, Any] = {"use_rule_fallback": bool(args.use_fallback)}
        env_args: Dict[str, Any] = {
            "max_steps": int(args.max_steps),
            "reward_scale": float(args.reward_scale),
            "failure_penalty": float(args.failure_penalty),
            "step_penalty": float(args.step_penalty),
            "timeout_penalty": float(args.timeout_penalty),
            "repo_operation_limit": int(args.repo_op_limit) if args.repo_op_limit else None,
            "enable_cgm_synthesis": not bool(args.disable_cgm_synthesis),
            "synthesis_strategy": args.cgm_synthesis_strategy,
        }

        _set_if_exists(cfg, "rllm.agent.name", "graph_planner_repoenv")
        _set_if_exists(cfg, "rllm.env.name", "graph_planner_repoenv")
        _set_if_exists(cfg, "agent.name", "graph_planner_repoenv")
        _set_if_exists(cfg, "env.name", "graph_planner_repoenv")

        if args.cgm_model_path:
            os.environ.setdefault("CGM_ENABLED", "1")
            os.environ["CGM_MODEL_PATH"] = str(args.cgm_model_path)
            if args.cgm_tokenizer_path:
                os.environ["CGM_TOKENIZER_PATH"] = str(args.cgm_tokenizer_path)
    else:
        agent_cls = CGMRLLMAgent
        env_cls = CGMRLLMEnv
        agent_args = {}
        env_args = {
            "max_steps": int(max(1, args.max_steps)),
            "instruction": args.cgm_instruction,
            "reward_scale": float(args.reward_scale),
            "failure_penalty": float(args.failure_penalty),
            "step_penalty": float(args.step_penalty),
            "timeout_penalty": float(args.timeout_penalty),
            "repo_operation_limit": int(args.repo_op_limit) if args.repo_op_limit else None,
            "synthesis_strategy": args.cgm_synthesis_strategy,
        }
        _set_if_exists(cfg, "rllm.agent.name", "graph_planner_cgm")
        _set_if_exists(cfg, "rllm.env.name", "graph_planner_cgm")
        _set_if_exists(cfg, "agent.name", "graph_planner_cgm")
        _set_if_exists(cfg, "env.name", "graph_planner_cgm")

    _set_if_exists(cfg, "agent.max_steps", int(args.max_steps))
    _set_if_exists(cfg, "env.env_args.max_steps", int(args.max_steps))
    return agent_cls, agent_args, env_cls, env_args


def _apply_training_overrides(cfg: OmegaConf, args: argparse.Namespace) -> Path:
    """应用训练相关参数覆写并返回输出目录。"""

    output_dir = Path(args.output_dir or args.model_path).expanduser().resolve()
    _set(cfg, "trainer.output_dir", str(output_dir))
    _set(cfg, "trainer.default_local_dir", str(output_dir))

    if args.save_interval is not None:
        _set(cfg, "trainer.save_interval", int(args.save_interval))
        _set(cfg, "trainer.save_freq", int(args.save_interval))
    if args.eval_interval is not None:
        _set(cfg, "trainer.test_freq", int(args.eval_interval))
    if args.resume is not None:
        resume_path = Path(args.resume).expanduser().resolve()
        _set(cfg, "trainer.resume_from", str(resume_path))
        _set(cfg, "trainer.resume_from_path", str(resume_path))
        _set(cfg, "trainer.resume_mode", "manual")

    if args.precision is not None:
        _set(cfg, "trainer.mixed_precision", args.precision)
    if args.grad_accum_steps is not None:
        _set(cfg, "trainer.gradient_accumulation_steps", int(args.grad_accum_steps))

    if args.lr is not None:
        _set(cfg, "trainer.optimizer.lr", float(args.lr))
    if args.weight_decay is not None:
        _set(cfg, "trainer.optimizer.weight_decay", float(args.weight_decay))
    if args.warmup_steps is not None:
        _set(cfg, "trainer.scheduler.warmup_steps", int(args.warmup_steps))

    if args.early_stop_metric:
        _set(cfg, "trainer.early_stop.metric", args.early_stop_metric)
        _set(cfg, "trainer.early_stop.mode", args.early_stop_mode or "max")
        if args.early_stop_patience is not None:
            _set(cfg, "trainer.early_stop.patience", int(args.early_stop_patience))
    elif args.early_stop_mode or args.early_stop_patience:
        warnings.warn("--early-stop-* flags require --val-dataset; ignoring early stop settings.", stacklevel=2)

    return output_dir


def _apply_logging_overrides(cfg: OmegaConf, args: argparse.Namespace) -> None:
    """根据日志相关参数调整配置并设置环境变量。"""

    backends = ["console"]
    if args.log_to_wandb:
        backends.append("wandb")
        os.environ.setdefault("WANDB_PROJECT", args.project_name or "graph_planner")
        os.environ.setdefault("WANDB_MODE", "offline" if args.wandb_offline else "online")
    else:
        if args.log_backend == "tensorboard" or args.log_backend is None:
            backends.append("tensorboard")
        if args.log_backend == "none":
            backends = ["console"]
    _set(cfg, "trainer.logger", backends)


def _print_run_summary(
    *,
    args: argparse.Namespace,
    train_path: str,
    val_path: str | None,
    train_rows: int,
    val_rows: int | None,
    output_dir: Path,
    header: str = "Graph Planner rLLM training launch summary:",
) -> None:
    """打印关键运行配置，方便审阅日志。"""

    lines = [
        header,
        f"  agent: {args.agent}",
        f"  train samples: {train_rows}",
        f"  val samples: {val_rows if val_rows is not None else 0}",
        f"  train file: {train_path}",
        f"  val file: {val_path or 'N/A'}",
        f"  max steps: {args.max_steps}",
        f"  reward scale: {args.reward_scale}",
        f"  step penalty: {args.step_penalty}",
        f"  failure penalty: {args.failure_penalty}",
        f"  timeout penalty: {args.timeout_penalty}",
        f"  parallel agents: {getattr(args, 'parallel_agents', None) or 'auto'}",
        f"  rollout workers: {getattr(args, 'rollout_workers', None) or getattr(args, 'parallel_agents', None) or 'auto'}",
        f"  tensor parallel: {getattr(args, 'tensor_parallel', 'n/a')}",
        f"  total epochs: {getattr(args, 'total_epochs', 'n/a')}",
        f"  total steps: {getattr(args, 'total_steps', None) or 'auto'}",
        f"  save interval: {getattr(args, 'save_interval', None) or 'default'}",
        f"  eval interval: {getattr(args, 'eval_interval', None) or 'default'}",
        f"  output dir: {output_dir}",
        f"  ray address: {args.ray_address or 'auto'}",
    ]
    print("\n".join(lines))


def _sanity_checks(train_path: str, val_path: str | None, args: argparse.Namespace) -> None:
    """执行训练前的安全检查。"""

    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if not Path(train_path).exists():
        raise FileNotFoundError(f"Training parquet not found: {train_path}")
    if val_path and not Path(val_path).exists():
        raise FileNotFoundError(f"Validation parquet not found: {val_path}")
    if args.precision in {"bf16", "fp16"} and torch is None:
        raise RuntimeError("Requested mixed precision but PyTorch is unavailable")


def _prepare_container_images(args: argparse.Namespace, final_run_cfg: Dict[str, Any]) -> List[str]:
    """Ensure docker manifest exists and optionally pre-pull containers."""

    env_cfg = final_run_cfg.setdefault("env", {})
    manifest_path = getattr(args, "docker_manifest", None) or env_cfg.get("docker_manifest")
    if manifest_path:
        manifest = Path(manifest_path).expanduser().resolve()
    else:
        manifest = (args.dataset.parent / "docker_images.txt").resolve()
    env_cfg["docker_manifest"] = str(manifest)

    if manifest.exists():
        images = load_docker_manifest(manifest)
        LOGGER.info("Loaded %d docker images from manifest %s", len(images), manifest)
    else:
        records = load_task_entries(args.dataset)
        if args.val_dataset:
            records.extend(load_task_entries(args.val_dataset))
        collection = collect_docker_images(records=records)
        write_docker_manifest(manifest, collection.images)
        LOGGER.info(
            "Docker manifest written to %s (%d images, %d missing metadata)",
            manifest,
            len(collection.images),
            collection.missing,
        )
        images = collection.images

    if args.prepull_containers and images:
        LOGGER.info("Pre-pulling %d docker images prior to training", len(images))
        prepull_docker_images(
            images,
            max_workers=args.prepull_max_workers,
            retries=args.prepull_retries,
            delay=args.prepull_delay,
            pull_timeout=args.prepull_timeout,
        )
    elif args.prepull_containers:
        LOGGER.warning("Pre-pull requested but no docker images discovered for %s", args.dataset)

    return images


def _requested_agents(agent_flag: str) -> List[str]:
    if agent_flag == "both":
        return ["planner", "cgm"]
    return [agent_flag]


def _prepare_agent_args(base_args: argparse.Namespace, agent: str) -> argparse.Namespace:
    run_args = copy.deepcopy(base_args)
    run_args.agent = agent
    if agent == "cgm":
        run_args.disable_cgm_synthesis = True
    return run_args


def _run_training(args: argparse.Namespace, *, run_index: int, total_runs: int) -> None:
    _absolutise_args(args)

    defaults = default_training_run_config(args.agent)
    yaml_cfg = load_run_config_file(getattr(args, "config_file", None))
    cli_overrides = build_cli_overrides(args, mode="train")
    if args.yaml_only and cli_overrides:
        LOGGER.info("--yaml-only enabled; ignoring CLI overrides: %s", sorted(cli_overrides.keys()))
        cli_overrides = {}

    final_run_cfg = merge_run_config(
        defaults,
        yaml_cfg,
        cli_overrides,
        yaml_only=args.yaml_only,
        agent=args.agent,
    )
    logging_cfg = final_run_cfg.setdefault("logging", {})
    logging_cfg["agent_kind"] = args.agent
    logging_cfg["multi_agent_index"] = run_index
    logging_cfg["multi_agent_total"] = total_runs

    wandb_cfg = logging_cfg.setdefault("wandb", {})
    if not wandb_cfg.get("run_name"):
        wandb_cfg["run_name"] = f"{args.agent}-run"
        run_name_user_supplied = False
    else:
        run_name_user_supplied = True

    if total_runs > 1 and run_name_user_supplied:
        current = wandb_cfg.get("run_name") or ""
        suffix = args.agent
        if suffix and not current.endswith(f"-{suffix}"):
            wandb_cfg["run_name"] = f"{current}-{suffix}"

    run_name = wandb_cfg.get("run_name")

    update_args_from_config(args, final_run_cfg, respect_cli=not args.yaml_only)
    _absolutise_args(args)

    final_run_cfg.setdefault("io", {})["strict_planner_io"] = bool(args.strict_planner_io)
    if args.strict_planner_io:
        os.environ["GRAPH_PLANNER_STRICT_IO"] = "1"
    else:
        os.environ.pop("GRAPH_PLANNER_STRICT_IO", None)

    if args.model_path is None:
        default_model = DEFAULT_PLANNER_MODEL_PATH if args.agent == "planner" else DEFAULT_CGM_MODEL_PATH
        args.model_path = default_model
        key = "planner_model" if args.agent == "planner" else "cgm_model"
        final_run_cfg.setdefault("paths", {})[key] = str(default_model)
    else:
        args.model_path = _resolve_model_path(args.model_path)

    output_base = Path(
        logging_cfg.get("output_dir")
        or getattr(args, "output_dir", None)
        or args.model_path
    )
    output_base = output_base.expanduser().resolve()
    if not run_name:
        run_name = f"{args.agent}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        wandb_cfg["run_name"] = run_name
    run_dir = (output_base / run_name).resolve()
    logging_cfg["resolved_run_dir"] = str(run_dir)
    resolved_cfg_path = run_dir / "resolved_config.yaml"
    logging_cfg["resolved_config_path"] = str(resolved_cfg_path)
    serialise_resolved_config(final_run_cfg, resolved_cfg_path)
    args.output_dir = run_dir

    pcfg = resolve_parallel(final_run_cfg)
    preflight_check(pcfg)

    logging.basicConfig(level=logging.INFO)
    _seed_everything(getattr(args, "seed", None))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    _validate_parallel_config(args)

    dataset_name = args.dataset_name
    if not dataset_name:
        dataset_name = GRAPH_PLANNER_CGM_DATASET_NAME if args.agent == "cgm" else GRAPH_PLANNER_DATASET_NAME

    resolved_train_ds = resolve_task_file(args.dataset, split=args.dataset_split)
    args.dataset = resolved_train_ds
    final_run_cfg.setdefault("paths", {})["dataset_train"] = str(resolved_train_ds)

    resolved_val_ds: Path | None = None
    if args.val_dataset is not None:
        resolved_val_ds = resolve_task_file(args.val_dataset, split=args.val_split)
        args.val_dataset = resolved_val_ds
        final_run_cfg.setdefault("paths", {})["dataset_val"] = str(resolved_val_ds)

    train_path, val_path, train_rows, val_rows = _resolve_dataset(
        dataset_path=resolved_train_ds,
        dataset_name=dataset_name,
        split=args.dataset_split,
        val_dataset=resolved_val_ds,
        val_split=args.val_split,
    )

    container_images = _prepare_container_images(args, final_run_cfg)

    cfg = _load_config(
        args.config,
        overrides=args.overrides,
        unknown=getattr(args, "_unknown_overrides", None),
    )

    _ensure_required_verl_flags(cfg)

    _apply_training_hyperparameters(cfg, final_run_cfg.get("training"))

    _set(cfg, "data.train_files", str(train_path))
    if val_path is None:
        warnings.warn(
            "Validation dataset not provided; disabling evaluation/early stop triggers.",
            stacklevel=2,
        )
        effective_val_path = train_path
        val_rows = val_rows or 0
    else:
        effective_val_path = val_path
    _set(cfg, "data.val_files", str(effective_val_path))
    _set(cfg, "data.train_batch_size", int(args.train_batch_size))
    if args.val_batch_size is not None:
        _set(cfg, "data.val_batch_size", int(args.val_batch_size))
    _set(cfg, "data.shuffle", False)
    _set(cfg, "io.strict_planner_io", bool(args.strict_planner_io))

    if args.project_name:
        _set(cfg, "trainer.project_name", args.project_name)
    if args.experiment_name:
        _set(cfg, "trainer.experiment_name", args.experiment_name)

    _set(cfg, "trainer.total_epochs", int(args.total_epochs))
    if args.total_steps:
        _set(cfg, "trainer.total_training_steps", int(args.total_steps))
    _set(cfg, "trainer.n_gpus_per_node", int(args.num_gpus))

    _apply_model_overrides(cfg, args)
    _apply_parallel_overrides(cfg, args)
    output_dir = _apply_training_overrides(cfg, args)
    _apply_logging_overrides(cfg, args)
    _apply_verl_overrides(cfg, final_run_cfg.get("verl_overrides"))
    agent_cls, agent_args, env_cls, env_args = _configure_agent_env(cfg, args)

    if val_path is None:
        _set(cfg, "trainer.test_freq", -1)

    _print_run_summary(
        args=args,
        train_path=train_path,
        val_path=val_path,
        train_rows=train_rows,
        val_rows=val_rows,
        output_dir=output_dir,
    )

    if container_images:
        LOGGER.info("Container manifest includes %d images", len(container_images))

    wandb_run = init_wandb(
        enabled=bool(wandb_cfg.get("enabled", False)),
        offline=bool(wandb_cfg.get("offline", False)),
        project=wandb_cfg.get("project") or "graph_planner",
        entity=wandb_cfg.get("entity"),
        run_name=wandb_cfg.get("run_name", run_name),
        config=final_run_cfg,
    )

    try:
        parallel_metrics = {
            "parallel/tensor_parallel_planner": pcfg.tensor_parallel_planner,
            "parallel/tensor_parallel_cgm": pcfg.tensor_parallel_cgm,
            "parallel/replicas": pcfg.replicas,
            "parallel/agents": pcfg.parallel_agents,
            "parallel/workers": pcfg.rollout_workers,
            "parallel/workflow_parallel": pcfg.workflow_parallel,
            "resources/num_gpus": pcfg.num_gpus,
            "resources/ray_gpus": pcfg.ray_num_gpus,
            "resources/ray_cpus": pcfg.ray_num_cpus,
        }
        gpu_snapshot = make_gpu_snapshot()
        ray_snapshot = make_ray_snapshot()
        log_metrics(0, {**parallel_metrics, **gpu_snapshot(), **ray_snapshot()})

        if args.print_config:
            print(OmegaConf.to_yaml(cfg))
            if getattr(args, "print_config_only", False):
                return

        output_dir.mkdir(parents=True, exist_ok=True)

        _sanity_checks(train_path=train_path, val_path=val_path, args=args)

        global AgentTrainer
        trainer_cls = AgentTrainer
        if trainer_cls is None:
            from rllm.trainer.agent_trainer import AgentTrainer as _AgentTrainer  # noqa: E402

            trainer_cls = _AgentTrainer
            AgentTrainer = trainer_cls

        trainer = trainer_cls(
            agent_class=agent_cls,
            env_class=env_cls,
            agent_args=agent_args,
            env_args=env_args,
            config=cfg,
        )

        if args.ray_address:
            os.environ.setdefault("RAY_ADDRESS", args.ray_address)

        trainer.train()
    finally:
        if wandb_run is not None:  # pragma: no cover - network side effect
            try:
                wandb_run.finish()
            except Exception:
                pass


def main() -> None:
    """脚本入口：解析参数、准备数据集并触发 rLLM 训练。"""

    parsed = _parse_args()
    agents = _requested_agents(parsed.agent)

    for index, agent in enumerate(agents):
        run_args = _prepare_agent_args(parsed, agent)
        LOGGER.info(
            "Launching %s training run (%d/%d)",
            agent,
            index + 1,
            len(agents),
        )
        _run_training(run_args, run_index=index, total_runs=len(agents))


if __name__ == "__main__":
    main()
