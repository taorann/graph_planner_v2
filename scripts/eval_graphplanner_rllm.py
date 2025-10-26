"""评估 Graph Planner/CGM 代理在 rLLM 上的推理表现（仅验证，不更新参数）。

English summary
    Runs the rLLM GRPO pipeline in validation-only mode so we can collect
    pass@k, success rate, and trajectory statistics without performing any
    optimisation steps.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from omegaconf import OmegaConf

from graph_planner.infra.config import (
    build_cli_overrides,
    default_training_run_config,
    load_run_config_file,
    merge_run_config,
    serialise_resolved_config,
    update_args_from_config,
)
from graph_planner.infra.metrics import init_wandb, log_metrics, make_gpu_snapshot, make_ray_snapshot
from graph_planner.infra.parallel import preflight_check, resolve_parallel
from graph_planner.infra.vendor import ensure_rllm_importable

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
from scripts.train_graphplanner_rllm import (  # noqa: E402
    DEFAULT_CGM_MODEL_PATH,
    DEFAULT_PLANNER_MODEL_PATH,
    _absolutise_args,
    _apply_logging_overrides,
    _apply_model_overrides,
    _apply_parallel_overrides,
    _apply_training_hyperparameters,
    _apply_verl_overrides,
    _coerce_int,
    _configure_agent_env,
    _ensure_batch_size_defaults,
    _ensure_required_verl_flags,
    _load_config,
    _print_run_summary,
    _sanity_checks,
    _seed_everything,
    _set,
    _resolve_effective_batch_size,
    _prepare_container_images,
    _validate_parallel_config,
    _resolve_model_path,
)


LOGGER = logging.getLogger(__name__)


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
    parser = argparse.ArgumentParser(
        description="Evaluate Graph Planner agents with rLLM GRPO (validation only)"
    )
    parser.add_argument("--agent", choices=["planner", "cgm"], default="planner")
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Evaluation dataset in JSON/JSONL format.",
    )
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-split", default="eval")
    parser.add_argument(
        "--docker-manifest",
        type=Path,
        default=None,
        help="Optional docker manifest path (defaults to <dataset dir>/docker_images.txt).",
    )
    parser.add_argument(
        "--prepull-containers",
        action="store_true",
        help="Pre-pull docker images before evaluation.",
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
        help="Policy checkpoint to evaluate (defaults to models/Qwen3-14B or models/CodeFuse-CGM).",
    )
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--critic-model-path", type=Path, default=None)
    parser.add_argument("--critic-tokenizer-path", type=Path, default=None)
    parser.add_argument("--cgm-model-path", type=Path, default=DEFAULT_CGM_MODEL_PATH)
    parser.add_argument("--cgm-tokenizer-path", type=Path, default=None)
    parser.add_argument("--cgm-instruction", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--stop", nargs="*", default=None)
    parser.add_argument("--stop-ids", nargs="*", type=int, default=None)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--parallel-agents", type=int, default=None)
    parser.add_argument("--rollout-workers", type=int, default=None)
    parser.add_argument("--rollout-replicas", type=int, default=1)
    parser.add_argument("--engine-max-workers", type=int, default=None)
    parser.add_argument("--workflow-parallel", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default=None)
    parser.add_argument("--ray-address", default=None)
    parser.add_argument("--ray-num-cpus", type=int, default=None)
    parser.add_argument("--ray-num-gpus", type=int, default=None)
    parser.add_argument("--ray-memory", type=int, default=None)
    parser.add_argument("--ray-object-store-memory", type=int, default=None)
    parser.add_argument("--config", type=Path, required=True, help="Base trainer config YAML.")
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=None,
        help="Optional OmegaConf-style dotlist overrides (space separated).",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument(
        "--print-config-only",
        action="store_true",
        help="Print the resolved Hydra configuration and exit without launching evaluation.",
    )
    parser.add_argument("--use-fallback", action="store_true")
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--failure-penalty", type=float, default=0.0)
    parser.add_argument("--step-penalty", type=float, default=0.0)
    parser.add_argument("--timeout-penalty", type=float, default=0.0)
    parser.add_argument("--repo-op-limit", type=int, default=None)
    parser.add_argument("--disable-cgm-synthesis", action="store_true")
    parser.add_argument("--cgm-synthesis-strategy", default=None)
    parser.add_argument("--project-name", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--log-to-wandb", action="store_true")
    parser.add_argument("--wandb-offline", action="store_true")
    parser.add_argument("--log-backend", choices=["tensorboard", "none"], default=None)
    args, unknown = parser.parse_known_args(argv)
    if getattr(args, "print_config_only", False):
        args.print_config = True
    args._specified_cli_args = _collect_specified_cli_args(parser, argv)
    args.overrides = list(args.overrides or [])
    args._unknown_overrides = [token for token in unknown if token]
    return args


def _resolve_eval_dataset(path: Path, *, name: str, split: str) -> tuple[Path, str, int]:
    resolved = resolve_task_file(path, split=split)
    rows = load_task_entries(resolved)
    if not rows:
        raise RuntimeError(f"Dataset {resolved} did not contain any rows")
    dataset = ensure_dataset_registered(name=name, split=split, path=resolved)
    verl_path = dataset.get_verl_data_path()
    if not verl_path:
        raise RuntimeError("Dataset registration did not produce a Verl parquet file")
    return resolved, verl_path, len(rows)


def main() -> None:
    args = _parse_args()
    _absolutise_args(args)

    defaults = default_training_run_config(args.agent)
    yaml_cfg = load_run_config_file(getattr(args, "config_file", None))
    cli_overrides = build_cli_overrides(args, mode="eval")
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
    wandb_cfg = final_run_cfg.setdefault("logging", {}).setdefault("wandb", {})
    if not wandb_cfg.get("run_name"):
        wandb_cfg["run_name"] = f"{args.agent}-eval"
    run_name = wandb_cfg["run_name"]

    update_args_from_config(args, final_run_cfg, respect_cli=not args.yaml_only)
    _absolutise_args(args)

    if args.model_path is None:
        default_model = DEFAULT_PLANNER_MODEL_PATH if args.agent == "planner" else DEFAULT_CGM_MODEL_PATH
        args.model_path = default_model
        key = "planner_model" if args.agent == "planner" else "cgm_model"
        final_run_cfg.setdefault("paths", {})[key] = str(default_model)
    else:
        args.model_path = _resolve_model_path(args.model_path)

    output_base = Path(
        final_run_cfg.get("logging", {}).get("output_dir")
        or getattr(args, "output_dir", None)
        or args.model_path
    )
    output_base = output_base.expanduser().resolve()
    if not run_name:
        run_name = f"{args.agent}-eval-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        wandb_cfg["run_name"] = run_name
    run_dir = (output_base / run_name).resolve()
    final_run_cfg.setdefault("logging", {})["resolved_run_dir"] = str(run_dir)
    resolved_cfg_path = run_dir / "resolved_config.yaml"
    final_run_cfg["logging"]["resolved_config_path"] = str(resolved_cfg_path)
    serialise_resolved_config(final_run_cfg, resolved_cfg_path)
    args.output_dir = run_dir

    pcfg = resolve_parallel(final_run_cfg)
    preflight_check(pcfg)

    logging.basicConfig(level=logging.INFO)
    _seed_everything(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    _validate_parallel_config(args)

    dataset_name = args.dataset_name
    if not dataset_name:
        dataset_name = GRAPH_PLANNER_CGM_DATASET_NAME if args.agent == "cgm" else GRAPH_PLANNER_DATASET_NAME
    eval_dataset_name = f"{dataset_name}_eval"
    dataset_jsonl, eval_path, sample_count = _resolve_eval_dataset(
        args.dataset,
        name=eval_dataset_name,
        split=args.dataset_split,
    )
    args.dataset = dataset_jsonl
    final_run_cfg.setdefault("paths", {})["dataset_train"] = str(dataset_jsonl)

    container_images = _prepare_container_images(args, final_run_cfg)

    cfg = _load_config(
        args.config,
        overrides=args.overrides,
        unknown=getattr(args, "_unknown_overrides", None),
    )

    _ensure_required_verl_flags(cfg)

    _apply_training_hyperparameters(cfg, final_run_cfg.get("training"))

    _set(cfg, "data.train_files", str(eval_path))
    _set(cfg, "data.val_files", str(eval_path))

    requested_batch = _coerce_int(getattr(args, "batch_size", None))
    effective_batch, capped_batch = _resolve_effective_batch_size(requested_batch, sample_count)
    if capped_batch:
        LOGGER.warning(
            "Requested evaluation batch size %s exceeds dataset size (%s); capping to %s to avoid empty dataloader.",
            requested_batch,
            sample_count,
            effective_batch,
        )

    args.batch_size = effective_batch
    final_run_cfg.setdefault("training", {})["train_batch_size"] = effective_batch
    _set(cfg, "data.train_batch_size", effective_batch)
    _set(cfg, "data.val_batch_size", effective_batch)
    _set(cfg, "data.shuffle", False)
    _ensure_batch_size_defaults(cfg)

    if args.project_name:
        _set(cfg, "trainer.project_name", args.project_name)
    if args.experiment_name:
        _set(cfg, "trainer.experiment_name", args.experiment_name)

    _set(cfg, "trainer.n_gpus_per_node", int(args.num_gpus))
    _set(cfg, "trainer.total_epochs", 0)
    _set(cfg, "trainer.total_training_steps", 0)
    _set(cfg, "trainer.val_before_train", True)
    _set(cfg, "trainer.val_only", True)
    _set(cfg, "trainer.test_freq", -1)
    _set(cfg, "trainer.save_freq", -1)

    output_dir = Path(args.output_dir)
    _set(cfg, "trainer.output_dir", output_dir)
    _set(cfg, "trainer.default_local_dir", output_dir)

    _apply_model_overrides(cfg, args)
    _apply_parallel_overrides(cfg, args)
    _apply_logging_overrides(cfg, args)
    _apply_verl_overrides(cfg, final_run_cfg.get("verl_overrides"))
    serialise_resolved_config(final_run_cfg, resolved_cfg_path)

    agent_cls, agent_args, env_cls, env_args = _configure_agent_env(cfg, args)

    _print_run_summary(
        args=args,
        train_path=eval_path,
        val_path=eval_path,
        train_rows=sample_count,
        val_rows=sample_count,
        output_dir=output_dir,
        header="Graph Planner rLLM evaluation launch summary:",
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

    _sanity_checks(train_path=eval_path, val_path=eval_path, args=args)

    from rllm.trainer.agent_trainer import AgentTrainer  # noqa: E402

    trainer = AgentTrainer(
        agent_class=agent_cls,
        env_class=env_cls,
        agent_args=agent_args,
        env_args=env_args,
        config=cfg,
    )

    if args.ray_address:
        os.environ.setdefault("RAY_ADDRESS", args.ray_address)

    trainer.train()

    if wandb_run is not None:  # pragma: no cover - network side effect
        try:
            wandb_run.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
