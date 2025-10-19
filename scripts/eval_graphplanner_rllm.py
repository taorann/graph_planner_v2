"""评估 Graph Planner/CGM 代理在 rLLM 上的推理表现（仅验证，不更新参数）。

English summary
    Runs the rLLM PPO pipeline in validation-only mode so we can collect
    pass@k, success rate, and trajectory statistics without performing any
    optimisation steps.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict

from omegaconf import OmegaConf

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
)
from scripts.train_graphplanner_rllm import (  # noqa: E402
    _apply_logging_overrides,
    _apply_model_overrides,
    _apply_parallel_overrides,
    _configure_agent_env,
    _default_config_path,
    _load_config,
    _print_run_summary,
    _sanity_checks,
    _seed_everything,
    _set,
    _validate_parallel_config,
)


LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Graph Planner agents with rLLM (validation only)"
    )
    parser.add_argument("--agent", choices=["planner", "cgm"], default="planner")
    parser.add_argument("--dataset", type=Path, required=True, help="Evaluation dataset in JSON/JSONL format.")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-split", default="eval")
    parser.add_argument("--model-path", type=Path, required=True, help="Policy checkpoint to evaluate.")
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--critic-model-path", type=Path, default=None)
    parser.add_argument("--critic-tokenizer-path", type=Path, default=None)
    parser.add_argument("--cgm-model-path", type=Path, default=None)
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
    parser.add_argument("--config", type=Path, default=_default_config_path())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--print-config", action="store_true")
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
    return parser.parse_args()


def _resolve_eval_dataset(path: Path, *, name: str, split: str) -> tuple[str, int]:
    rows = load_task_entries(path)
    if not rows:
        raise RuntimeError(f"Dataset {path} did not contain any rows")
    dataset = ensure_dataset_registered(name=name, split=split, path=path)
    verl_path = dataset.get_verl_data_path()
    if not verl_path:
        raise RuntimeError("Dataset registration did not produce a Verl parquet file")
    return verl_path, len(rows)


def main() -> None:
    args = _parse_args()

    logging.basicConfig(level=logging.INFO)
    _seed_everything(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    _validate_parallel_config(args)

    dataset_name = args.dataset_name
    if not dataset_name:
        dataset_name = GRAPH_PLANNER_CGM_DATASET_NAME if args.agent == "cgm" else GRAPH_PLANNER_DATASET_NAME
    # Avoid clashing with training splits by suffixing _eval.
    eval_dataset_name = f"{dataset_name}_eval"
    eval_path, sample_count = _resolve_eval_dataset(
        args.dataset,
        name=eval_dataset_name,
        split=args.dataset_split,
    )

    cfg = _load_config(args.config)

    _set(cfg, "data.train_files", str(eval_path))
    _set(cfg, "data.val_files", str(eval_path))
    _set(cfg, "data.train_batch_size", int(args.batch_size))
    _set(cfg, "data.val_batch_size", int(args.batch_size))
    _set(cfg, "data.shuffle", False)

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

    output_dir = Path(args.output_dir or args.model_path)
    _set(cfg, "trainer.output_dir", output_dir)
    _set(cfg, "trainer.default_local_dir", output_dir)

    _apply_model_overrides(cfg, args)
    _apply_parallel_overrides(cfg, args)
    _apply_logging_overrides(cfg, args)

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

    if args.print_config:
        print(OmegaConf.to_yaml(cfg))
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


if __name__ == "__main__":
    main()
