#!/usr/bin/env python
"""Launch PPO training for the Graph Planner agent via rLLM/VERL."""

from __future__ import annotations

import argparse
from pathlib import Path

from graph_planner.infra.vendor import ensure_rllm_importable

ensure_rllm_importable()

import ray
from omegaconf import OmegaConf

from graph_planner.integrations.rllm import (
    GRAPH_PLANNER_DATASET_NAME,
    GraphPlannerRLLMAgent,
    GraphPlannerRLLMEnv,
    ensure_dataset_registered,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/graphplanner_repoenv_sample.jsonl"),
        help="Graph Planner RL task specification (JSON/JSONL).",
    )
    parser.add_argument(
        "--dataset-name",
        default=GRAPH_PLANNER_DATASET_NAME,
        help="Dataset registry name (default: %(default)s).",
    )
    parser.add_argument(
        "--dataset-split",
        default="train",
        help="Dataset split name (default: %(default)s).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=False,
        help="Path to the policy checkpoint to fine-tune (required for actual training).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=6,
        help="Maximum environment steps per episode (default: %(default)s).",
    )
    parser.add_argument(
        "--use-fallback",
        action="store_true",
        help="Enable rule-based fallback parsing for debugging runs.",
    )
    parser.add_argument(
        "--ray-address",
        default=None,
        help="Optional Ray cluster address; defaults to local in-process cluster.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Dump the resolved Hydra config instead of launching training.",
    )
    parser.add_argument(
        "--trainer-epochs",
        type=int,
        default=1,
        help="Number of PPO epochs to schedule (default: %(default)s).",
    )
    return parser.parse_args()


def load_base_config() -> OmegaConf:
    import rllm as rllm_pkg

    base_cfg = (
        Path(rllm_pkg.__file__).resolve().parent / "trainer" / "config" / "ppo_trainer.yaml"
    )
    return OmegaConf.load(str(base_cfg))


def build_config(args: argparse.Namespace, dataset_path: Path) -> OmegaConf:
    cfg = load_base_config()

    train_path = str(dataset_path)
    cfg.data.train_files = train_path
    cfg.data.val_files = train_path
    cfg.data.train_batch_size = 1
    cfg.data.val_batch_size = 1
    cfg.data.shuffle = False
    if cfg.data.max_prompt_length:
        cfg.data.max_prompt_length = min(cfg.data.max_prompt_length, 4096)
    if cfg.data.max_response_length:
        cfg.data.max_response_length = min(cfg.data.max_response_length, 2048)

    cfg.env.name = "graph_planner_repoenv"
    cfg.env.env_args = {"max_steps": args.max_steps}

    cfg.agent.name = "graph_planner_repoenv"
    cfg.agent.max_steps = args.max_steps
    cfg.agent.agent_args = {"use_rule_fallback": args.use_fallback}
    if hasattr(cfg.agent, "trajectory_timeout"):
        cfg.agent.trajectory_timeout = max(args.max_steps * 120, cfg.agent.trajectory_timeout)

    cfg.trainer.total_epochs = args.trainer_epochs
    cfg.trainer.project_name = cfg.trainer.project_name or "graph-planner"
    cfg.trainer.experiment_name = cfg.trainer.experiment_name or "graph-planner-rllm"

    if args.model_path:
        model_path = str(args.model_path)
        cfg.actor_rollout_ref.model.path = model_path
        cfg.critic.model.path = model_path
        cfg.critic.model.tokenizer_path = model_path
    return cfg


def main() -> None:
    args = parse_args()
    dataset = ensure_dataset_registered(
        name=args.dataset_name,
        split=args.dataset_split,
        path=args.dataset,
    )
    verl_path = dataset.get_verl_data_path()
    if not verl_path:
        raise RuntimeError("Dataset registration did not generate a Verl parquet file")

    cfg = build_config(args, Path(verl_path))

    if args.print_config:
        print(OmegaConf.to_yaml(cfg))
        return

    if not ray.is_initialized():
        ray.init(address=args.ray_address, ignore_reinit_error=True)

    from rllm.trainer.verl.train_agent_ppo import train_agent

    trainer_ref = train_agent.options(num_cpus=1).remote(
        cfg,
        agent_class=GraphPlannerRLLMAgent,
        env_class=GraphPlannerRLLMEnv,
        agent_args={"use_rule_fallback": args.use_fallback},
        env_args={"max_steps": args.max_steps},
    )
    ray.get(trainer_ref)


if __name__ == "__main__":
    main()
