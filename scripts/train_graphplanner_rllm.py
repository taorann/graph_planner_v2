"""使用 rLLM 训练 Graph Planner/CGM 代理的命令行脚本。

English summary
    Provides a thin CLI that prepares datasets, config overrides and agent/env
    wiring before delegating to rLLM's PPO trainer.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Tuple

from omegaconf import OmegaConf

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
)


def _default_config_path() -> Path:
    """返回 rLLM 默认 PPO 配置文件路径。"""

    return find_in_rllm("trainer", "config", "agent_ppo_trainer.yaml")


def _parse_args() -> argparse.Namespace:
    """解析命令行参数，支持模型/数据集/调度配置。"""

    parser = argparse.ArgumentParser(description="Train Graph Planner agents with rLLM PPO")
    parser.add_argument("--agent", choices=["planner", "cgm"], default="planner")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/graphplanner_repoenv_sample.jsonl"),
        help="Training dataset in JSON/JSONL format.",
    )
    parser.add_argument("--dataset-name", default=None, help="Override dataset registry name.")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--val-dataset", type=Path, default=None)
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--model-path", type=Path, required=True, help="Target policy checkpoint path.")
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
        default=None,
        help="Optional CGM model used for planner patch synthesis.",
    )
    parser.add_argument("--cgm-tokenizer-path", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=None)
    parser.add_argument("--total-epochs", type=int, default=1)
    parser.add_argument("--total-steps", type=int, default=None)
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
    parser.add_argument("--ray-address", default=None)
    parser.add_argument("--ray-num-cpus", type=int, default=None)
    parser.add_argument("--ray-num-gpus", type=int, default=None)
    parser.add_argument("--ray-memory", type=int, default=None)
    parser.add_argument("--ray-object-store-memory", type=int, default=None)
    parser.add_argument("--config", type=Path, default=_default_config_path())
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--use-fallback", action="store_true", help="Enable rule fallback when training planner agent.")
    parser.add_argument(
        "--cgm-instruction",
        default=None,
        help="Instruction prompt passed to the CGM environment (cgm agent only).",
    )
    parser.add_argument("--project-name", default=None)
    parser.add_argument("--experiment-name", default=None)
    return parser.parse_args()


def _load_config(path: Path) -> OmegaConf:
    """从 YAML 文件加载 ``OmegaConf`` 配置。"""

    return OmegaConf.load(str(path))


def _set_if_exists(cfg: OmegaConf, key: str, value: Any) -> None:
    """仅在键已存在时更新配置值。"""

    if value is None:
        return
    if OmegaConf.select(cfg, key) is not None:
        OmegaConf.update(cfg, key, value, merge=False)


def _set(cfg: OmegaConf, key: str, value: Any) -> None:
    """在 OmegaConf 中写入（或创建）指定键。"""

    if value is None:
        return
    OmegaConf.update(cfg, key, value, merge=True)


def _resolve_dataset(
    *,
    dataset_path: Path,
    dataset_name: str,
    split: str,
    val_dataset: Path | None,
    val_split: str,
) -> Tuple[str, str]:
    """注册训练/验证集并返回 rLLM 生成的 verl parquet 路径。"""

    train_ds = ensure_dataset_registered(name=dataset_name, split=split, path=dataset_path)
    train_path = train_ds.get_verl_data_path()
    if not train_path:
        raise RuntimeError("Training dataset registration did not produce a Verl parquet file")

    if val_dataset is None:
        return train_path, train_path

    val_ds = ensure_dataset_registered(name=f"{dataset_name}_val", split=val_split, path=val_dataset)
    val_path = val_ds.get_verl_data_path()
    if not val_path:
        raise RuntimeError("Validation dataset registration did not produce a Verl parquet file")
    return train_path, val_path


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


def _configure_agent_env(cfg: OmegaConf, args: argparse.Namespace) -> Tuple[type, Dict[str, Any], type, Dict[str, Any]]:
    """返回需注册的 Agent/Env 类及其构造参数，并同步写入配置。"""

    if args.agent == "planner":
        agent_cls = GraphPlannerRLLMAgent
        env_cls = GraphPlannerRLLMEnv
        agent_args: Dict[str, Any] = {"use_rule_fallback": bool(args.use_fallback)}
        env_args: Dict[str, Any] = {"max_steps": int(args.max_steps)}

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
        }
        _set_if_exists(cfg, "rllm.agent.name", "graph_planner_cgm")
        _set_if_exists(cfg, "rllm.env.name", "graph_planner_cgm")
        _set_if_exists(cfg, "agent.name", "graph_planner_cgm")
        _set_if_exists(cfg, "env.name", "graph_planner_cgm")

    _set_if_exists(cfg, "agent.max_steps", int(args.max_steps))
    _set_if_exists(cfg, "env.env_args.max_steps", int(args.max_steps))
    return agent_cls, agent_args, env_cls, env_args


def main() -> None:
    """脚本入口：解析参数、准备数据集并触发 rLLM 训练。"""

    args = _parse_args()

    dataset_name = args.dataset_name
    if not dataset_name:
        dataset_name = GRAPH_PLANNER_CGM_DATASET_NAME if args.agent == "cgm" else GRAPH_PLANNER_DATASET_NAME

    train_path, val_path = _resolve_dataset(
        dataset_path=args.dataset,
        dataset_name=dataset_name,
        split=args.dataset_split,
        val_dataset=args.val_dataset,
        val_split=args.val_split,
    )

    cfg = _load_config(args.config)

    _set(cfg, "data.train_files", str(train_path))
    _set(cfg, "data.val_files", str(val_path))
    _set(cfg, "data.train_batch_size", int(args.train_batch_size))
    if args.val_batch_size is not None:
        _set(cfg, "data.val_batch_size", int(args.val_batch_size))
    _set(cfg, "data.shuffle", False)

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
    agent_cls, agent_args, env_cls, env_args = _configure_agent_env(cfg, args)

    if args.print_config:
        print(OmegaConf.to_yaml(cfg))
        return

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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

