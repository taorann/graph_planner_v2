from __future__ import annotations

from pathlib import Path
from pathlib import Path
from types import SimpleNamespace

import yaml

from graph_planner.infra.config import (
    build_cli_overrides,
    default_training_run_config,
    load_run_config_file,
    merge_run_config,
    serialise_resolved_config,
    update_args_from_config,
)


def _make_args(**overrides):
    defaults = dict(
        agent="planner",
        seed=99,
        dataset=Path("cli-train.jsonl"),
        dataset_split="train",
        val_dataset=None,
        model_path=Path("cli-model"),
        tokenizer_path=None,
        cgm_model_path=None,
        cgm_tokenizer_path=None,
        train_batch_size=16,
        total_epochs=3,
        grad_accum_steps=2,
        lr=0.0003,
        weight_decay=0.01,
        warmup_steps=5,
        total_steps=200,
        resume=Path("resume-run"),
        temperature=0.7,
        top_p=0.8,
        max_input_tokens=2048,
        max_output_tokens=256,
        stop=["STOP"],
        stop_ids=[1, 2],
        tensor_parallel=4,
        rollout_replicas=2,
        parallel_agents=3,
        rollout_workers=5,
        workflow_parallel=6,
        num_gpus=8,
        num_nodes=2,
        ray_num_gpus=4,
        ray_num_cpus=32,
        ray_memory=1024,
        ray_object_store_memory=512,
        log_backend="tensorboard",
        output_dir=Path("cli-output"),
        save_interval=10,
        eval_interval=20,
        log_to_wandb=True,
        wandb_offline=True,
        project_name="cli-project",
        experiment_name="cli-run",
        max_steps=7,
        reward_scale=1.5,
        failure_penalty=0.2,
        step_penalty=0.05,
        timeout_penalty=0.3,
        repo_op_limit=10,
        disable_cgm_synthesis=True,
        apply_patches=True,
        docker_manifest=Path("cli-manifest.txt"),
        prepull_containers=True,
        prepull_max_workers=8,
        prepull_retries=3,
        prepull_delay=5,
        prepull_timeout=300,
        cgm_synthesis_strategy=None,
        ray_address=None,
        cgm_instruction=None,
        val_split="val",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_merge_run_config_priority(tmp_path):
    defaults = default_training_run_config("planner")
    yaml_cfg = {
        "experiment": {"seed": 1234},
        "paths": {
            "dataset_train": "yaml-train.jsonl",
            "planner_model": "yaml-model",
            "cgm_model": "yaml-cgm",
        },
        "training": {"train_batch_size": 8, "grad_accum_steps": 4},
        "parallel": {
            "tensor_parallel_planner": 2,
            "tensor_parallel_cgm": 1,
            "replicas": 1,
            "parallel_agents": 2,
            "rollout_workers": 2,
            "workflow_parallel": 3,
        },
        "resources": {"num_gpus": 12, "num_nodes": 1, "ray_num_gpus": 6, "ray_num_cpus": 48},
        "logging": {
            "output_dir": str(tmp_path / "yaml-out"),
            "wandb": {"enabled": True, "run_name": "yaml-run", "project": "yaml-proj"},
        },
    }
    args = _make_args()

    cli_overrides = build_cli_overrides(args, mode="train")
    merged = merge_run_config(defaults, yaml_cfg, cli_overrides, yaml_only=False)

    assert merged["experiment"]["seed"] == 99  # CLI override wins
    assert merged["training"]["train_batch_size"] == 16
    assert merged["training"]["grad_accum_steps"] == 2
    assert merged["paths"]["planner_model"].endswith("cli-model")
    assert merged["parallel"]["tensor_parallel_planner"] == 4
    assert merged["parallel"]["parallel_agents"] == 3
    assert merged["logging"]["wandb"]["enabled"] is True
    assert merged["logging"]["wandb"]["offline"] is True
    assert merged["env"]["docker_manifest"].endswith("cli-manifest.txt")
    assert merged["env"]["prepull_containers"] is True
    assert merged["env"]["prepull_max_workers"] == 8
    assert merged["env"]["prepull_retries"] == 3
    assert merged["env"]["prepull_delay"] == 5
    assert merged["env"]["prepull_timeout"] == 300

    update_args_from_config(args, merged)
    assert args.train_batch_size == 16
    assert args.grad_accum_steps == 2
    assert args.tensor_parallel == 4
    assert args.parallel_agents == 3
    assert args.rollout_workers == 5
    assert args.save_interval == 10
    assert args.output_dir == Path("cli-output").resolve()
    assert args.project_name == "cli-project"
    assert args.disable_cgm_synthesis is True
    assert args.docker_manifest == Path("cli-manifest.txt").resolve()
    assert args.prepull_containers is True
    assert args.prepull_max_workers == 8
    assert args.prepull_retries == 3
    assert args.prepull_delay == 5
    assert args.prepull_timeout == 300

    resolved = tmp_path / "resolved.yaml"
    serialise_resolved_config(merged, resolved)
    saved = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    assert saved["training"]["train_batch_size"] == 16


def test_merge_run_config_yaml_only_blocks_cli(tmp_path):
    defaults = default_training_run_config("planner")
    yaml_cfg = {
        "experiment": {"seed": 1234},
        "training": {"train_batch_size": 8},
        "logging": {"output_dir": str(tmp_path / "yaml-out")},
    }
    args = _make_args()

    cli_overrides = build_cli_overrides(args, mode="train")
    merged = merge_run_config(defaults, yaml_cfg, cli_overrides, yaml_only=True)

    assert merged["experiment"]["seed"] == 1234
    assert merged["training"]["train_batch_size"] == 8

    update_args_from_config(args, merged)
    assert args.train_batch_size == 8
    assert args.seed == 1234


def test_load_run_config_file_roundtrip(tmp_path):
    yaml_cfg = {"training": {"train_batch_size": 2}}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(yaml_cfg), encoding="utf-8")

    loaded = load_run_config_file(path)
    assert loaded == yaml_cfg
