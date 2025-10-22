from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

from graph_planner.infra.vendor import ensure_rllm_importable, find_in_rllm
import graph_planner.memory as gp_memory

if not ensure_rllm_importable():  # pragma: no cover - optional dependency
    pytest.skip("rLLM dependency not available", allow_module_level=True)

sys.modules.setdefault("memory", gp_memory)

import rllm  # noqa: F401  # confirm module is importable
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage, compute_policy_loss

from graph_planner.integrations.rllm.agent import GraphPlannerRLLMAgent
from graph_planner.integrations.rllm.cgm_env import CGMRLLMEnv
from graph_planner.integrations.rllm import dataset as dataset_mod
from graph_planner.infra.config import DEFAULT_TRAIN_DATASET
from graph_planner.models.toy_lm import create_toy_checkpoint


DATASET_FILE = Path(DEFAULT_TRAIN_DATASET)


def test_find_in_rllm_handles_namespace_packages():
    cfg_path = find_in_rllm("trainer", "config", "agent_ppo_trainer.yaml")

    assert cfg_path.name == "agent_ppo_trainer.yaml"
    assert cfg_path.is_file()


def test_planner_agent_parses_text_trajectory():
    agent = GraphPlannerRLLMAgent(use_rule_fallback=False)

    observation = {
        "issue": {"id": "demo", "title": "Bug"},
        "subgraph": {"nodes": [{"id": "n1", "path": "foo.py", "span": {"start": 1, "end": 1}}], "edges": []},
        "last_info": {
            "kind": "explore",
            "op": "read",
            "snippets": [{"path": "foo.py", "start": 1, "end": 1, "snippet": ["0001: pass"]}],
        },
    }

    agent.update_from_env(observation, reward=0.0, done=False, info=None)
    response = (
        "<function=repair>\n"
        "  <param name=\"thought\">Investigate buffer bounds</param>\n"
        "  <param name=\"subplan\"><![CDATA[\nFix foo.py bounds check\n]]></param>\n"
        "  <param name=\"focus_ids\">[\"n1\"]</param>\n"
        "  <param name=\"apply\">true</param>\n"
        "</function>"
    )

    result = agent.update_from_model(response)
    repair_action = result.action

    assert repair_action.plan == "Fix foo.py bounds check"
    assert repair_action.apply is True
    assert repair_action.patch is None
    assert repair_action.plan_targets == [{"id": "n1", "why": "planner-focus"}]


def test_cgm_env_uses_provided_plan(monkeypatch):
    class DummyPlanner:
        def __init__(self, *, issue, sandbox_cfg):
            self.issue = issue
            self.config = SimpleNamespace(max_nodes_per_anchor=2)
            self.subgraph = SimpleNamespace(to_json_obj=lambda: {"nodes": [], "edges": []})

        def reset(self):
            return {"issue": self.issue, "subgraph": self.subgraph.to_json_obj()}

        def step(self, action):
            if action.__class__.__name__ == "RepairAction":
                return ({"issue": self.issue}, 0.0, False, {"kind": "repair", "applied": True})
            if action.__class__.__name__ == "SubmitAction":
                return (
                    {"issue": self.issue},
                    1.0,
                    True,
                    {"submit": True, "tests": {"passed": True}},
                )
            return ({"issue": self.issue}, 0.0, False, {})

        def close(self):
            pass

    monkeypatch.setattr("graph_planner.integrations.rllm.cgm_env.PlannerEnv", DummyPlanner)

    entry = {
        "task_id": "demo",
        "issue": {"id": "demo", "title": "Bug"},
        "sandbox": {"docker_image": "dummy", "workdir": "."},
        "plan": {
            "text": "Fix foo",
            "targets": [{"path": "foo.py", "start": 1, "end": 1}],
        },
        "snippets": [{"path": "foo.py", "start": 1, "end": 1, "snippet": ["0001: old"]}],
        "graph": {"nodes": [{"id": "n1", "path": "foo.py", "text": "code"}], "edges": []},
    }

    env = CGMRLLMEnv(entry, max_steps=1)
    obs, info = env.reset()

    assert obs["plan_text"] == "Fix foo"
    assert info["task_id"] == "demo"
    assert info["issue_uid"]
    assert info["source_issue_id"] == "demo"
    assert obs["issue"]["metadata"]["source_issue_id"] == "demo"
    assert obs["issue"]["id"] != "demo"

    patch = {"edits": [{"path": "foo.py", "start": 1, "end": 1, "new_text": "print()\n"}], "summary": "Fix foo"}
    nxt_obs, reward, done, step_info = env.step(SimpleNamespace(action=patch))

    assert done is True
    assert reward == pytest.approx(1.0)
    assert step_info["submit"]["tests"]["passed"] is True


def test_load_task_entries_normalises_paths(tmp_path):
    mount_src = tmp_path / "repo"
    mount_src.mkdir()
    r2e_file = tmp_path / "config.json"
    r2e_file.write_text("{}", encoding="utf-8")

    entry = {
        "task_id": "demo",
        "sandbox": {
            "docker_image": "image",
            "workdir": ".",
            "mounts": {"./repo": "/repo"},
            "r2e_ds_json": "config.json",
        },
    }
    ds_path = tmp_path / "tasks.jsonl"
    ds_path.write_text(json.dumps(entry) + "\n", encoding="utf-8")

    rows = dataset_mod.load_task_entries(ds_path)
    assert rows[0]["task_id"] == "demo"
    raw_entry = json.loads(rows[0]["raw_entry_json"])
    assert raw_entry["sandbox"]["r2e_ds_json"] == str(r2e_file.resolve())
    mounts = raw_entry["sandbox"]["mounts"]
    assert str(mount_src.resolve()) in mounts


def test_ensure_dataset_registered_uses_registry(tmp_path, monkeypatch):
    entry = {"task_id": "demo", "sandbox": {"docker_image": "image", "workdir": "."}}
    ds_path = tmp_path / "tasks.jsonl"
    ds_path.write_text(json.dumps(entry) + "\n", encoding="utf-8")

    calls = {}

    def fake_register(cls, name, data, split):
        calls["name"] = name
        calls["split"] = split
        calls["entries"] = data
        return SimpleNamespace(
            get_verl_data_path=lambda: str(tmp_path / "train_verl.parquet"),
            get_data_path=lambda: str(tmp_path / "train.parquet"),
        )

    monkeypatch.setattr(dataset_mod.DatasetRegistry, "register_dataset", classmethod(fake_register))

    dataset = dataset_mod.ensure_dataset_registered(name="foo", split="train", path=ds_path)

    assert calls["name"] == "foo"
    assert calls["split"] == "train"
    assert isinstance(calls["entries"], list) and calls["entries"]
    assert dataset.get_verl_data_path().endswith("train_verl.parquet")


class _ToyRewardEnv:
    """Utility providing deterministic rewards for GRPO toy experiments."""

    def __init__(self, success_token: str = "POS") -> None:
        self._success_token = success_token

    def score(self, text: str) -> float:
        return 1.0 if self._success_token in text else 0.0


def test_grpo_step_updates_toy_model(tmp_path):
    model_dir = create_toy_checkpoint(tmp_path / "toy_grpo")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.train()

    prompt = "Respond with POS or NEG depending on whether the fix works."
    responses = ["POS", "NEG"]

    env = _ToyRewardEnv()
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    prompt_len = len(prompt_ids)

    response_token_ids = []
    reward_values = []
    for text in responses:
        ids = tokenizer(text, add_special_tokens=False).input_ids
        ids.append(tokenizer.eos_token_id)
        response_token_ids.append(ids)
        reward_values.append(env.score(text))

    max_response_len = max(len(ids) for ids in response_token_ids)
    full_sequences = []
    attention_masks = []
    pad_id = tokenizer.pad_token_id

    for ids in response_token_ids:
        padded = ids + [pad_id] * (max_response_len - len(ids))
        full = prompt_ids + padded
        full_sequences.append(full)
        seq_len = prompt_len + len(ids)
        mask = [1] * seq_len + [0] * (max_response_len - len(ids))
        attention_masks.append(mask)

    input_ids = torch.tensor(full_sequences, dtype=torch.long)
    attention_mask = torch.tensor(attention_masks, dtype=torch.long)

    lm_inputs = input_ids[:, :-1]
    lm_attention_mask = attention_mask[:, :-1]

    outputs = model(lm_inputs, attention_mask=lm_attention_mask)
    logits = outputs.logits
    log_probs_full = torch.log_softmax(logits, dim=-1)

    log_prob_seqs = []
    for idx, ids in enumerate(response_token_ids):
        start = prompt_len - 1
        stop = start + len(ids)
        token_logits = log_probs_full[idx, start:stop, :]
        token_ids = torch.tensor(ids, dtype=torch.long)
        log_prob_seq = token_logits.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
        log_prob_seqs.append(log_prob_seq)

    log_prob = pad_sequence(log_prob_seqs, batch_first=True, padding_value=0.0)
    old_log_prob = log_prob.detach()

    response_mask = pad_sequence(
        [torch.ones(len(ids), dtype=torch.float32) for ids in response_token_ids],
        batch_first=True,
        padding_value=0.0,
    )

    reward_per_token = []
    for value, ids in zip(reward_values, response_token_ids, strict=True):
        if ids:
            reward_tensor = torch.full((len(ids),), float(value) / len(ids), dtype=torch.float32)
        else:
            reward_tensor = torch.zeros(0, dtype=torch.float32)
        reward_per_token.append(reward_tensor)

    token_level_rewards = pad_sequence(reward_per_token, batch_first=True, padding_value=0.0)
    indices = np.zeros(len(responses), dtype=int)

    advantages, _ = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=indices,
        norm_adv_by_std_in_grpo=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    before = model.lm_head.weight.detach().clone()

    policy_loss, *_ = compute_policy_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        cliprange=0.2,
    )

    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

    after = model.lm_head.weight.detach()
    assert not torch.allclose(before, after)


def test_train_cli_print_config_planner(tmp_path, monkeypatch, capsys):
    dataset_file = DATASET_FILE
    assert dataset_file.exists(), "sample dataset missing"

    verl_path = tmp_path / "train_verl.parquet"
    dataset_stub = SimpleNamespace(
        get_verl_data_path=lambda: str(verl_path),
        get_data_path=lambda: str(tmp_path / "train.parquet"),
    )

    def fake_register(*, name, split, path):
        assert name == dataset_mod.GRAPH_PLANNER_DATASET_NAME
        assert split == "train"
        assert Path(path) == dataset_file
        return dataset_stub

    monkeypatch.setattr("scripts.train_graphplanner_rllm.ensure_dataset_registered", fake_register)

    monkeypatch.setenv("WANDB_MODE", "test")
    monkeypatch.setenv("WANDB_PROJECT", "demo")

    argv = [
        "train_graphplanner_rllm.py",
        "--model-path",
        str(tmp_path / "policy"),
        "--print-config",
    ]
    monkeypatch.setattr(sys, "argv", argv, raising=False)

    from scripts import train_graphplanner_rllm as train_mod

    train_mod.main()

    output = capsys.readouterr().out
    assert "data:" in output
    assert str(verl_path) in output


def test_train_cli_parallel_overrides(tmp_path, monkeypatch, capsys):
    dataset_file = DATASET_FILE
    assert dataset_file.exists(), "sample dataset missing"

    verl_path = tmp_path / "train_verl.parquet"
    dataset_stub = SimpleNamespace(
        get_verl_data_path=lambda: str(verl_path),
        get_data_path=lambda: str(tmp_path / "train.parquet"),
    )

    def fake_register(*, name, split, path):
        assert Path(path) == dataset_file
        return dataset_stub

    monkeypatch.setattr("scripts.train_graphplanner_rllm.ensure_dataset_registered", fake_register)

    argv = [
        "train_graphplanner_rllm.py",
        "--model-path",
        str(tmp_path / "policy"),
        "--print-config",
        "--num-gpus",
        "16",
        "--num-nodes",
        "2",
        "--tensor-parallel",
        "8",
        "--parallel-agents",
        "32",
        "--engine-max-workers",
        "96",
        "--rollout-workers",
        "40",
        "--workflow-parallel",
        "64",
        "--rollout-replicas",
        "3",
        "--ray-num-cpus",
        "256",
        "--ray-num-gpus",
        "32",
        "--ray-memory",
        "107374182400",
        "--ray-object-store-memory",
        "21474836480",
    ]
    monkeypatch.setattr(sys, "argv", argv, raising=False)

    from scripts import train_graphplanner_rllm as train_mod

    train_mod.main()

    output = capsys.readouterr().out
    assert "n_gpus_per_node: 16" in output
    assert "nnodes: 2" in output
    assert "tensor_model_parallel_size: 8" in output
    assert "'n': 3" in output
    assert "n_parallel_agents: 32" in output
    assert "max_workers: 96" in output
    assert "num_workers: 40" in output
    assert "n_parallel_tasks: 64" in output
    assert "num_cpus: 256" in output
    assert "num_gpus: 32" in output
    assert "memory: 107374182400" in output
    assert "object_store_memory: 21474836480" in output


def test_train_cli_training_overrides(tmp_path, monkeypatch, capsys):
    dataset_file = DATASET_FILE
    assert dataset_file.exists(), "sample dataset missing"

    train_verl = tmp_path / "train_verl.parquet"
    val_verl = tmp_path / "val_verl.parquet"

    def fake_register(*, name, split, path):
        assert Path(path) == dataset_file
        target = train_verl if "val" not in name else val_verl
        return SimpleNamespace(
            get_verl_data_path=lambda: str(target),
            get_data_path=lambda: str(target.with_suffix(".raw.parquet")),
        )

    monkeypatch.setattr("scripts.train_graphplanner_rllm.ensure_dataset_registered", fake_register)

    argv = [
        "train_graphplanner_rllm.py",
        "--model-path",
        str(tmp_path / "policy"),
        "--output-dir",
        str(tmp_path / "outputs"),
        "--val-dataset",
        str(dataset_file),
        "--save-interval",
        "10",
        "--eval-interval",
        "20",
        "--resume",
        str(tmp_path / "resume"),
        "--precision",
        "fp16",
        "--grad-accum-steps",
        "3",
        "--lr",
        "0.0001",
        "--weight-decay",
        "0.01",
        "--warmup-steps",
        "5",
        "--early-stop-metric",
        "val/test_score/pass@k/graph_planner",
        "--early-stop-mode",
        "max",
        "--early-stop-patience",
        "2",
        "--log-to-wandb",
        "--wandb-offline",
        "--project-name",
        "demo_proj",
        "--experiment-name",
        "demo_exp",
        "--print-config",
    ]
    monkeypatch.setattr(sys, "argv", argv, raising=False)

    from scripts import train_graphplanner_rllm as train_mod

    train_mod.main()

    output = capsys.readouterr().out
    assert "save_interval: 10" in output
    assert "test_freq: 20" in output
    assert "resume_from:" in output and "resume_mode: manual" in output
    assert "mixed_precision: fp16" in output
    assert "gradient_accumulation_steps: 3" in output
    assert "optimizer:" in output and "lr: 0.0001" in output
    assert "weight_decay: 0.01" in output
    assert "scheduler:" in output and "warmup_steps: 5" in output
    assert "early_stop:" in output and "metric: val/test_score/pass@k/graph_planner" in output
    assert "logger:" in output and "wandb" in output


def test_eval_cli_print_config(tmp_path, monkeypatch, capsys):
    dataset_file = DATASET_FILE
    assert dataset_file.exists(), "sample dataset missing"

    verl_path = tmp_path / "eval_verl.parquet"

    def fake_register(*, name, split, path):
        assert Path(path) == dataset_file
        return SimpleNamespace(
            get_verl_data_path=lambda: str(verl_path),
            get_data_path=lambda: str(verl_path.with_suffix(".raw.parquet")),
        )

    monkeypatch.setattr("scripts.eval_graphplanner_rllm.ensure_dataset_registered", fake_register)

    argv = [
        "eval_graphplanner_rllm.py",
        "--dataset",
        str(dataset_file),
        "--model-path",
        str(tmp_path / "policy"),
        "--print-config",
    ]
    monkeypatch.setattr(sys, "argv", argv, raising=False)

    from scripts import eval_graphplanner_rllm as eval_mod

    eval_mod.main()

    output = capsys.readouterr().out
    assert "val_only: true" in output
    assert str(verl_path) in output
    assert "Graph Planner rLLM evaluation launch summary:" in output


def test_train_cli_config_file_and_resolved_config(tmp_path, monkeypatch, capsys):
    dataset_file = DATASET_FILE
    assert dataset_file.exists(), "sample dataset missing"

    verl_path = tmp_path / "train_verl.parquet"
    dataset_stub = SimpleNamespace(
        get_verl_data_path=lambda: str(verl_path),
        get_data_path=lambda: str(tmp_path / "train.raw.parquet"),
    )
    val_stub = SimpleNamespace(
        get_verl_data_path=lambda: str(tmp_path / "val_verl.parquet"),
        get_data_path=lambda: str(tmp_path / "val.raw.parquet"),
    )

    def fake_register(*, name, split, path):
        target = val_stub if ("val" in split or str(name).endswith("_val")) else dataset_stub
        return target

    monkeypatch.setattr("scripts.train_graphplanner_rllm.ensure_dataset_registered", fake_register)

    yaml_cfg = {
        "experiment": {"seed": 1234, "name": "yaml-exp"},
        "paths": {"dataset_train": str(dataset_file)},
        "training": {"train_batch_size": 8, "grad_accum_steps": 4},
        "parallel": {"tensor_parallel_planner": 1, "tensor_parallel_cgm": 1, "replicas": 1},
        "resources": {"num_gpus": 4, "ray_num_gpus": 2, "ray_num_cpus": 16},
        "logging": {
            "output_dir": str(tmp_path / "runs"),
            "wandb": {"enabled": True, "run_name": "yaml-run", "project": "yaml-proj"},
        },
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(yaml_cfg), encoding="utf-8")

    metrics = []

    def fake_log_metrics(step, payload):
        metrics.append((step, payload))

    monkeypatch.setattr("scripts.train_graphplanner_rllm.log_metrics", fake_log_metrics)
    monkeypatch.setattr(
        "scripts.train_graphplanner_rllm.make_gpu_snapshot",
        lambda: (lambda: {"gpu/0/util": 10.0}),
    )
    monkeypatch.setattr(
        "scripts.train_graphplanner_rllm.make_ray_snapshot",
        lambda: (lambda: {"ray/cpus_avail": 4.0}),
    )
    monkeypatch.setattr("scripts.train_graphplanner_rllm.init_wandb", lambda **_: None)

    argv = [
        "train_graphplanner_rllm.py",
        "--agent",
        "planner",
        "--config-file",
        str(cfg_path),
        "--model-path",
        str(tmp_path / "policy"),
        "--print-config",
        "--train-batch-size",
        "16",
        "--tensor-parallel",
        "2",
        "--parallel-agents",
        "3",
        "--rollout-workers",
        "3",
        "--workflow-parallel",
        "3",
        "--num-gpus",
        "4",
        "--ray-num-gpus",
        "4",
        "--ray-num-cpus",
        "16",
    ]
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setattr(sys, "argv", argv, raising=False)

    from scripts import train_graphplanner_rllm as train_mod

    train_mod.main()

    output = capsys.readouterr().out
    assert "tensor_model_parallel_size: 2" in output
    assert "train_batch_size: 16" in output

    assert metrics and metrics[0][0] == 0
    assert metrics[0][1]["parallel/tensor_parallel_planner"] == 2

    resolved_cfg = tmp_path / "runs" / "yaml-run" / "resolved_config.yaml"
    assert resolved_cfg.is_file()
    resolved = yaml.safe_load(resolved_cfg.read_text(encoding="utf-8"))
    assert resolved["training"]["train_batch_size"] == 16
    assert resolved["parallel"]["tensor_parallel_planner"] == 2
    assert resolved["logging"]["wandb"]["run_name"] == "yaml-run"


def test_train_cli_yaml_only_ignores_overrides(tmp_path, monkeypatch, capsys):
    dataset_file = DATASET_FILE
    assert dataset_file.exists(), "sample dataset missing"

    verl_path = tmp_path / "train_verl.parquet"

    val_stub = SimpleNamespace(
        get_verl_data_path=lambda: str(tmp_path / "val_verl.parquet"),
        get_data_path=lambda: str(tmp_path / "val.raw.parquet"),
    )

    def fake_register(*, name, split, path):
        if "val" in split or str(name).endswith("_val"):
            return val_stub
        assert Path(path) == dataset_file
        return SimpleNamespace(
            get_verl_data_path=lambda: str(verl_path),
            get_data_path=lambda: str(tmp_path / "train.raw.parquet"),
        )

    monkeypatch.setattr("scripts.train_graphplanner_rllm.ensure_dataset_registered", fake_register)

    yaml_cfg = {
        "training": {"train_batch_size": 5},
        "parallel": {"tensor_parallel_planner": 1, "tensor_parallel_cgm": 1, "replicas": 1},
        "resources": {"num_gpus": 2, "ray_num_gpus": 2, "ray_num_cpus": 8},
        "logging": {"output_dir": str(tmp_path / "runs"), "wandb": {"run_name": "yaml-only"}},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(yaml_cfg), encoding="utf-8")

    monkeypatch.setattr("scripts.train_graphplanner_rllm.log_metrics", lambda *_, **__: None)
    monkeypatch.setattr("scripts.train_graphplanner_rllm.make_gpu_snapshot", lambda: (lambda: {}))
    monkeypatch.setattr("scripts.train_graphplanner_rllm.make_ray_snapshot", lambda: (lambda: {}))
    monkeypatch.setattr("scripts.train_graphplanner_rllm.init_wandb", lambda **_: None)

    argv = [
        "train_graphplanner_rllm.py",
        "--agent",
        "planner",
        "--config-file",
        str(cfg_path),
        "--yaml-only",
        "--model-path",
        str(tmp_path / "policy"),
        "--print-config",
        "--train-batch-size",
        "16",
        "--tensor-parallel",
        "2",
        "--num-gpus",
        "4",
    ]
    monkeypatch.setattr(sys, "argv", argv, raising=False)

    from scripts import train_graphplanner_rllm as train_mod

    train_mod.main()

    resolved_cfg = tmp_path / "runs" / "yaml-only" / "resolved_config.yaml"
    saved = yaml.safe_load(resolved_cfg.read_text(encoding="utf-8"))
    assert saved["training"]["train_batch_size"] == 5


def test_train_cli_preflight_failure(tmp_path, monkeypatch):
    dataset_file = DATASET_FILE
    assert dataset_file.exists(), "sample dataset missing"

    verl_path = tmp_path / "train_verl.parquet"

    def fake_register(*, name, split, path):
        assert Path(path) == dataset_file
        return SimpleNamespace(
            get_verl_data_path=lambda: str(verl_path),
            get_data_path=lambda: str(tmp_path / "train.raw.parquet"),
        )

    monkeypatch.setattr("scripts.train_graphplanner_rllm.ensure_dataset_registered", fake_register)

    yaml_cfg = {
        "parallel": {"tensor_parallel_planner": 4, "tensor_parallel_cgm": 4, "replicas": 1},
        "resources": {"num_gpus": 4, "ray_num_gpus": 1, "ray_num_cpus": 4},
        "logging": {"output_dir": str(tmp_path / "runs"), "wandb": {"run_name": "bad"}},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(yaml_cfg), encoding="utf-8")

    monkeypatch.setattr("scripts.train_graphplanner_rllm.log_metrics", lambda *_, **__: None)
    monkeypatch.setattr("scripts.train_graphplanner_rllm.make_gpu_snapshot", lambda: (lambda: {}))
    monkeypatch.setattr("scripts.train_graphplanner_rllm.make_ray_snapshot", lambda: (lambda: {}))
    monkeypatch.setattr("scripts.train_graphplanner_rllm.init_wandb", lambda **_: None)

    argv = [
        "train_graphplanner_rllm.py",
        "--agent",
        "planner",
        "--config-file",
        str(cfg_path),
        "--model-path",
        str(tmp_path / "policy"),
        "--print-config",
    ]
    monkeypatch.setattr(sys, "argv", argv, raising=False)

    from scripts import train_graphplanner_rllm as train_mod

    with pytest.raises(ValueError) as exc:
        train_mod.main()
    assert "Parallel resource configuration invalid" in str(exc.value)

