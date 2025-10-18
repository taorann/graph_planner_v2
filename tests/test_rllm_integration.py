from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from graph_planner.infra.vendor import ensure_rllm_importable
import graph_planner.memory as gp_memory

if not ensure_rllm_importable():  # pragma: no cover - optional dependency
    pytest.skip("rLLM dependency not available", allow_module_level=True)

sys.modules.setdefault("memory", gp_memory)

import rllm  # noqa: F401  # confirm module is importable

from graph_planner.integrations.rllm.agent import GraphPlannerRLLMAgent
from graph_planner.integrations.rllm.cgm_env import CGMRLLMEnv
from graph_planner.integrations.rllm import dataset as dataset_mod


def test_planner_agent_generates_patch(monkeypatch):
    agent = GraphPlannerRLLMAgent(use_rule_fallback=False)

    stub_patch = {"edits": [{"path": "foo.py", "start": 1, "end": 1, "new_text": "print('hi')\n"}], "summary": "fix"}

    monkeypatch.setattr(
        "graph_planner.integrations.rllm.agent.cgm_adapter.generate",
        lambda **_: stub_patch,
    )
    monkeypatch.setattr(
        "graph_planner.integrations.rllm.agent.subgraph_store.linearize",
        lambda *args, **kwargs: [{"path": "foo.py", "text": "body"}],
    )

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
        "{" "\"thought\": \"apply fix\","
        " \"action\": {\"type\": \"repair\", \"apply\": true,"
        " \"plan_targets\": [{\"path\": \"foo.py\", \"start\": 1, \"end\": 1}]}}"
    )

    result = agent.update_from_model(response)
    repair_action = result.action

    assert repair_action.patch == stub_patch
    assert repair_action.plan_targets


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
    assert rows[0]["sandbox"]["r2e_ds_json"] == str(r2e_file.resolve())
    mounts = rows[0]["sandbox"]["mounts"]
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


def test_train_cli_print_config_planner(tmp_path, monkeypatch, capsys):
    dataset_file = Path("datasets/graphplanner_repoenv_sample.jsonl")
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

