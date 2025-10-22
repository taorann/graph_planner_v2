from __future__ import annotations

import json
from pathlib import Path

import pytest

from graph_planner.datasets import convert_r2e_entries, convert_swebench_entries


@pytest.fixture()
def tmp_dataset_dir(tmp_path: Path) -> Path:
    return tmp_path / "dataset"


def test_convert_r2e_entries_creates_json_and_jsonl(tmp_dataset_dir: Path):
    entries = [
        {
            "task_id": "numpy#123",
            "max_steps": 32,
            "problem_statement": "Fix numpy bug",
            "gym_config": {
                "docker_image": "r2e/numpy:latest",
                "repo_name": "numpy",
                "parsed_commit_content": json.dumps({"file_diffs": []}),
                "repo_path": "/repo",
            },
        }
    ]

    result = convert_r2e_entries(entries, output_dir=tmp_dataset_dir, dataset_name="demo/r2e", split="train")

    assert len(result.records) == 1
    record = result.records[0]
    assert record["task_id"] == "numpy#123"
    assert record["sandbox"]["backend"] == "repoenv"
    rel_path = Path(record["sandbox"]["r2e_ds_json"])
    assert (tmp_dataset_dir / rel_path).is_file()
    with (tmp_dataset_dir / rel_path).open("r", encoding="utf-8") as handle:
        saved = json.load(handle)
    assert saved["docker_image"] == "r2e/numpy:latest"


def test_convert_swebench_entries_requires_docker_image(tmp_dataset_dir: Path):
    entries = [
        {
            "instance_id": "django__001",
            "problem_statement": "Fix Django bug",
            "docker_image": "swebench/django:latest",
            "repo": "django/django",
        }
    ]

    result = convert_swebench_entries(entries, output_dir=tmp_dataset_dir, dataset_name="demo/swe", split="test")

    assert len(result.records) == 1
    record = result.records[0]
    rel_path = Path(record["sandbox"]["r2e_ds_json"])
    ds_file = tmp_dataset_dir / rel_path
    assert ds_file.exists()
    payload = json.loads(ds_file.read_text(encoding="utf-8"))
    assert payload["instance_id"] == "django__001"
    assert payload["docker_image"] == "swebench/django:latest"
    assert record["issue"]["title"] == "Fix Django bug"
