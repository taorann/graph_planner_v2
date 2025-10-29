from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Optional

import pytest

from graph_planner.datasets import (
    DatasetConversionResult,
    convert_r2e_entries,
    convert_swebench_entries,
    prepare as datasets_prepare,
)
from scripts import prepare_datasets


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
    assert result.skipped == 0
    record = result.records[0]
    assert record["task_id"] == "numpy#123"
    assert record["sandbox"]["backend"] == "repoenv"
    record_path = Path(record["sandbox"]["r2e_ds_json"])
    assert record_path.is_absolute()
    assert record_path.is_relative_to(tmp_dataset_dir)
    assert record_path.is_file()
    saved = json.loads(record_path.read_text(encoding="utf-8"))
    assert saved["docker_image"] == "r2e/numpy:latest"
    assert saved["task_id"] == "numpy#123"


def test_convert_r2e_entries_uses_nested_ds_identifier(tmp_dataset_dir: Path):
    entries = [
        {
            "ds": {
                "instance_id": "pytorch__issue-42",
                "docker_image": "pytorch:latest",
                "repo": "pytorch/pytorch",
                "repo_path": "/repo",
            },
            "problem_statement": "Investigate bug",
        }
    ]

    result = convert_r2e_entries(entries, output_dir=tmp_dataset_dir, dataset_name="demo/r2e", split="dev")

    assert [record["task_id"] for record in result.records] == ["pytorch__issue-42"]
    assert result.skipped == 0
    record_path = Path(result.records[0]["sandbox"]["r2e_ds_json"])
    assert record_path.is_absolute()
    assert record_path.is_relative_to(tmp_dataset_dir)
    assert record_path.suffix == ".json"


def test_convert_r2e_entries_generates_fallback_identifier(tmp_dataset_dir: Path):
    entries = [
        {
            "gym_config": {
                "docker_image": "fallback:latest",
            },
        },
        {
            "docker_image": "fallback:2",
        },
    ]

    result = convert_r2e_entries(entries, output_dir=tmp_dataset_dir, dataset_name="demo/r2e", split="train")

    assert len(result.records) == 2
    assert result.skipped == 0
    for record in result.records:
        assert record["task_id"].startswith("demo/r2e:train:")
        record_path = Path(record["sandbox"]["r2e_ds_json"])
        assert record_path.is_absolute()
        assert record_path.is_relative_to(tmp_dataset_dir)
        assert record_path.name.endswith(".json")


def test_convert_r2e_entries_skips_missing_metadata(tmp_dataset_dir: Path):
    entries = [
        {
            "task_id": "missing-docker",
        },
        {
            "ds": {"docker_image": "demo:image"},
        },
    ]

    result = convert_r2e_entries(entries, output_dir=tmp_dataset_dir, dataset_name="demo/r2e", split="train")

    assert len(result.records) == 1
    assert result.skipped == 1
    record = result.records[0]
    assert record["sandbox"]["docker_image"] == "demo:image"


def test_convert_swebench_entries_requires_docker_image(tmp_dataset_dir: Path, monkeypatch):
    monkeypatch.setattr(datasets_prepare, "_make_swebench_spec", lambda entry: None)
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
    assert result.skipped == 0
    record = result.records[0]
    record_path = Path(record["sandbox"]["r2e_ds_json"])
    assert record_path.is_absolute()
    assert record_path.is_relative_to(tmp_dataset_dir)
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    assert payload["instance_id"] == "django__001"
    assert payload["docker_image"] == "swebench/django:latest"
    assert "requires_build" not in record["sandbox"]
    assert record["issue"]["title"] == "Fix Django bug"


def test_convert_swebench_entries_supports_environment_block(tmp_dataset_dir: Path, monkeypatch):
    monkeypatch.setattr(datasets_prepare, "_make_swebench_spec", lambda entry: None)
    entries = [
        {
            "instance_id": "astropy__astropy-12907",
            "title": "Fix Astropy",
            "environment": {"image": "us-docker.pkg.dev/demo/astropy:latest"},
            "repo": "astropy/astropy",
        }
    ]

    result = convert_swebench_entries(entries, output_dir=tmp_dataset_dir, dataset_name="demo/swe", split="test")

    assert len(result.records) == 1
    record = result.records[0]
    record_path = Path(record["sandbox"]["r2e_ds_json"])
    assert record_path.is_absolute()
    assert record_path.is_relative_to(tmp_dataset_dir)
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    assert payload["docker_image"] == "us-docker.pkg.dev/demo/astropy:latest"
    assert record["sandbox"]["docker_image"] == "us-docker.pkg.dev/demo/astropy:latest"
    assert "requires_build" not in record["sandbox"]
    assert result.skipped == 0


def test_convert_swebench_entries_skips_missing_docker(
    tmp_dataset_dir: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch,
):
    monkeypatch.setattr(datasets_prepare, "_make_swebench_spec", lambda entry: None)
    entries = [
        {
            "instance_id": "missing",
            "title": "Broken entry",
        }
    ]

    caplog.set_level("WARNING")
    result = convert_swebench_entries(entries, output_dir=tmp_dataset_dir, dataset_name="demo/swe", split="test")

    assert result.records == []
    assert result.skipped == 1
    assert "docker image" in caplog.text


def test_convert_swebench_entries_uses_spec_when_available(tmp_dataset_dir: Path, monkeypatch):
    class DummySpec:
        instance_image_key = "sweb.eval.sympy-1"
        repo = "sympy/sympy"
        version = "v1"
        arch = "x86_64"
        repo_script_list = ["echo repo"]
        env_script_list = ["echo env"]
        eval_script_list = ["echo eval"]

    monkeypatch.setattr(datasets_prepare, "_make_swebench_spec", lambda entry: DummySpec())

    entries = [
        {
            "instance_id": "sympy__sympy-24066",
            "title": "Fix Sympy",
            "repo": "sympy/sympy",
        }
    ]

    result = convert_swebench_entries(entries, output_dir=tmp_dataset_dir, dataset_name="demo/swe", split="test")

    assert len(result.records) == 1
    record = result.records[0]
    sandbox = record["sandbox"]
    assert sandbox["docker_image"] == "sweb.eval.sympy-1"
    assert sandbox["requires_build"] is True
    spec_payload = sandbox["swebench_spec"]
    assert spec_payload["repo"] == "sympy/sympy"
    assert spec_payload["arch"] == "x86_64"
    record_path = Path(sandbox["r2e_ds_json"])
    assert record_path.is_absolute()
    assert record_path.is_relative_to(tmp_dataset_dir)
    ds_payload = json.loads(record_path.read_text(encoding="utf-8"))
    assert ds_payload["requires_build"] is True
    assert ds_payload["swebench_spec"]["repo_script_list"] == ["echo repo"]


def test_write_manifest_and_maybe_prepull(tmp_path: Path, monkeypatch):
    result = DatasetConversionResult(
        records=[
            {"sandbox": {"docker_image": "img:one"}},
            {"sandbox": {"docker_image": "img:two"}},
        ],
        instance_paths=[],
    )
    calls = []

    def fake_prepull(images, **kwargs):
        calls.append((list(images), kwargs))

    monkeypatch.setattr(prepare_datasets, "prepull_docker_images", fake_prepull)

    manifest = prepare_datasets._write_manifest_and_maybe_prepull(  # pylint: disable=protected-access
        output_dir=tmp_path,
        result=result,
        manifest_name="docker.txt",
        prepull=True,
        max_workers=4,
        retries=2,
        delay=1,
        pull_timeout=60,
    )

    assert manifest.read_text(encoding="utf-8").splitlines() == ["img:one", "img:two"]
    assert calls == [
        (["img:one", "img:two"], {"max_workers": 4, "retries": 2, "delay": 1, "pull_timeout": 60})
    ]


def test_write_manifest_handles_collections_without_build_only(tmp_path: Path, monkeypatch):
    class MinimalCollection:
        def __init__(self) -> None:
            self.images = ["demo:image"]
            self.missing = 0
            self.build_only = 0

    invoked: dict[str, bool] = {}

    def fake_collect(**_: object) -> MinimalCollection:
        return MinimalCollection()

    def fake_prepull(images, **__: object) -> None:
        invoked["called"] = True
        assert images == ["demo:image"]

    monkeypatch.setattr(prepare_datasets, "collect_docker_images", fake_collect)
    monkeypatch.setattr(prepare_datasets, "prepull_docker_images", fake_prepull)

    result = DatasetConversionResult(records=[], instance_paths=[], skipped=0)
    manifest = prepare_datasets._write_manifest_and_maybe_prepull(  # pylint: disable=protected-access
        output_dir=tmp_path,
        result=result,
        manifest_name="manifest.txt",
        prepull=True,
        max_workers=None,
        retries=None,
        delay=None,
        pull_timeout=None,
    )

    assert manifest.exists()
    assert invoked.get("called") is True


def test_prepare_swebench_downloads_from_hf(tmp_path: Path, monkeypatch):
    rows = [
        {
            "instance_id": "demo__repo-1",
            "title": "Fix bug",
            "docker_image": "demo/image:latest",
            "repo": "demo/repo",
        }
    ]

    def fake_load(dataset: str, split: str, token: Optional[str]):
        assert dataset == "princeton-nlp/SWE-bench_Verified"
        assert split == "validation"
        assert token is None
        return rows

    monkeypatch.setattr(prepare_datasets, "_load_dataset", fake_load)

    result = prepare_datasets._prepare_swebench(  # pylint: disable=protected-access
        output_dir=tmp_path / "out",
        dataset="princeton-nlp/SWE-bench_Verified",
        split="validation",
        token=None,
        limit=None,
    )

    assert len(result.records) == 1
    record = result.records[0]
    assert record["task_id"] == "demo__repo-1"
    ds_path = Path(record["sandbox"]["r2e_ds_json"])
    assert ds_path.is_absolute()
    assert ds_path.exists()
    assert ds_path.is_relative_to((tmp_path / "out").resolve())


def test_prepare_swebench_respects_limit(tmp_path: Path, monkeypatch):
    rows = [
        {
            "instance_id": f"demo__repo-{idx}",
            "title": "Fix issue",
            "docker_image": "demo/image:latest",
            "repo": "demo/repo",
        }
        for idx in range(3)
    ]

    monkeypatch.setattr(prepare_datasets, "_load_dataset", lambda *_: rows)

    result = prepare_datasets._prepare_swebench(  # pylint: disable=protected-access
        output_dir=tmp_path / "out",
        dataset="princeton-nlp/SWE-bench_Verified",
        split="validation",
        token="secret",
        limit=2,
    )

    assert len(result.records) == 2
    for record in result.records:
        assert record["task_id"].startswith("demo__repo-")

