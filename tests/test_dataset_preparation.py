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
from scripts import prepare_swebench_validation, prepare_training_datasets


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
    rel_path = Path(record["sandbox"]["r2e_ds_json"])
    assert (tmp_dataset_dir / rel_path).is_file()
    with (tmp_dataset_dir / rel_path).open("r", encoding="utf-8") as handle:
        saved = json.load(handle)
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
    rel_path = Path(result.records[0]["sandbox"]["r2e_ds_json"])
    assert rel_path.suffix == ".json"


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
        rel_path = Path(record["sandbox"]["r2e_ds_json"])
        assert rel_path.is_relative_to(Path("."))
        assert rel_path.name.endswith(".json")


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
    rel_path = Path(record["sandbox"]["r2e_ds_json"])
    ds_file = tmp_dataset_dir / rel_path
    assert ds_file.exists()
    payload = json.loads(ds_file.read_text(encoding="utf-8"))
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
    ds_file = tmp_dataset_dir / record["sandbox"]["r2e_ds_json"]
    payload = json.loads(ds_file.read_text(encoding="utf-8"))
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
    ds_payload = json.loads((tmp_dataset_dir / sandbox["r2e_ds_json"]).read_text(encoding="utf-8"))
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

    monkeypatch.setattr(prepare_training_datasets, "prepull_docker_images", fake_prepull)

    manifest = prepare_training_datasets._write_manifest_and_maybe_prepull(  # pylint: disable=protected-access
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

    invoked: dict[str, bool] = {}

    def fake_collect(**_: object) -> MinimalCollection:
        return MinimalCollection()

    def fake_prepull(images, **__: object) -> None:
        invoked["called"] = True
        assert images == ["demo:image"]

    monkeypatch.setattr(prepare_training_datasets, "collect_docker_images", fake_collect)
    monkeypatch.setattr(prepare_training_datasets, "prepull_docker_images", fake_prepull)

    result = DatasetConversionResult(records=[], instance_paths=[], skipped=0)
    manifest = prepare_training_datasets._write_manifest_and_maybe_prepull(  # pylint: disable=protected-access
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


def test_prepare_swebench_validation_prefers_local(tmp_path: Path, monkeypatch):
    swe_root = tmp_path / "SWE-bench"
    data_dir = swe_root / "data" / "verified"
    data_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "instance_id": "demo__repo-1",
        "title": "Fix bug",
        "docker_image": "demo/image:latest",
        "repo": "demo/repo",
    }
    (data_dir / "validation.jsonl").write_text(json.dumps(payload) + "\n", encoding="utf-8")

    invoked = {"hf": False}

    def fail_hf(*_: object, **__: object) -> tuple[Iterable[Mapping[str, Any]], str]:
        invoked["hf"] = True
        return [], "validation"

    monkeypatch.setattr(prepare_swebench_validation, "_load_hf_dataset", fail_hf)

    result = prepare_swebench_validation._prepare_swebench(  # pylint: disable=protected-access
        output_dir=tmp_path / "out",
        dataset="princeton-nlp/SWE-bench_Verified",
        split="validation",
        token=None,
        limit=None,
        dataset_path=swe_root,
    )

    assert invoked["hf"] is False
    assert len(result.records) == 1
    record = result.records[0]
    assert record["task_id"] == "demo__repo-1"
    ds_path = Path(record["sandbox"]["r2e_ds_json"])
    assert (tmp_path / "out" / ds_path).exists()


def test_prepare_swebench_validation_falls_back_to_test_split(tmp_path: Path, monkeypatch):
    payload = {
        "instance_id": "demo__repo-2",
        "title": "Fix issue",
        "docker_image": "demo/image:latest",
        "repo": "demo/repo",
    }

    def fake_local(_: Path, __: str) -> list[Mapping[str, Any]]:
        return []

    def fake_hf(name: str, split: str, token: Optional[str]):
        assert split == "validation"
        return iter([payload]), "test"

    monkeypatch.setattr(prepare_swebench_validation, "_load_local_swebench", fake_local)
    monkeypatch.setattr(prepare_swebench_validation, "_load_hf_dataset", fake_hf)

    result = prepare_swebench_validation._prepare_swebench(  # pylint: disable=protected-access
        output_dir=tmp_path / "out",
        dataset="princeton-nlp/SWE-bench_Verified",
        split="validation",
        token=None,
        limit=None,
        dataset_path=None,
    )

    assert len(result.records) == 1
    record = result.records[0]
    assert record["split"] == "test"
    assert (tmp_path / "out" / "test.jsonl").exists()
