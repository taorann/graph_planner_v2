from pathlib import Path

import json

import pytest

from graph_planner.runtime import containers


def test_collect_docker_images_from_multiple_sources(tmp_path: Path):
    records = [
        {"sandbox": {"docker_image": "img:one"}},
        {"sandbox": {"docker_image": "img:two"}},
        {"sandbox": {"docker_image": "local:build", "requires_build": True}},
        {"sandbox": {}},
    ]
    instance = tmp_path / "instances"
    instance.mkdir()
    file_a = instance / "a.json"
    file_a.write_text(json.dumps({"docker_image": "img:three"}), encoding="utf-8")
    file_b = instance / "b.json"
    file_b.write_text("{}", encoding="utf-8")
    jsonl = tmp_path / "tasks.jsonl"
    jsonl.write_text(
        "\n".join(
            [
                json.dumps({"sandbox": {"docker_image": "img:two"}}),
                json.dumps({"sandbox": {"docker_image": "img:four"}}),
                "malformed",
            ]
        ),
        encoding="utf-8",
    )

    collection = containers.collect_docker_images(
        records=records,
        instance_paths=[file_a, file_b],
        jsonl_paths=[jsonl],
    )

    assert collection.images == ["img:one", "img:two", "img:three", "img:four"]
    assert collection.build_only == 1
    assert collection.missing >= 2  # missing sandbox + malformed JSON
    assert collection.inspected >= 6


def test_manifest_roundtrip(tmp_path: Path):
    images = ["img:one", "img:two"]
    manifest = containers.write_docker_manifest(tmp_path / "manifest.txt", images)
    assert manifest.exists()
    assert containers.load_docker_manifest(manifest) == images


def test_prepull_docker_images_invokes_helper(monkeypatch):
    calls = []

    def fake_helper(images, **kwargs):
        calls.append((tuple(images), kwargs))

    monkeypatch.setattr(containers, "_resolve_pre_pull", lambda: fake_helper)

    result = containers.prepull_docker_images(
        ["img:one", "img:two", "img:one"],
        max_workers=4,
        retries=2,
        delay=3,
        pull_timeout=120,
    )

    assert result.invoked is True
    assert result.images == ["img:one", "img:two"]
    assert calls == [
        (
            ("img:one", "img:two"),
            {"max_workers": 4, "retries": 2, "delay": 3, "pull_timeout": 120},
        )
    ]


def test_prepull_docker_images_no_images(monkeypatch):
    monkeypatch.setattr(containers, "_resolve_pre_pull", lambda: pytest.fail("should not resolve helper"))
    result = containers.prepull_docker_images([])
    assert result.invoked is False
    assert result.images == []
