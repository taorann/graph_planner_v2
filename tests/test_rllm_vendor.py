"""Tests for rLLM submodule path discovery helpers."""

from __future__ import annotations

import importlib
import sys

import pytest

from graph_planner.infra.vendor import ensure_rllm_importable


@pytest.fixture(autouse=True)
def _reset_cache():
    ensure_rllm_importable.cache_clear()
    original_path = list(sys.path)
    try:
        yield
    finally:
        ensure_rllm_importable.cache_clear()
        sys.modules.pop("rllm", None)
        sys.path[:] = original_path


def test_ensure_rllm_importable_env_hint(monkeypatch, tmp_path):
    fake_repo = tmp_path / "fake_rllm_repo"
    pkg = fake_repo / "rllm"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("FLAG = 'graph-planner'")

    monkeypatch.setenv("GRAPH_PLANNER_RLLM_PATH", str(fake_repo))
    monkeypatch.delenv("PYTHONPATH", raising=False)
    sys.modules.pop("rllm", None)

    assert ensure_rllm_importable() is True
    module = importlib.import_module("rllm")
    assert getattr(module, "FLAG") == "graph-planner"


def test_ensure_rllm_importable_src_layout(monkeypatch, tmp_path):
    fake_repo = tmp_path / "external" / "rllm"
    src = fake_repo / "src"
    pkg = src / "rllm"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("FLAG = 'src-layout'")

    monkeypatch.delenv("GRAPH_PLANNER_RLLM_PATH", raising=False)
    monkeypatch.setenv("GRAPH_PLANNER_RLLM_PATH", str(fake_repo))
    sys.modules.pop("rllm", None)

    assert ensure_rllm_importable() is True
    module = importlib.reload(importlib.import_module("rllm"))
    assert getattr(module, "FLAG") == "src-layout"
