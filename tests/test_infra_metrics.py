"""Tests for telemetry helpers in ``graph_planner.infra.metrics``."""

from __future__ import annotations

import sys
import types

from graph_planner.infra import metrics


def test_make_ray_snapshot_handles_missing_runtime(monkeypatch):
    """If Ray is importable but not initialised, we should return an empty payload."""

    dummy_module = types.ModuleType("ray")

    class DummySystemError(Exception):
        pass

    def _raise():
        raise DummySystemError("ray not started")

    dummy_module.available_resources = _raise
    dummy_module.exceptions = types.SimpleNamespace(RaySystemError=DummySystemError)

    monkeypatch.setitem(sys.modules, "ray", dummy_module)

    snapshot = metrics.make_ray_snapshot()
    assert snapshot() == {}
