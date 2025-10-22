import importlib
from typing import Dict

import pytest

from aci.schema import Plan, PlanTarget


@pytest.fixture(autouse=True)
def reset_config_env(monkeypatch):
    """Ensure CGM-related environment variables do not leak across tests."""
    keys = [
        "CGM_ENABLED",
        "CGM_ENDPOINT",
        "CGM_MODEL",
        "CGM_TEMPERATURE",
        "CGM_MAX_TOKENS",
        "CGM_TIMEOUT_S",
        "CGM_API_KEY_ENV",
        "CGM_API_KEY",
        "CGM_MODEL_PATH",
        "CGM_TOKENIZER_PATH",
        "CGM_DEVICE",
        "CGM_MAX_INPUT_TOKENS",
        "CGM_TOP_P",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    yield


def _make_plan() -> Plan:
    return Plan(targets=[PlanTarget(path="foo.py", start=1, end=1, id="n1")])


def _make_snippet() -> Dict[str, object]:
    return {
        "path": "foo.py",
        "start": 1,
        "end": 1,
        "snippet": ["0001: print('hello')"],
    }


def test_generate_local_fallback(monkeypatch):
    module = importlib.import_module("graph_planner.agents.rule_based.cgm_adapter")
    module._CLIENT_CACHE = None
    module._CLIENT_FINGERPRINT = None

    patch = module.generate(
        subgraph_linearized=None,
        plan=_make_plan(),
        constraints={"max_edits": 1},
        snippets=[_make_snippet()],
        plan_text="plan",
        issue={"id": "ISSUE-1"},
    )

    assert patch["edits"], "local fallback should create marker edit"
    assert patch["edits"][0]["new_text"].strip().endswith("CGM-LOCAL")


def test_generate_remote_invocation(monkeypatch):
    monkeypatch.setenv("CGM_ENABLED", "1")
    monkeypatch.setenv("CGM_ENDPOINT", "http://example.com/cgm")
    monkeypatch.setenv("CGM_MODEL", "codefuse-cgm")

    module = importlib.import_module("graph_planner.agents.rule_based.cgm_adapter")
    importlib.reload(module)

    from graph_planner.integrations import codefuse_cgm as cgm_pkg

    captured = {}

    def fake_post(self, payload):
        captured["payload"] = payload
        return {
            "patch": {
                "edits": [
                    {"path": "foo.py", "start": 1, "end": 1, "new_text": "print('ok')\n"}
                ],
                "summary": "codefuse-remote",
            }
        }

    monkeypatch.setattr(cgm_pkg.CodeFuseCGMClient, "_post_json", fake_post)

    module._CLIENT_CACHE = None
    module._CLIENT_FINGERPRINT = None

    patch = module.generate(
        subgraph_linearized=[{"id": "n1"}],
        plan=_make_plan(),
        constraints={"max_edits": 2},
        snippets=[_make_snippet()],
        plan_text="Plan text",
        issue={"id": "ISSUE-2"},
    )

    assert patch["summary"] == "codefuse-remote"
    assert patch["edits"][0]["new_text"].endswith("\n")
    assert captured["payload"]["issue"]["id"] == "ISSUE-2"
    assert captured["payload"]["constraints"] == {"max_edits": 2}
    assert captured["payload"]["model_config"]["model"] == "codefuse-cgm"


def test_generate_with_local_runtime(monkeypatch):
    module = importlib.import_module("graph_planner.agents.rule_based.cgm_adapter")
    importlib.reload(module)

    class DummyRuntime:
        def __init__(self):
            config = type("Cfg", (), {"model_name_or_path": "dummy"})
            self.generator = type("Gen", (), {"config": config})()

        def generate_patch(self, **kwargs):
            return {
                "edits": [
                    {"path": "foo.py", "start": 1, "end": 1, "new_text": "print('ok')\n"}
                ],
                "summary": "local-runtime",
            }

    monkeypatch.setattr(module, "_get_client", lambda: None)
    monkeypatch.setattr(module, "_get_local_runtime", lambda: DummyRuntime())

    patch = module.generate(
        subgraph_linearized=[{"path": "foo.py", "start": 1, "end": 1, "text": "print('hello')"}],
        plan=_make_plan(),
        constraints={"max_edits": 1},
        snippets=[_make_snippet()],
        plan_text="Plan text",
        issue={"id": "ISSUE-3"},
    )

    assert patch["summary"] == "local-runtime"
    assert patch["edits"][0]["new_text"].endswith("\n")
