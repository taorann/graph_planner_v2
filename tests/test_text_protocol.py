"""Unit tests for the text trajectory repair helpers."""

from __future__ import annotations

import pytest

from graph_planner.agents.common import text_protocol
from graph_planner.memory.subgraph_store import WorkingSubgraph


class FakeSandbox:
    def __init__(self, *, apply_ok: bool = True, tests_ok: bool = True) -> None:
        self.apply_ok = apply_ok
        self.tests_ok = tests_ok
        self.applied: list[str] = []
        self.reset_calls = 0
        self.lint_calls = 0
        self.test_calls = 0

    def apply_patch(self, patch: str) -> bool:
        self.applied.append(patch)
        return self.apply_ok

    def lint(self) -> bool:
        self.lint_calls += 1
        return True

    def test(self) -> dict:
        self.test_calls += 1
        return {"passed": self.tests_ok, "stdout": "ok" if self.tests_ok else "fail"}

    def reset_soft(self) -> None:
        self.reset_calls += 1


def _make_state(sandbox: FakeSandbox, *, payloads: list[dict], candidate: dict) -> text_protocol.RepairRuntimeState:
    def _generator(payload: dict, k: int) -> list[dict]:
        payloads.append(payload)
        return [candidate]

    subgraph = WorkingSubgraph(nodes={"n1": {"id": "n1", "path": "app.py", "span": {"start": 1, "end": 1}}}, edges=[])
    state = text_protocol.RepairRuntimeState(
        issue={"title": "Bug", "body": "Fix"},
        subgraph=subgraph,
        sandbox=sandbox,
        repo_root=".",
        text_memory={"session_summary": "summary", "turn_notes": "notes"},
        related_files={"app.py": "print('hello')\n"},
        default_focus_ids=["n1"],
        cgm_generate=_generator,
    )
    return state


def test_parse_action_block_success() -> None:
    text = (
        "<function=repair>\n"
        "  <param name=\"subplan\"><![CDATA[1) fix it\n2) add test]]></param>\n"
        "  <param name=\"focus_ids\">[\"n1\"]</param>\n"
        "  <param name=\"apply\">true</param>\n"
        "</function>"
    )
    parsed = text_protocol.parse_action_block(text, {"repair"})
    assert parsed["name"] == "repair"
    assert parsed["params"]["subplan"].strip().startswith("1) fix")
    assert parsed["params"]["focus_ids"] == ["n1"]
    assert parsed["params"]["apply"] is True


def test_parse_action_block_rejects_extra_text() -> None:
    with pytest.raises(text_protocol.ActionParseError):
        text_protocol.parse_action_block("hello <function=noop></function>", {"noop"})


def test_parse_action_block_allows_noop_without_params() -> None:
    parsed = text_protocol.parse_action_block("<function=noop></function>", {"noop"})
    assert parsed == {"name": "noop", "params": {}}


def test_handle_planner_repair_apply_true_success() -> None:
    candidate = {
        "patch": "--- a/app.py\n+++ b/app.py\n@@\n-print('hello')\n+print('patched')\n",
        "path": "app.py",
        "confidence": 0.7,
        "rationale": "reason",
        "tests": [],
    }
    sandbox = FakeSandbox()
    payloads: list[dict] = []
    state = _make_state(sandbox, payloads=payloads, candidate=candidate)
    result = text_protocol.handle_planner_repair(
        {"name": "repair", "params": {"subplan": "1) change", "focus_ids": ["n1"], "apply": True}},
        state,
    )
    assert result["ok"] is True
    assert result["applied"] is True
    assert result["hunks"] == 1
    assert sandbox.applied == [candidate["patch"]]
    assert payloads and payloads[0]["plan"] == ["change"]


def test_handle_planner_repair_apply_false_returns_candidate() -> None:
    candidate = {
        "patch": "--- a/app.py\n+++ b/app.py\n@@\n-print('hello')\n+print('patched')\n",
        "path": "app.py",
        "confidence": 0.3,
        "rationale": "reason",
        "tests": [],
    }
    sandbox = FakeSandbox()
    payloads: list[dict] = []
    state = _make_state(sandbox, payloads=payloads, candidate=candidate)
    result = text_protocol.handle_planner_repair(
        {"name": "repair", "params": {"subplan": "1) change", "focus_ids": ["n1"], "apply": False}},
        state,
    )
    assert result["ok"] is True
    assert result["applied"] is False
    assert sandbox.applied == []
    assert result["candidate"]["patch"].startswith("--- a/app.py")


def test_handle_planner_repair_invalid_diff_triggers_error() -> None:
    candidate = {
        "patch": "--- a/other.py\n+++ b/other.py\n@@\n-print('x')\n+print('y')\n",
        "path": "app.py",
        "confidence": 0.2,
    }
    sandbox = FakeSandbox()
    payloads: list[dict] = []
    state = _make_state(sandbox, payloads=payloads, candidate=candidate)
    result = text_protocol.handle_planner_repair(
        {"name": "repair", "params": {"subplan": "1) change", "focus_ids": ["n1"], "apply": True}},
        state,
    )
    assert result["ok"] is False
    assert result["error"] == "invalid-diff"
    assert sandbox.applied == []


def test_emit_observation_wraps_json() -> None:
    out = text_protocol.emit_observation("repair", {"ok": True})
    assert out == '<observation for="repair">{"ok": true}</observation>'
