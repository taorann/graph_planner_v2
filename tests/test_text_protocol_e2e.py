import json
from pathlib import Path
from typing import Dict

import pytest

from graph_planner.agents.common import text_protocol
from graph_planner.agents.common.chat import FALLBACK_REASON_KEY
from graph_planner.agents.model_based.planner import LocalLLMPlannerAgent, _AgentState
from graph_planner.core.actions import SubmitAction
from graph_planner.memory.subgraph_store import WorkingSubgraph


class FakeSandbox:
    def __init__(self, repo_root: Path, *, tests_ok: bool = True, lint_ok: bool = True) -> None:
        self.repo_root = repo_root
        self.tests_ok = tests_ok
        self.lint_ok = lint_ok
        self.lint_invocations: list[Path] = []
        self.test_invocations: list[Path] = []

    def lint(self, temp_dir: Path | None = None) -> Dict[str, object]:
        target = temp_dir or self.repo_root
        self.lint_invocations.append(target)
        return {"ok": self.lint_ok, "stdout": "lint" if self.lint_ok else "lint failed"}

    def test(self, temp_dir: Path | None = None) -> Dict[str, object]:
        target = temp_dir or self.repo_root
        self.test_invocations.append(target)
        content = target.joinpath("app.py").read_text(encoding="utf-8")
        passed = self.tests_ok and "patched" in content
        return {"passed": passed, "stdout": "ok" if passed else "fail"}

    def reset_soft(self) -> None:  # pragma: no cover - compatibility stub
        pass


def _make_repo(tmp_path: Path, text: str) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    repo.joinpath("app.py").write_text(text, encoding="utf-8")
    return repo


def _make_state(tmp_path: Path, sandbox: FakeSandbox, candidate: dict) -> text_protocol.RepairRuntimeState:
    payloads: list[dict] = []

    def _generator(payload: dict, k: int) -> list[dict]:
        payloads.append(payload)
        return [candidate]

    subgraph = WorkingSubgraph(nodes={"n1": {"id": "n1", "path": "app.py", "span": {"start": 1, "end": 1}}}, edges=[])
    repo_root = str(sandbox.repo_root)
    related_files = {"app.py": sandbox.repo_root.joinpath("app.py").read_text(encoding="utf-8")}
    state = text_protocol.RepairRuntimeState(
        issue={"title": "Bug", "body": "Fix"},
        subgraph=subgraph,
        sandbox=sandbox,
        repo_root=repo_root,
        text_memory={"session_summary": "summary", "turn_notes": "notes"},
        related_files=related_files,
        default_focus_ids=["n1"],
        cgm_generate=_generator,
    )
    state._payloads = payloads  # type: ignore[attr-defined]
    return state


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    return _make_repo(tmp_path, "print('hello')\n")


def test_handle_planner_repair_success(tmp_path: Path, repo: Path) -> None:
    candidate = {
        "patch": {"edits": [{"path": "app.py", "start": 1, "end": 1, "new_text": "print('patched')\n"}]},
        "summary": "reason",
        "confidence": 0.8,
        "tests": [],
    }
    sandbox = FakeSandbox(repo)
    state = _make_state(tmp_path, sandbox, candidate)
    raw = (
        "<function=repair>\n"
        "  <param name=\"thought\">fix</param>\n"
        "  <param name=\"subplan\"><![CDATA[1) patch]]></param>\n"
        "  <param name=\"focus_ids\">[\"n1\"]</param>\n"
        "</function>"
    )
    result = text_protocol.handle_planner_repair(raw, state)
    assert result["ok"] is True
    assert result["applied"] is True
    assert Path(repo, "app.py").read_text(encoding="utf-8") == "print('patched')\n"
    assert result["n_hunks"] == 1 and result["added_lines"] >= 1
    assert "patch_id" in result and result["temp_path"]
    assert sandbox.test_invocations and sandbox.lint_invocations


def test_handle_planner_repair_apply_false_returns_candidate(tmp_path: Path, repo: Path) -> None:
    candidate = {
        "patch": {"edits": [{"path": "app.py", "start": 1, "end": 1, "new_text": "print('patched')\n"}]},
        "summary": "reason",
        "confidence": 0.3,
        "tests": [],
    }
    sandbox = FakeSandbox(repo)
    state = _make_state(tmp_path, sandbox, candidate)
    raw = (
        "<function=repair>\n"
        "  <param name=\"thought\">fix</param>\n"
        "  <param name=\"subplan\"><![CDATA[1) patch]]></param>\n"
        "  <param name=\"focus_ids\">[\"n1\"]</param>\n"
        "  <param name=\"apply\">false</param>\n"
        "</function>"
    )
    result = text_protocol.handle_planner_repair(raw, state)
    assert result["ok"] is True
    assert result["applied"] is False
    assert result["candidate"]["patch"]["edits"][0]["path"] == "app.py"
    assert Path(repo, "app.py").read_text(encoding="utf-8") == "print('hello')\n"


def test_handle_planner_repair_reports_invalid_patch(tmp_path: Path, repo: Path) -> None:
    candidate = {
        "patch": {"edits": [{"path": "missing.py", "start": 1, "end": 1, "new_text": "print('patched')\n"}]},
        "summary": "reason",
        "confidence": 0.5,
        "tests": [],
    }
    sandbox = FakeSandbox(repo)
    state = _make_state(tmp_path, sandbox, candidate)
    raw = (
        "<function=repair>\n"
        "  <param name=\"thought\">fix</param>\n"
        "  <param name=\"subplan\"><![CDATA[1) patch]]></param>\n"
        "  <param name=\"focus_ids\">[\"n1\"]</param>\n"
        "</function>"
    )
    result = text_protocol.handle_planner_repair(raw, state)
    assert result["ok"] is False
    assert result["error"] in {"path-missing", "invalid-unified-diff"}


class DummyRuleAgent:
    def step(self, obs: dict) -> dict:
        return {"action_obj": SubmitAction(), "plan": "fallback"}


def test_agent_fallback_records_protocol_error_code() -> None:
    agent = LocalLLMPlannerAgent.__new__(LocalLLMPlannerAgent)
    agent._rule_agent = DummyRuleAgent()
    agent.state = _AgentState(issue={})
    agent._messages = [{"role": "system", "content": ""}]
    thought, action, assistant_msg, meta = agent._parse_model_response("not a block", {"issue": {}})
    assert meta["used_fallback"] is True
    assert meta[FALLBACK_REASON_KEY] == "missing-function-tag"
    assert isinstance(action, SubmitAction)
