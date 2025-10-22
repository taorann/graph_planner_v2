from pathlib import Path

from graph_planner.agents.common import text_protocol
from graph_planner.memory.subgraph_store import WorkingSubgraph


class FailingSandbox:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root

    def lint(self, temp_dir: Path | None = None) -> dict:
        return {"ok": True}

    def test(self, temp_dir: Path | None = None) -> dict:
        return {"passed": False, "stdout": "tests failed"}

    def reset_soft(self) -> None:  # pragma: no cover - compatibility
        pass


def _make_repo(tmp_path: Path, text: str = "print('hello')\n") -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    repo.joinpath("app.py").write_text(text, encoding="utf-8")
    return repo


def _make_state(repo: Path) -> text_protocol.RepairRuntimeState:
    subgraph = WorkingSubgraph(nodes={"n1": {"id": "n1", "path": "app.py", "span": {"start": 1, "end": 1}}}, edges=[])
    related = {"app.py": repo.joinpath("app.py").read_text(encoding="utf-8")}
    sandbox = FailingSandbox(repo)
    state = text_protocol.RepairRuntimeState(
        issue={"title": "Bug"},
        subgraph=subgraph,
        sandbox=sandbox,
        repo_root=str(repo),
        text_memory={},
        related_files=related,
        default_focus_ids=["n1"],
        cgm_generate=lambda payload, k: [
            {
                "patch": {"edits": [{"path": "app.py", "start": 1, "end": 1, "new_text": "print('patched')\n"}]},
                "confidence": 0.4,
            }
        ],
    )
    return state


def test_handle_repair_failure_includes_telemetry(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    state = _make_state(repo)
    action = (
        "<function=repair>\n"
        "  <param name=\"subplan\"><![CDATA[1) patch]]></param>\n"
        "  <param name=\"focus_ids\">[\"n1\"]</param>\n"
        "</function>"
    )
    result = text_protocol.handle_planner_repair(action, state)
    assert result["ok"] is False
    assert result["error"] == "build-failed"
    assert result["fallback_reason"] == "build-failed"
    assert result["patch_id"]
    assert result["n_hunks"] >= 1
    assert result.get("temp_path")
