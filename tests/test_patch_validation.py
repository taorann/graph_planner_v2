from pathlib import Path

import pytest

from graph_planner.agents.common import text_protocol
from graph_planner.agents.common.contracts import ProtocolError
from graph_planner.memory.subgraph_store import WorkingSubgraph


class DummySandbox:
    def lint(self, temp_dir: Path | None = None):  # pragma: no cover - not used
        return {"ok": True}

    def test(self, temp_dir: Path | None = None):  # pragma: no cover - not used
        return {"passed": True}


def _make_state(tmp_path: Path, content: str = "print('hello')\n") -> text_protocol.RepairRuntimeState:
    repo = tmp_path / "repo"
    repo.mkdir()
    repo.joinpath("app.py").write_text(content, encoding="utf-8")
    subgraph = WorkingSubgraph(nodes={}, edges=[])
    state = text_protocol.RepairRuntimeState(
        issue={},
        subgraph=subgraph,
        sandbox=DummySandbox(),
        repo_root=str(repo),
        text_memory={},
        related_files={"app.py": content},
        default_focus_ids=[],
        cgm_generate=lambda payload, k: [],
    )
    return state


def test_validate_unified_diff_accepts_valid_patch(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    diff = (
        "--- a/app.py\n"
        "+++ b/app.py\n"
        "@@ -1 +1 @@\n"
        "-print('hello')\n"
        "+print('patched')\n"
    )
    with text_protocol._bind_state(state):  # type: ignore[attr-defined]
        analysis = text_protocol.validate_unified_diff(diff, "app.py")
    assert analysis.n_hunks == 1
    assert analysis.added_lines == 1
    assert analysis.removed_lines == 1
    assert analysis.new_text.endswith("\n")


def test_validate_unified_diff_rejects_multifile(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    diff = (
        "--- a/app.py\n"
        "+++ b/other.py\n"
        "@@ -1 +1 @@\n"
        "-print('hello')\n"
        "+print('patched')\n"
    )
    with text_protocol._bind_state(state):  # type: ignore[attr-defined]
        with pytest.raises(ProtocolError) as exc:
            text_protocol.validate_unified_diff(diff, "app.py")
    assert exc.value.code == "multi-file-diff"


def test_validate_unified_diff_rejects_out_of_range(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    diff = (
        "--- a/app.py\n"
        "+++ b/app.py\n"
        "@@ -10 +10 @@\n"
        "-print('hello')\n"
        "+print('patched')\n"
    )
    with text_protocol._bind_state(state):  # type: ignore[attr-defined]
        with pytest.raises(ProtocolError) as exc:
            text_protocol.validate_unified_diff(diff, "app.py")
    assert exc.value.code == "range-invalid"


def test_validate_unified_diff_rejects_hunk_mismatch(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    diff = (
        "--- a/app.py\n"
        "+++ b/app.py\n"
        "@@ -1 +1 @@\n"
        "-print('different')\n"
        "+print('patched')\n"
    )
    with text_protocol._bind_state(state):  # type: ignore[attr-defined]
        with pytest.raises(ProtocolError) as exc:
            text_protocol.validate_unified_diff(diff, "app.py")
    assert exc.value.code == "hunk-mismatch"


def test_validate_unified_diff_normalises_crlf(tmp_path: Path) -> None:
    state = _make_state(tmp_path, "print('hello')\r\n")
    diff = (
        "--- a/app.py\r\n"
        "+++ b/app.py\r\n"
        "@@ -1 +1 @@\r\n"
        "-print('hello')\r\n"
        "+print('patched')\r\n"
    )
    with text_protocol._bind_state(state):  # type: ignore[attr-defined]
        analysis = text_protocol.validate_unified_diff(diff, "app.py")
    assert analysis.added_lines == 1
