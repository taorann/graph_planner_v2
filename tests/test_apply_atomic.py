from pathlib import Path

import pytest

from graph_planner.runtime.sandbox import PatchApplier
from graph_planner.agents.common.contracts import ProtocolError


def _make_repo(tmp_path: Path, text: str = "print('hello')\n") -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    repo.joinpath("app.py").write_text(text, encoding="utf-8")
    return repo


def _diff() -> str:
    return (
        "--- a/app.py\n"
        "+++ b/app.py\n"
        "@@ -1 +1 @@\n"
        "-print('hello')\n"
        "+print('patched')\n"
    )


def test_patch_applier_rolls_back_on_failed_tests(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    applier = PatchApplier()

    def run_tests(temp_dir: Path):
        return {"passed": False, "stdout": "fail"}

    def run_lint(temp_dir: Path):
        return {"ok": True}

    with pytest.raises(ProtocolError) as exc:
        applier.apply_in_temp_then_commit(
            repo,
            _diff(),
            "app.py",
            run_tests,
            run_lint,
            patch_id="pid-1",
            new_content="print('patched')\n",
            stats={"n_hunks": 1, "added_lines": 1, "removed_lines": 1},
        )
    assert exc.value.code == "build-failed"
    assert repo.joinpath("app.py").read_text(encoding="utf-8") == "print('hello')\n"
    assert getattr(exc.value, "temp_path", "")


def test_patch_applier_commits_after_success(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    applier = PatchApplier()

    def run_tests(temp_dir: Path):
        return {"passed": True, "stdout": "ok"}

    def run_lint(temp_dir: Path):
        return {"ok": True}

    result = applier.apply_in_temp_then_commit(
        repo,
        _diff(),
        "app.py",
        run_tests,
        run_lint,
        patch_id="pid-2",
        new_content="print('patched')\n",
        stats={"n_hunks": 1, "added_lines": 1, "removed_lines": 1},
    )
    assert result["ok"] is True
    assert result["n_hunks"] == 1
    assert repo.joinpath("app.py").read_text(encoding="utf-8") == "print('patched')\n"
    assert result["temp_path"]


def test_patch_applier_detects_duplicate(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    applier = PatchApplier()

    def run_tests(temp_dir: Path):
        return {"passed": True}

    def run_lint(temp_dir: Path):
        return {"ok": True}

    applier.apply_in_temp_then_commit(
        repo,
        _diff(),
        "app.py",
        run_tests,
        run_lint,
        patch_id="dup",
        new_content="print('patched')\n",
        stats={"n_hunks": 1, "added_lines": 1, "removed_lines": 1},
    )

    with pytest.raises(ProtocolError) as exc:
        applier.apply_in_temp_then_commit(
            repo,
            _diff(),
            "app.py",
            run_tests,
            run_lint,
            patch_id="dup",
            new_content="print('patched')\n",
            stats={"n_hunks": 1, "added_lines": 1, "removed_lines": 1},
        )
    assert exc.value.code == "duplicate-patch"
