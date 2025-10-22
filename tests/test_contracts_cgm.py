import pytest

from graph_planner.agents.common.contracts import ProtocolError, validate_cgm_patch


def test_validate_cgm_patch_success() -> None:
    obj = {
        "patch": {
            "edits": [
                {"path": "app.py", "start": 1, "end": 1, "new_text": "print('hi')\n"},
            ]
        },
        "summary": "add print",
    }
    patch = validate_cgm_patch(obj)
    assert patch.path == "app.py"
    assert patch.edits[0]["new_text"].endswith("\n")


def test_validate_cgm_patch_rejects_multi_file() -> None:
    obj = {
        "patch": {
            "edits": [
                {"path": "a.py", "start": 1, "end": 1, "new_text": "a\n"},
                {"path": "b.py", "start": 1, "end": 1, "new_text": "b\n"},
            ]
        }
    }
    with pytest.raises(ProtocolError) as excinfo:
        validate_cgm_patch(obj)
    assert excinfo.value.code == "multi-file-diff"


def test_validate_cgm_patch_requires_fields() -> None:
    with pytest.raises(ProtocolError) as excinfo:
        validate_cgm_patch({"patch": {"edits": [{}]}})
    assert excinfo.value.code == "path-missing"


def test_validate_cgm_patch_requires_newline() -> None:
    obj = {
        "patch": {
            "edits": [
                {"path": "app.py", "start": 1, "end": 1, "new_text": "print('hi')"},
            ]
        }
    }
    with pytest.raises(ProtocolError) as excinfo:
        validate_cgm_patch(obj)
    assert excinfo.value.code == "newline-missing"


def test_validate_cgm_patch_invalid_range() -> None:
    obj = {
        "patch": {
            "edits": [
                {"path": "app.py", "start": 5, "end": 2, "new_text": "print('hi')\n"},
            ]
        }
    }
    with pytest.raises(ProtocolError) as excinfo:
        validate_cgm_patch(obj)
    assert excinfo.value.code == "range-invalid"
