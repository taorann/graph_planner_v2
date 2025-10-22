#!/usr/bin/env python3
"""Validate CGM patches against the local repository."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from graph_planner.agents.common import text_protocol
from graph_planner.agents.common.contracts import ProtocolError, validate_cgm_patch
from graph_planner.memory.subgraph_store import WorkingSubgraph


class _CLISandbox:
    def lint(self, temp_dir: Path | None = None) -> dict[str, Any]:  # pragma: no cover - trivial
        return {"ok": True}

    def test(self, temp_dir: Path | None = None) -> dict[str, Any]:  # pragma: no cover - trivial
        return {"passed": True}

    def reset_soft(self) -> None:  # pragma: no cover - trivial
        pass


def _load_patch(payload: str) -> dict[str, Any]:
    candidate_path = Path(payload)
    if candidate_path.exists():
        return json.loads(candidate_path.read_text(encoding="utf-8"))
    return json.loads(payload)


def _build_state(target: Path, patch_obj: dict[str, Any]) -> text_protocol.RepairRuntimeState:
    repo_root = target.parent
    related = {target.name: target.read_text(encoding="utf-8")}
    subgraph = WorkingSubgraph(nodes={}, edges=[])
    state = text_protocol.RepairRuntimeState(
        issue={},
        subgraph=subgraph,
        sandbox=_CLISandbox(),
        repo_root=str(repo_root),
        text_memory={},
        related_files=related,
        default_focus_ids=[],
        cgm_generate=lambda payload, k: [patch_obj],
    )
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a CGM patch against a file")
    parser.add_argument("--file", required=True, help="Path to the file being patched")
    parser.add_argument("--json", required=True, help="Patch candidate JSON or path to a JSON file")
    args = parser.parse_args()

    target = Path(args.file).resolve()
    if not target.exists():
        print(f"ERROR path-missing: target file '{target}' not found")
        return 1

    try:
        candidate = _load_patch(args.json)
        patch = validate_cgm_patch(candidate)
    except ProtocolError as exc:
        print(f"ERROR {exc.code}: {exc.detail}")
        return 1
    except Exception as exc:  # pragma: no cover - defensive
        print(f"ERROR invalid-json: {exc}")
        return 1

    state = _build_state(target, candidate)
    try:
        diff_text, analysis, _ = text_protocol._prepare_patch(state, patch)  # type: ignore[attr-defined]
        with text_protocol._bind_state(state):  # type: ignore[attr-defined]
            text_protocol.validate_unified_diff(diff_text, patch.path)
    except ProtocolError as exc:
        print(f"ERROR {exc.code}: {exc.detail}")
        return 1

    print(
        "OK",
        json.dumps(
            {
                "path": patch.path,
                "n_hunks": analysis.n_hunks,
                "added": analysis.added_lines,
                "removed": analysis.removed_lines,
            }
        ),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
