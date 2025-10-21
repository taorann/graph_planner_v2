from __future__ import annotations

from graph_planner.memory import text_memory
from graph_planner.memory import subgraph_store


def _caps() -> dict[str, int]:
    return {
        "nodes": 5,
        "edges": 20,
        "planner_tokens": 2000,
        "cgm_tokens": 16000,
        "frontier": 10,
    }


def test_memory_commit_and_delete_explore() -> None:
    subgraph = subgraph_store.new()
    state = text_memory.TurnState(
        graph_store=text_memory.WorkingGraphStore(subgraph),
        text_store=text_memory.NoteTextStore(),
    )
    state.latest_explore = {
        "kind": "explore",
        "op": "expand",
        "candidates": [
            {
                "id": "n1",
                "path": "pkg/module.py",
                "span": {"start": 10, "end": 20},
                "score": 0.9,
            }
        ],
    }

    commit = text_memory.memory_commit(state, "explore", "session", None, _caps())
    assert commit["ok"] is True
    assert commit["applied"]["graph_nodes"] == 1
    assert state.size.nodes == 1

    delete = text_memory.memory_delete(state, "explore", "session", None)
    assert delete["ok"] is True
    assert delete["applied"]["graph_nodes"] == 1
    assert state.size.nodes == 0


def test_memory_commit_observation_reject_missing() -> None:
    subgraph = subgraph_store.new()
    state = text_memory.TurnState(
        graph_store=text_memory.WorkingGraphStore(subgraph),
        text_store=text_memory.NoteTextStore(),
    )
    result = text_memory.memory_commit(state, "observation", "session", None, _caps())
    assert result["ok"] is False
    assert result["error"] == "missing-observation"


def test_emit_observation() -> None:
    payload = {"ok": True, "target": "explore"}
    rendered = text_memory.emit_observation("memory", payload)
    assert rendered == '<observation for="memory">{"ok":true,"target":"explore"}</observation>'
