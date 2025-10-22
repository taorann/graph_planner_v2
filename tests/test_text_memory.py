from __future__ import annotations

import json

import pytest

from graph_planner.agents.common.contracts import ProtocolError
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
    assert state.size.frontier == 1
    assert state.frontier_history and state.frontier_history[-1][0] == commit["tag"]

    delete = text_memory.memory_delete(state, "explore", "session", commit["tag"])
    assert delete["ok"] is True
    assert delete["applied"]["graph_nodes"] == 1
    assert state.size.nodes == 0
    assert state.size.frontier == 0
    assert not state.frontier_history


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


def test_handle_memory_delete_explore_selector_dict() -> None:
    subgraph = subgraph_store.new()
    state = text_memory.TurnState(
        graph_store=text_memory.WorkingGraphStore(subgraph),
        text_store=text_memory.NoteTextStore(),
    )
    state.latest_explore = {
        "kind": "explore",
        "op": "expand",
        "candidates": [
            {"id": "n2", "path": "pkg/foo.py", "span": {"start": 1, "end": 2}},
        ],
    }
    commit_obs = text_memory.handle_memory(
        {"target": "explore", "scope": "session", "intent": "commit"},
        state,
        _caps(),
    )
    payload = json.loads(commit_obs.split(">", 1)[1].rsplit("<", 1)[0])
    tag = payload["tag"]

    delete_obs = text_memory.handle_memory(
        {
            "target": "explore",
            "scope": "session",
            "intent": "delete",
            "selector": {"tag": tag},
        },
        state,
        _caps(),
    )
    delete_payload = json.loads(delete_obs.split(">", 1)[1].rsplit("<", 1)[0])
    assert delete_payload["ok"] is True
    assert delete_payload["applied"]["graph_nodes"] == 1
    assert state.size.nodes == 0
    assert state.size.frontier == 0


def test_memory_delete_observation_selector_variants() -> None:
    subgraph = subgraph_store.new()
    state = text_memory.TurnState(
        graph_store=text_memory.WorkingGraphStore(subgraph),
        text_store=text_memory.NoteTextStore(),
    )
    state.latest_observation = {"kind": "repair", "applied": True, "tests": {"passed": True}}
    commit = text_memory.memory_commit(state, "observation", "session", None, _caps())
    assert commit["ok"] is True
    note_ids = [note_id for (_, note_id) in state.note_tokens.keys()]
    assert note_ids
    note_id = note_ids[0]

    # delete by whitespace padded string
    delete = text_memory.memory_delete(state, "observation", "session", "  %03d  " % note_id)
    assert delete["ok"] is True

    # commit again to test dict selector and invalid handling
    state.latest_observation = {"kind": "lint", "applied": False, "msg": "warn"}
    text_memory.memory_commit(state, "observation", "session", None, _caps())
    delete_dict = text_memory.handle_memory(
        {
            "target": "observation",
            "scope": "session",
            "intent": "delete",
            "selector": {"id": note_id + 1},
        },
        state,
        _caps(),
    )
    payload = json.loads(delete_dict.split(">", 1)[1].rsplit("<", 1)[0])
    assert payload["ok"] is True

    bad_delete = text_memory.handle_memory(
        {
            "target": "observation",
            "scope": "session",
            "intent": "delete",
            "selector": {"id": "bad"},
        },
        state,
        _caps(),
    )
    bad_payload = json.loads(bad_delete.split(">", 1)[1].rsplit("<", 1)[0])
    assert bad_payload["ok"] is False
    assert bad_payload["error"] == "nothing-to-delete"


def test_parse_action_block_rejects_disallowed_action() -> None:
    with pytest.raises(ProtocolError):
        text_memory.parse_action_block(
            "<function=memory></function>",
            allowed={"repair"},
        )
