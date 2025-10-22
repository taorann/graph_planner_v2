# 2025-10-22 memory hardening
"""Text-trajectory memory helpers for planner environments.

This module implements the lightweight memory layer described in the
text-trajectory protocol: it consumes planner ``<function=memory>`` blocks,
persists explore results and tool observations, and emits structured
``<observation>`` payloads for the next planner turn.  The implementation
keeps the behaviour intentionally small and testable so the environment can
control quota enforcement, deduplication, and persistence policy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Protocol, Tuple

from ..agents.common.contracts import ProtocolError, parse_action_block as _parse_action_block

__all__ = [
    "parse_action_block",
    "emit_observation",
    "GraphStore",
    "TextStore",
    "ApplyStats",
    "Size",
    "TurnState",
    "estimate_costs",
    "is_over_budget",
    "memory_commit",
    "memory_delete",
    "handle_memory",
    "WorkingGraphStore",
    "NoteTextStore",
]


_CHANNEL = "memory"


def parse_action_block(text: str, allowed: Iterable[str]) -> Dict[str, Any]:
    """Parse a planner block and enforce an allowed-action whitelist."""

    result = _parse_action_block(text)
    allowed_set = set(allowed or [])
    if allowed_set and result.get("name") not in allowed_set:
        raise ProtocolError("unknown-action", f"action '{result.get('name')}' not allowed")
    return result


def emit_observation(name: str, data: Mapping[str, Any]) -> str:
    """Serialize an observation dict into the protocol wire format."""

    body = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"<observation for=\"{name}\">{body}</observation>"


class GraphStore(Protocol):
    """Lightweight interface for persisting graph deltas."""

    def get(self, scope: str) -> Any:
        """Return the current graph snapshot for ``scope``."""

    def apply_delta(self, scope: str, delta: Mapping[str, Any]) -> "ApplyStats":
        """Apply a graph delta and return the net change statistics."""

    def revert_last(self, scope: str, tag: Optional[str] = None) -> "ApplyStats":
        """Rollback the latest delta (or the one identified by ``tag``)."""


class TextStore(Protocol):
    """Minimal interface for persisting textual memory notes."""

    def append(self, scope: str, note: str) -> int:
        """Append a note and return its identifier within ``scope``."""

    def remove(self, scope: str, selector: Optional[str] = None) -> int:
        """Remove the latest note or the one identified by ``selector``."""


@dataclass
class ApplyStats:
    """Statistics about graph mutations."""

    nodes: int = 0
    edges: int = 0
    tag: Optional[str] = None


@dataclass
class Size:
    """Approximate memory footprint estimates used for quota checks."""

    nodes: int = 0
    edges: int = 0
    frontier: int = 0
    planner_tokens_est: int = 0
    cgm_tokens_est: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "frontier": self.frontier,
            "planner_tokens_est": self.planner_tokens_est,
            "cgm_tokens_est": self.cgm_tokens_est,
        }


@dataclass
class TurnState:
    """State shared across memory actions within an episode."""

    graph_store: GraphStore
    text_store: TextStore
    latest_explore: Optional[Mapping[str, Any]] = None
    latest_observation: Optional[Mapping[str, Any]] = None
    size: Size = field(default_factory=Size)
    version: int = 0
    note_tokens: MutableMapping[Tuple[str, int], int] = field(default_factory=dict)
    frontier_history: List[Tuple[str, int]] = field(default_factory=list)

    def next_version(self) -> int:
        self.version += 1
        return self.version


def estimate_costs(state: TurnState) -> Size:
    """Return the current size estimates maintained inside ``state``."""

    return state.size


_CAP_KEY_MAP = {
    "nodes": "nodes",
    "edges": "edges",
    "frontier": "frontier",
    "planner_tokens": "planner_tokens_est",
    "planner_tokens_est": "planner_tokens_est",
    "cgm_tokens": "cgm_tokens_est",
    "cgm_tokens_est": "cgm_tokens_est",
}


def is_over_budget(size: Size, caps: Mapping[str, int]) -> Tuple[bool, List[str]]:
    """Return whether ``size`` exceeds the provided ``caps``."""

    exceeded: List[str] = []
    for key, limit in (caps or {}).items():
        attr = _CAP_KEY_MAP.get(key, key)
        value = getattr(size, attr, None)
        if value is None:
            continue
        if limit >= 0 and value > limit:
            exceeded.append(key)
    return (len(exceeded) > 0, exceeded)


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def _copy_dict(obj: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in obj.items()}


def _extract_explore_delta(obs: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    candidates = obs.get("candidates") or []
    if not isinstance(candidates, list):
        return [], []
    nodes: List[Dict[str, Any]] = []
    for cand in candidates:
        if not isinstance(cand, Mapping):
            continue
        node_id = str(cand.get("id") or "").strip()
        if not node_id:
            continue
        data = {
            "id": node_id,
            "path": cand.get("path"),
            "span": cand.get("span"),
            "score": cand.get("score"),
            "kind": cand.get("kind"),
            "summary": cand.get("summary"),
        }
        nodes.append({k: v for k, v in data.items() if v is not None})
    edges = []
    for edge in obs.get("edges", []) or []:
        if isinstance(edge, Mapping):
            edges.append(_copy_dict(edge))
    return nodes, edges


def _note_from_explore(obs: Mapping[str, Any]) -> str:
    candidates = obs.get("candidates") or []
    if not isinstance(candidates, list) or not candidates:
        return ""
    rows = []
    for cand in candidates[:3]:
        if not isinstance(cand, Mapping):
            continue
        path = cand.get("path") or cand.get("file") or "?"
        span = cand.get("span") or {}
        start = int(span.get("start") or 0) if isinstance(span, Mapping) else 0
        end = int(span.get("end") or 0) if isinstance(span, Mapping) else 0
        score = cand.get("score")
        rows.append(f"{path}:{start}-{end} (score={score})")
    return "Explore committed:\n" + "\n".join(rows)


def _note_from_observation(obs: Mapping[str, Any]) -> str:
    if not obs:
        return ""
    parts = [f"tool={obs.get('kind', 'unknown')}" ]
    if obs.get("applied") is not None:
        parts.append(f"applied={bool(obs.get('applied'))}")
    tests = obs.get("tests")
    if isinstance(tests, Mapping):
        parts.append(f"tests_passed={bool(tests.get('passed'))}")
    lint = obs.get("lint")
    if isinstance(lint, Mapping) and lint.get("rc") is not None:
        parts.append(f"lint_rc={lint.get('rc')}")
    msg = obs.get("msg") or obs.get("summary")
    if isinstance(msg, str) and msg.strip():
        parts.append(msg.strip())
    return " | ".join(parts)


def _project_size(base: Size, nodes: int, edges: int, note_tokens: int, frontier: int) -> Size:
    projected = Size(
        nodes=base.nodes + nodes,
        edges=base.edges + edges,
        frontier=max(frontier, base.frontier),
        planner_tokens_est=base.planner_tokens_est + note_tokens,
        cgm_tokens_est=max(base.cgm_tokens_est + note_tokens, base.planner_tokens_est + note_tokens),
    )
    return projected


def _caps_dict(caps: Mapping[str, int]) -> Dict[str, int]:
    return {str(k): int(v) for k, v in (caps or {}).items()}


def _clone_size(size: Size) -> Size:
    return Size(
        nodes=size.nodes,
        edges=size.edges,
        frontier=size.frontier,
        planner_tokens_est=size.planner_tokens_est,
        cgm_tokens_est=size.cgm_tokens_est,
    )


def _normalize_explore_selector(selector: Any) -> Optional[str]:
    if selector is None:
        return None
    if isinstance(selector, str):
        value = selector.strip()
        return value or None
    if isinstance(selector, Mapping):
        tag = selector.get("tag")
        if isinstance(tag, str):
            value = tag.strip()
            return value or None
    return None


def _normalize_observation_selector(selector: Any) -> Tuple[Optional[str], bool]:
    if selector is None:
        return (None, True)
    if isinstance(selector, str):
        value = selector.strip()
        if not value:
            return (None, True)
        if value.lower() == "latest":
            return ("latest", True)
        try:
            num = int(value)
        except ValueError:
            return (value, False)
        return (str(num), True)
    if isinstance(selector, Mapping):
        for key in ("id", "note_id"):
            if key in selector:
                return _normalize_observation_selector(selector[key])
        return (None, False)
    if isinstance(selector, (int, float)):
        return (str(int(selector)), True)
    return (None, False)


def memory_commit(
    state: TurnState,
    target: str,
    scope: str,
    selector: Optional[str],
    caps: Mapping[str, int],
) -> Dict[str, Any]:
    """Commit the latest explore or observation payload into memory."""

    base_size = _clone_size(state.size)
    selector = selector or "latest"

    if target == "explore":
        obs = state.latest_explore
        if not obs:
            return {
                "ok": False,
                "rejected": False,
                "error": "missing-explore",
                "msg": "no explore observation available",
            }
        nodes, edges = _extract_explore_delta(obs)
        if not nodes and not edges:
            return {
                "ok": False,
                "rejected": False,
                "error": "empty-explore",
                "msg": "latest explore observation has no candidates",
            }
        existing = state.graph_store.get(scope)
        existing_nodes = {}
        if hasattr(existing, "nodes"):
            raw_nodes = getattr(existing, "nodes")
            if isinstance(raw_nodes, Mapping):
                existing_nodes = raw_nodes
            elif isinstance(raw_nodes, list):
                existing_nodes = {n.get("id"): n for n in raw_nodes if isinstance(n, Mapping)}
        unique_nodes = [n for n in nodes if n.get("id") and n.get("id") not in existing_nodes]
        note = _note_from_explore(obs)
        note_tokens = _estimate_tokens(note)
        frontier_value = int(obs.get("frontier") or len(nodes))
        projected = _project_size(base_size, len(unique_nodes), len(edges), note_tokens, frontier_value)
        over, exceeded = is_over_budget(projected, caps)
        if over:
            return {
                "ok": False,
                "rejected": True,
                "error": "over-budget",
                "exceeded": exceeded,
                "limits": _caps_dict(caps),
                "size_before": base_size.to_dict(),
                "size_after": projected.to_dict(),
            }
        stats = state.graph_store.apply_delta(scope, {"nodes": nodes, "edges": edges, "frontier": frontier_value})
        tag = stats.tag
        text_chunks = 0
        if note:
            note_id = state.text_store.append(scope, note)
            state.note_tokens[(scope, note_id)] = note_tokens
            text_chunks = 1
            state.size.planner_tokens_est = base_size.planner_tokens_est + note_tokens
            state.size.cgm_tokens_est = max(base_size.cgm_tokens_est + note_tokens, state.size.planner_tokens_est)
        else:
            state.size.planner_tokens_est = base_size.planner_tokens_est
            state.size.cgm_tokens_est = base_size.cgm_tokens_est
        state.size.nodes = max(0, base_size.nodes + stats.nodes)
        state.size.edges = max(0, base_size.edges + stats.edges)
        state.size.frontier = frontier_value
        if tag:
            state.frontier_history.append((tag, frontier_value))
        version = state.next_version()
        return {
            "ok": True,
            "rejected": False,
            "target": "explore",
            "scope": scope,
            "intent": "commit",
            "applied": {
                "graph_nodes": stats.nodes,
                "graph_edges": stats.edges,
                "text_chunks": text_chunks,
            },
            "size": state.size.to_dict(),
            "selector": selector,
            "tag": tag,
            "version": version,
        }

    if target == "observation":
        obs = state.latest_observation
        if not obs:
            return {
                "ok": False,
                "rejected": False,
                "error": "missing-observation",
                "msg": "no observation available",
            }
        note = _note_from_observation(obs)
        if not note:
            return {
                "ok": False,
                "rejected": False,
                "error": "empty-observation",
                "msg": "latest observation has no content to persist",
            }
        note_tokens = _estimate_tokens(note)
        projected = _project_size(base_size, 0, 0, note_tokens, base_size.frontier)
        over, exceeded = is_over_budget(projected, caps)
        if over:
            return {
                "ok": False,
                "rejected": True,
                "error": "over-budget",
                "exceeded": exceeded,
                "limits": _caps_dict(caps),
                "size_before": base_size.to_dict(),
                "size_after": projected.to_dict(),
            }
        note_id = state.text_store.append(scope, note)
        state.note_tokens[(scope, note_id)] = note_tokens
        state.size.planner_tokens_est = base_size.planner_tokens_est + note_tokens
        state.size.cgm_tokens_est = max(base_size.cgm_tokens_est + note_tokens, state.size.planner_tokens_est)
        state.size.nodes = base_size.nodes
        state.size.edges = base_size.edges
        state.size.frontier = base_size.frontier
        version = state.next_version()
        return {
            "ok": True,
            "rejected": False,
            "target": "observation",
            "scope": scope,
            "intent": "commit",
            "applied": {
                "graph_nodes": 0,
                "graph_edges": 0,
                "text_chunks": 1,
            },
            "size": state.size.to_dict(),
            "selector": selector,
            "version": version,
        }

    return {
        "ok": False,
        "rejected": False,
        "error": "unsupported-target",
        "msg": f"unsupported memory target: {target}",
    }


def memory_delete(
    state: TurnState,
    target: str,
    scope: str,
    selector: Optional[str],
) -> Dict[str, Any]:
    """Delete the latest committed unit for ``target``."""

    selector = selector or "latest"
    base_size = _clone_size(state.size)
    if target == "explore":
        tag = selector if selector not in (None, "latest", "") else None
        stats = state.graph_store.revert_last(scope, tag=tag)
        if stats.nodes == 0 and stats.edges == 0:
            return {
                "ok": False,
                "rejected": False,
                "error": "nothing-to-delete",
                "msg": "no explore delta to delete",
            }
        state.size.nodes = max(0, base_size.nodes + stats.nodes)
        state.size.edges = max(0, base_size.edges + stats.edges)
        history = state.frontier_history
        idx: Optional[int] = None
        if tag:
            for i in range(len(history) - 1, -1, -1):
                if history[i][0] == tag:
                    idx = i
                    break
        if idx is None and history:
            idx = len(history) - 1
        if idx is not None and 0 <= idx < len(history):
            history.pop(idx)
        state.size.frontier = history[-1][1] if history else 0
        version = state.next_version()
        return {
            "ok": True,
            "rejected": False,
            "target": "explore",
            "scope": scope,
            "intent": "delete",
            "applied": {
                "graph_nodes": abs(stats.nodes),
                "graph_edges": abs(stats.edges),
                "text_chunks": 0,
            },
            "size": state.size.to_dict(),
            "selector": selector,
            "version": version,
        }

    if target == "observation":
        removed_id = state.text_store.remove(scope, selector)
        if removed_id < 0:
            return {
                "ok": False,
                "rejected": False,
                "error": "nothing-to-delete",
                "msg": "no observation note to delete",
            }
        tokens = state.note_tokens.pop((scope, removed_id), 0)
        state.size.planner_tokens_est = max(0, base_size.planner_tokens_est - tokens)
        state.size.cgm_tokens_est = max(state.size.planner_tokens_est, max(0, base_size.cgm_tokens_est - tokens))
        state.size.nodes = base_size.nodes
        state.size.edges = base_size.edges
        state.size.frontier = base_size.frontier
        version = state.next_version()
        return {
            "ok": True,
            "rejected": False,
            "target": "observation",
            "scope": scope,
            "intent": "delete",
            "applied": {
                "graph_nodes": 0,
                "graph_edges": 0,
                "text_chunks": 1 if tokens else 0,
            },
            "size": state.size.to_dict(),
            "selector": selector,
            "version": version,
        }

    return {
        "ok": False,
        "rejected": False,
        "error": "unsupported-target",
        "msg": f"unsupported memory target: {target}",
    }


def handle_memory(action_params: Mapping[str, Any], state: TurnState, caps: Mapping[str, int]) -> str:
    """Execute a memory action and return the serialized observation."""

    target = str(action_params.get("target") or "explore").lower()
    scope = str(action_params.get("scope") or "turn").lower()
    intent = str(action_params.get("intent") or "commit").lower()
    selector = action_params.get("selector")
    selector_value: Optional[str]
    if intent == "delete":
        if target == "explore":
            selector_value = _normalize_explore_selector(selector)
        elif target == "observation":
            selector_value, selector_ok = _normalize_observation_selector(selector)
            if not selector_ok:
                data = {
                    "ok": False,
                    "rejected": False,
                    "error": "nothing-to-delete",
                    "msg": "no observation note to delete",
                }
                return emit_observation(_CHANNEL, data)
        else:
            selector_value = None
    else:
        if isinstance(selector, str):
            selector_value = selector.strip() or None
        else:
            selector_value = None

    if intent == "commit":
        data = memory_commit(state, target, scope, selector_value, caps)
    elif intent == "delete":
        data = memory_delete(state, target, scope, selector_value)
    else:
        data = {
            "ok": False,
            "rejected": False,
            "error": "unsupported-intent",
            "msg": f"unsupported intent: {intent}",
        }
    return emit_observation(_CHANNEL, data)


# ---------------------------------------------------------------------------
# In-memory store implementations
# ---------------------------------------------------------------------------


@dataclass
class _GraphRecord:
    tag: str
    nodes: List[str]
    edges: List[Dict[str, Any]]
    frontier: int


class WorkingGraphStore(GraphStore):
    """GraphStore implementation backed by a WorkingSubgraph."""

    def __init__(self, subgraph: Any):
        from . import subgraph_store

        self.subgraph = subgraph_store.wrap(subgraph)
        self._history: MutableMapping[str, List[_GraphRecord]] = {}

    def get(self, scope: str) -> Any:
        return self.subgraph

    def apply_delta(self, scope: str, delta: Mapping[str, Any]) -> ApplyStats:
        nodes = [n for n in delta.get("nodes", []) if isinstance(n, Mapping)]
        edges = [e for e in delta.get("edges", []) if isinstance(e, Mapping)]
        scope_key = scope or "session"
        history = self._history.setdefault(scope_key, [])
        before_nodes = len(self.subgraph.nodes)
        before_edges = len(self.subgraph.edges)
        added_ids: List[str] = []
        for node in nodes:
            node_id = str(node.get("id") or "").strip()
            if not node_id:
                continue
            existing = self.subgraph.nodes.get(node_id)
            if existing:
                existing.update({k: v for k, v in node.items() if k != "id"})
            else:
                self.subgraph.nodes[node_id] = dict(node)
                self.subgraph.node_ids.add(node_id)
                added_ids.append(node_id)
        if edges:
            existing = {
                (
                    edge.get("src"),
                    edge.get("dst"),
                    edge.get("etype") or edge.get("type")
                )
                for edge in self.subgraph.edges
            }
            for edge in edges:
                key = (edge.get("src"), edge.get("dst"), edge.get("etype") or edge.get("type"))
                if key in existing:
                    continue
                existing.add(key)
                self.subgraph.edges.append(dict(edge))
        after_nodes = len(self.subgraph.nodes)
        after_edges = len(self.subgraph.edges)
        tag = f"{scope_key}:{len(history)+1}"
        history.append(
            _GraphRecord(
                tag=tag,
                nodes=added_ids,
                edges=[dict(e) for e in edges],
                frontier=int(delta.get("frontier") or len(nodes)),
            )
        )
        return ApplyStats(nodes=after_nodes - before_nodes, edges=after_edges - before_edges, tag=tag)

    def revert_last(self, scope: str, tag: Optional[str] = None) -> ApplyStats:
        scope_key = scope or "session"
        history = self._history.get(scope_key)
        if not history:
            return ApplyStats()
        record: Optional[_GraphRecord] = None
        if tag:
            for idx in range(len(history) - 1, -1, -1):
                if history[idx].tag == tag:
                    record = history.pop(idx)
                    break
        if record is None:
            record = history.pop() if history else None
        if record is None:
            return ApplyStats()
        removed_nodes = 0
        removed_edges = 0
        for node_id in record.nodes:
            if node_id in self.subgraph.nodes:
                removed_nodes += 1
                self.subgraph.nodes.pop(node_id, None)
                self.subgraph.node_ids.discard(node_id)
        if record.edges:
            for edge in record.edges:
                try:
                    self.subgraph.edges.remove(edge)
                    removed_edges += 1
                except ValueError:
                    continue
        return ApplyStats(nodes=-removed_nodes, edges=-removed_edges, tag=record.tag)


@dataclass
class _NoteRecord:
    note_id: int
    text: str


class NoteTextStore(TextStore):
    """Simple TextStore that keeps notes in memory."""

    def __init__(self) -> None:
        self._notes: MutableMapping[str, List[_NoteRecord]] = {}
        self._counter: MutableMapping[str, int] = {}

    def append(self, scope: str, note: str) -> int:
        scope_key = scope or "session"
        notes = self._notes.setdefault(scope_key, [])
        self._counter[scope_key] = self._counter.get(scope_key, 0) + 1
        note_id = self._counter[scope_key]
        notes.append(_NoteRecord(note_id=note_id, text=note))
        return note_id

    def remove(self, scope: str, selector: Optional[str] = None) -> int:
        scope_key = scope or "session"
        notes = self._notes.get(scope_key)
        if not notes:
            return -1
        target_id: Optional[int] = None
        if selector is not None:
            value = selector.strip() if isinstance(selector, str) else str(selector)
            value = value.strip()
            if not value:
                value = "latest"
            if value.lower() != "latest":
                try:
                    target_id = int(value)
                except ValueError:
                    return -1
        record: Optional[_NoteRecord] = None
        if target_id is not None:
            for idx in range(len(notes) - 1, -1, -1):
                if notes[idx].note_id == target_id:
                    record = notes.pop(idx)
                    break
        if record is None:
            record = notes.pop() if notes else None
        if record is None:
            return -1
        return record.note_id

