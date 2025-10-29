"""Facade over the graph planner collate helper for actor workers."""

from __future__ import annotations

from typing import Any, Tuple

from graph_planner.agents.rule_based.collater import collate as _collate


def collate(subgraph: Any, plan: Any, cfg: Any) -> Tuple[Any, Any]:
    """Proxy to :func:`graph_planner.agents.rule_based.collater.collate`."""

    return _collate(subgraph=subgraph, plan=plan, cfg=cfg)


__all__ = ["collate"]
