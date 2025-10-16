# graph_planner/core/actions.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

# A) Explore（定位/阅读/扩展）
@dataclass
class ExploreAction:
    type: str = field(init=False, default="explore")
    op: str = "find"
    anchors: List[Dict[str, Any]] = field(default_factory=list)
    nodes: List[str] = field(default_factory=list)
    hop: int = 1
    limit: int = 50

# B) Memory（记忆维护，外部只给策略信号）
@dataclass
class MemoryAction:
    type: str = field(init=False, default="memory")
    ops: List[Dict[str, Any]] = field(default_factory=list)
    budget: int = 30
    diversify_by_dir: int = 3

# C) Repair（是否打补丁；仅 apply=True 需要 plan）
@dataclass
class RepairAction:
    type: str = field(init=False, default="repair")
    apply: bool = False
    issue: Dict[str, Any] = field(default_factory=dict)
    plan: Optional[str] = None  # 仅 apply=True 时需要，用于 Collater→CGM
    plan_targets: List[Dict[str, Any]] = field(default_factory=list)
    patch: Optional[Dict[str, Any]] = None

# D) Submit（终局评测）
@dataclass
class SubmitAction:
    type: str = field(init=False, default="submit")

ActionUnion = Union[ExploreAction, MemoryAction, RepairAction, SubmitAction]
