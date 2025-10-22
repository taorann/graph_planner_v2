# graph_planner/core/actions.py
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any, Union

# A) Explore（定位/阅读/扩展）
class ExploreAction(BaseModel):
    type: Literal["explore"] = "explore"
    op: Literal["find", "read", "expand"] = "find"
    anchors: List[Dict[str, Any]] = Field(default_factory=list)
    nodes: List[str] = Field(default_factory=list)
    hop: int = 1
    limit: int = 50

# B) Memory（记忆维护，外部只给策略信号）
class MemoryAction(BaseModel):
    type: Literal["memory"] = "memory"
    target: Literal["explore", "observation"] = "explore"
    scope: Literal["turn", "session"] = "turn"
    intent: Literal["commit", "delete"] = "commit"
    selector: Optional[str] = None

# C) Repair（是否打补丁；仅 apply=True 需要 plan）
class RepairAction(BaseModel):
    type: Literal["repair"] = "repair"
    apply: bool
    issue: Dict[str, Any]
    plan: Optional[str] = None  # 仅 apply=True 时需要，用于 Collater→CGM
    plan_targets: List[Dict[str, Any]] = Field(default_factory=list)
    patch: Optional[Dict[str, Any]] = None

# D) Submit（终局评测）
class SubmitAction(BaseModel):
    type: Literal["submit"] = "submit"

class NoopAction(BaseModel):
    type: Literal["noop"] = "noop"


ActionUnion = Union[ExploreAction, MemoryAction, RepairAction, SubmitAction, NoopAction]
