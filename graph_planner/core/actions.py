# graph_planner/core/actions.py
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, RootModel, conint

# A) Explore（定位/阅读/扩展）
class ExploreAction(BaseModel):
    type: Literal["explore"] = "explore"
    op: Literal["find", "read", "expand"] = "find"
    anchors: List[Dict[str, Any]] = Field(default_factory=list)
    nodes: List[str] = Field(default_factory=list)
    hop: conint(ge=0, le=2) = 1
    limit: conint(ge=1, le=100) = 50
    schema_version: int = 1

# B) Memory（记忆维护，外部只给策略信号）
class MemoryAction(BaseModel):
    type: Literal["memory"] = "memory"
    target: Literal["explore", "observation"] = "explore"
    scope: Literal["turn", "session"] = "turn"
    intent: Literal["commit", "delete"] = "commit"
    selector: Optional[str] = None
    schema_version: int = 1

# C) Repair（是否打补丁；仅 apply=True 需要 plan）
class RepairAction(BaseModel):
    type: Literal["repair"] = "repair"
    apply: bool
    issue: Dict[str, Any]
    plan: Optional[str] = None  # 仅 apply=True 时需要，用于 Collater→CGM
    plan_targets: List[Dict[str, Any]] = Field(default_factory=list)
    patch: Optional[Dict[str, Any]] = None
    schema_version: int = 1

# D) Submit（终局评测）
class SubmitAction(BaseModel):
    type: Literal["submit"] = "submit"
    schema_version: int = 1

class NoopAction(BaseModel):
    type: Literal["noop"] = "noop"
    schema_version: int = 1


ActionUnion = Union[ExploreAction, MemoryAction, RepairAction, SubmitAction, NoopAction]


class _ActionSchema(RootModel[ActionUnion]):
    pass


def export_action_schema() -> Dict[str, Any]:
    """Return the serialisable schema for planner actions."""

    return _ActionSchema.model_json_schema()
