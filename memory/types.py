from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class Target:
    path: str
    start: int
    end: int
    confidence: float
    why: str = ""

@dataclass
class Plan:
    targets: List[Target]
    budget: Dict[str, Any] = field(default_factory=lambda: {"max_steps": 12, "max_lines_per_edit": 6, "lint_required": True})
    priority_tests: List[str] = field(default_factory=list)

@dataclass
class SubgraphNode:
    kind: str         # "HIT" | "STRUCT" | "TP" | "SEL"
    path: str
    line: int
    score: float
    window: Optional[List[int]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Subgraph:
    nodes: List[SubgraphNode] = field(default_factory=list)

@dataclass
class Feedback:
    diff_summary: str = ""
    lines_changed: List[int] = field(default_factory=list)
    top_assert: Optional[str] = None
    search_hits: List[Dict[str, Any]] = field(default_factory=list)
    tests_failed: int = 0
    first_failure_frame: Optional[Dict[str, Any]] = None
