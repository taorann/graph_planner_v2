# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Shared typed structures for memory layer.
"""

from typing import TypedDict, Literal, List, Dict, Any, Optional, Protocol, Iterable

NodeKind = str

# ----------------------------
# Common literals / enums
# ----------------------------
MemOpLiteral = Literal["KEEP", "ADD", "UPDATE", "DELETE", "NOOP"]

# ----------------------------
# Core graph types
# ----------------------------
class Span(TypedDict, total=False):
    start: int           # 1-based
    end: int             # 1-based (inclusive)

class Node(TypedDict, total=False):
    id: str
    kind: NodeKind
    path: str            # repo-root relative path (for file-related nodes)
    name: str            # function/class/symbol name ('' for file nodes)
    span: Span           # optional for file (full file), required for func/class if known
    degree: int          # cached degree for convenience

class Edge(TypedDict):
    src: str             # node id
    dst: str             # node id
    etype: str           # e.g. "contains", "imports", "ref", "calls"

# 兼容更多锚点类型（function/symbol/file/class/module/t-file）
class Anchor(TypedDict, total=False):
    kind: Literal["function", "symbol", "file", "class", "module", "t-file"]
    text: Optional[str]
    id: Optional[str]

class DocChunk(TypedDict):
    path: str            # repo-root relative
    start: int
    end: int
    text: str

# ----------------------------
# Step 3 相关：候选与最小 Subgraph 协议
# ----------------------------
class Candidate(TypedDict, total=False):
    id: str
    kind: str
    path: Optional[str]
    name: Optional[str]
    span: Optional[Span]
    degree: int
    from_anchor: bool
    score: float
    reasons: List[str]

class SubgraphLike(Protocol):
    """最小读接口：供候选构造与记忆维护流程使用。"""
    def iter_node_ids(self) -> Iterable[str]: ...
    def contains(self, node_id: str) -> bool: ...
    def get_node(self, node_id: str) -> Dict[str, Any]: ...
