# -*- coding: utf-8 -*-
"""
MemCandidates builder for Step 3.1

从现有子图与锚点出发做 1-hop 扩展，生成用于记忆操作决策的候选集：
- 去重、过滤已在子图中的节点
- 计算 explainable 的 score 与 reasons
- 施加目录多样性与配额约束
- 输出按 score 降序的 candidates 列表（TypedDict）

设计要点：
- 采用“无向邻接”的 1-hop（graph_adapter 内已封装）
- 同文件/同目录启发式加分，t-file 适度放大
- 目录多样性（round-robin）避免单一路径淹没
- 与 RepoGraph 的仓库级导航思想、Memory-R1 的结构化记忆操作兼容
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterable, List, Dict, Set, Tuple

from .types import Anchor, Candidate, Node, SubgraphLike
from . import graph_adapter


# --------------------------
# 可调权重（默认足够保守）
# --------------------------
@dataclass(frozen=True)
class CandidateScoringWeights:
    w_from_anchor: float = 1.0
    w_degree: float = 0.6
    w_same_file: float = 0.8
    w_same_dir: float = 0.3
    w_test_file: float = 0.25
    w_novelty: float = 0.5  # 不在子图中的适度加分


@dataclass(frozen=True)
class CandidateSelectionBudget:
    max_per_anchor: int = 50          # 每个锚点最多保留的 1-hop 节点
    total_limit: int = 200            # 全局候选上限（进入 3.2 决策的数量）
    dir_diversity_k: int = 3          # 每个目录至少保留的 top-k（round-robin 选取）
    prefer_test_files: bool = True    # 轻度偏好 t-file（单测上下文）


def _is_test_file(node: Node) -> bool:
    kind = (node.get("kind") or "").lower()
    path = (node.get("path") or "").lower()
    if kind == "t-file":
        return True
    return "test" in path or "/tests/" in path or path.endswith("_test.py") or path.endswith("test.py")


def _dirname(node: Node) -> str:
    path = node.get("path") or ""
    try:
        return str(PurePosixPath(path).parent)
    except Exception:
        return ""


def _filename(node: Node) -> str:
    return (node.get("path") or "").split("/")[-1]


def _score_node(
    node: Node,
    anchor_paths: Set[str],
    in_subgraph_ids: Set[str],
    weights: CandidateScoringWeights,
    from_anchor_flag: bool,
) -> Tuple[float, List[str]]:
    reasons: List[str] = []
    score = 0.0

    # 来源标记
    if from_anchor_flag:
        score += weights.w_from_anchor
        reasons.append("from_anchor")

    # 度（若 adapter 已给出）
    degree = int(node.get("degree") or 0)
    if degree > 0:
        score += weights.w_degree * min(degree / 8.0, 1.0)  # 归一化到 [0,1]
        reasons.append(f"degree={degree}")

    # 同文件
    path = node.get("path") or ""
    if path and path in anchor_paths:
        score += weights.w_same_file
        reasons.append("same_file")

    # 同目录
    if path:
        dirnames = {p.rsplit("/", 1)[0] for p in anchor_paths if "/" in p}
        my_dir = path.rsplit("/", 1)[0] if "/" in path else ""
        if my_dir and my_dir in dirnames:
            score += weights.w_same_dir
            reasons.append("same_dir")

    # t-file
    if _is_test_file(node):
        score += weights.w_test_file
        reasons.append("t_file")

    # 新颖性（不在现有子图）
    if node.get("id") not in in_subgraph_ids:
        score += weights.w_novelty
        reasons.append("novel")

    return score, reasons


def _to_candidate(node: Node, score: float, reasons: List[str], from_anchor: bool) -> Candidate:
    return {
        "id": node.get("id"),
        "kind": (node.get("kind") or "").lower(),
        "path": node.get("path"),
        "span": node.get("span"),
        "degree": int(node.get("degree") or 0),
        "from_anchor": bool(from_anchor),
        "score": float(score),
        "reasons": reasons,
        "name": node.get("name"),
    }


def build_mem_candidates(
    subgraph: SubgraphLike,
    anchors: Iterable[Anchor | Node],
    *,
    max_nodes_per_anchor: int = 50,
    total_limit: int = 200,
    dir_diversity_k: int = 3,
    weights: CandidateScoringWeights | None = None,
) -> List[Candidate]:
    """
    主入口：生成用于 3.2 决策头的候选列表（按 score 降序）。

    参数
    ----
    subgraph: SubgraphLike
        当前工作子图（需支持 `iter_node_ids()` / `contains(node_id)` / `get_node(node_id)`）
    anchors: Iterable[Anchor | Node]
        锚点（可为 Anchor 或已解析的 Node）
    max_nodes_per_anchor: int
        每个锚点最多保留的 1-hop 候选
    total_limit: int
        返回的候选全局上限
    dir_diversity_k: int
        目录多样性：同一目录将以 round-robin 方式至少保留 top-k
    weights: CandidateScoringWeights
        打分权重；不传则使用默认

    返回
    ----
    List[Candidate]
        候选节点列表（含 explainable 的 reasons）
    """
    weights = weights or CandidateScoringWeights()

    # 1) 解析锚点对应的路径集合（用于 same_file/same_dir）
    anchor_nodes: List[Node] = []
    for a in anchors:
        if isinstance(a, dict) and "id" in a:  # Node
            anchor_nodes.append(a)  # type: ignore
        else:
            # Anchor ←→ Node 解析由 adapter 负责
            resolved = graph_adapter.find_nodes_by_anchor(a)  # type: ignore
            anchor_nodes.extend(resolved)
    anchor_paths: Set[str] = {n.get("path") for n in anchor_nodes if n.get("path")}

    # 2) 获取现有子图节点 id 集（避免重复）
    in_subgraph_ids: Set[str] = set(subgraph.iter_node_ids())  # type: ignore

    # 3) 针对每个锚点做 1-hop 扩展并初步打分
    raw_bucket: Dict[str, Candidate] = {}
    for an in anchor_nodes:
        # one_hop_expand: 使用“无向邻接”，内部包含正/反向边 + 同文件/同目录启发式
        neighbors: List[Node] = graph_adapter.one_hop_expand(
            subgraph=subgraph,
            anchors=[an],
            max_nodes=max_nodes_per_anchor,
        )

        for nb in neighbors:
            nid = nb.get("id")
            if not nid:
                continue
            # 过滤：保持唯一
            already = raw_bucket.get(nid)
            score, reasons = _score_node(
                nb,
                anchor_paths=anchor_paths,
                in_subgraph_ids=in_subgraph_ids,
                weights=weights,
                from_anchor_flag=True,
            )
            cand = _to_candidate(nb, score, reasons, from_anchor=True)
            if (already is None) or (cand["score"] > already["score"]):
                raw_bucket[nid] = cand

    # 4) 目录多样性与全局配额
    # 先按分数降序分桶，再用 round-robin 每个目录取前 k，之后补齐到 total_limit
    by_dir: Dict[str, List[Candidate]] = defaultdict(list)
    for c in sorted(raw_bucket.values(), key=lambda x: x["score"], reverse=True):
        by_dir[_dirname(c)].append(c)

    # 每目录保底 k 的 round-robin
    selected: List[Candidate] = []
    queues = {d: deque(lst[:dir_diversity_k]) for d, lst in by_dir.items()}
    while queues and len(selected) < total_limit:
        for d in list(queues.keys()):
            if queues[d]:
                selected.append(queues[d].popleft())
                if len(selected) >= total_limit:
                    break
            else:
                queues.pop(d, None)

    # 若仍未满额，按分数全局补齐
    if len(selected) < total_limit:
        picked_ids = {c["id"] for c in selected}
        rest = [c for c in sorted(raw_bucket.values(), key=lambda x: x["score"], reverse=True)
                if c["id"] not in picked_ids]
        need = total_limit - len(selected)
        selected.extend(rest[:need])

    # 5) 最终排序与输出
    selected.sort(key=lambda x: x["score"], reverse=True)
    return selected
