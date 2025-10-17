# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Step 3.2: Memory Ops Head (规则基线版)

输入：
  - candidates: List[Candidate]     # 3.1 产生的候选（含 score / reasons / from_anchor / kind / path / span / degree）
  - subgraph: SubgraphLike          # 当前工作子图（仅需最小读接口）
  - context: Optional[dict]         # 轻量上下文（可包含 budgets / policy 等）
输出：
  - List[MemOp]                     # {op, id, path?, kind?, confidence, reasons} 列表

策略（规则基线）：
  1) 对已在子图中的节点：
     - 若 span 变化（或缺失→出现）→ UPDATE
     - 若评分 >= keep_threshold → KEEP
     - 若评分 < delete_threshold 且非锚点/非 t-file → DELETE
     - 否则 → NOOP
  2) 对不在子图中的节点：
     - 采用目录多样性轮询 + 全局上限 → 选择 ADD 集
     - 需满足 score >= add_threshold；t-file 轻度加权优先
  3) 预算与多样性：
     - add_limit / delete_limit / update_limit 控制
     - dir_diversity_k 保障各目录覆盖

后续可无缝替换为 LoRA / 小头 / RL：
  - 只需保留 suggest() 的签名与返回结构。
"""

from dataclasses import dataclass
from collections import defaultdict, deque
from typing import List, Dict, Optional, Iterable, Tuple
from pathlib import PurePosixPath

from .types import (
    Candidate, SubgraphLike, MemOpLiteral,
    Span, Node
)

# ----------- 本模块内部使用的输出结构（与 memory_bank.apply_ops 对齐） -----------
class MemOp(dict):
    """
    统一的记忆维护操作建议：
      - op: MemOpLiteral = "KEEP"|"ADD"|"UPDATE"|"DELETE"|"NOOP"
      - id: 节点 id
      - path/kind/name/span:（可选）对齐 Node 字段，便于落库或更新
      - confidence: 0~1 置信度（规则映射）
      - reasons: 触发该操作的解释列表
    """
    def __init__(self,
                 op: MemOpLiteral,
                 node: Candidate | Node,
                 confidence: float,
                 reasons: List[str]) -> None:
        super().__init__(
            op=op,
            id=node.get("id"),
            path=node.get("path"),
            kind=(node.get("kind") or ""),
            name=node.get("name"),
            span=node.get("span"),
            confidence=float(max(0.0, min(1.0, confidence))),
            reasons=list(reasons)
        )

# ------------------ 可调参数（默认即可用；可接 infra/config） ------------------
@dataclass(frozen=True)
class DecisionThresholds:
    add_threshold: float = 1.10      # 分数达到才考虑 ADD
    keep_threshold: float = 0.80     # 低于此值不鼓励 KEEP
    delete_threshold: float = 0.30   # 远低于此值考虑 DELETE（且须非锚点、非 t-file）
    update_span_delta_ratio: float = 0.05  # span 差异阈值（相对旧窗口大小）

@dataclass(frozen=True)
class DecisionBudgets:
    add_limit: int = 40
    delete_limit: int = 20
    update_limit: int = 20
    keep_soft_limit: int = 100        # 仅用于裁剪输出体积；不硬控

@dataclass(frozen=True)
class DiversityPolicy:
    dir_diversity_k: int = 2
    prefer_test_files: bool = True

# ------------------------------ 工具函数 ------------------------------
def _dirname(path: Optional[str]) -> str:
    if not path:
        return ""
    try:
        return str(PurePosixPath(path).parent)
    except Exception:
        return ""

def _is_tfile(node_like: Dict) -> bool:
    kind = (node_like.get("kind") or "").lower()
    path = (node_like.get("path") or "").lower()
    if kind == "t-file":
        return True
    if not path:
        return False
    return ("test" in path) or ("/tests/" in path) or path.endswith("_test.py") or path.endswith("test.py")

def _span_len(span: Optional[Span]) -> int:
    if not span:
        return 0
    return max(0, int(span.get("end", 0)) - int(span.get("start", 0)) + 1)

def _span_changed(old: Optional[Span], new: Optional[Span], ratio_thresh: float) -> bool:
    if old == new:
        return False
    # 如果旧/新之一为空，但另一个存在，则认为需要 UPDATE
    if (old and not new) or (new and not old):
        return True
    # 都在：看相对差异
    olen = _span_len(old)
    nlen = _span_len(new)
    if olen == 0 and nlen == 0:
        return False
    base = max(1, max(olen, nlen))
    return abs(olen - nlen) / base >= ratio_thresh

# ------------------------------ 核心：建议生成 ------------------------------
def suggest(
    candidates: List[Candidate],
    context: Optional[dict],
    *,
    subgraph: SubgraphLike,
    thresholds: DecisionThresholds = DecisionThresholds(),
    budgets: DecisionBudgets = DecisionBudgets(),
    diversity: DiversityPolicy = DiversityPolicy(),
) -> List[MemOp]:
    """
    生成记忆维护操作建议（规则基线版）。

    参数
    ----
    candidates: 3.1 的候选（含 score / reasons / from_anchor / kind / path / span / degree）
    context: 轻量上下文，可包含：
        - "policy": {"prefer_test_files": bool}
        - "budgets": {"add_limit": int, "delete_limit": int, "update_limit": int}
        - "thresholds": {"add_threshold": float, "keep_threshold": float, "delete_threshold": float}
    subgraph: 当前工作子图（需支持 iter_node_ids / contains / get_node）

    返回
    ----
    List[MemOp]: 操作建议列表（包含 KEEP/ADD/UPDATE/DELETE/NOOP），按重要度排序
    """
    # ---- 从 context 覆盖默认参数（可选） ----
    if context:
        t_over = (context.get("thresholds") or {})
        thresholds = DecisionThresholds(
            add_threshold=float(t_over.get("add_threshold", thresholds.add_threshold)),
            keep_threshold=float(t_over.get("keep_threshold", thresholds.keep_threshold)),
            delete_threshold=float(t_over.get("delete_threshold", thresholds.delete_threshold)),
            update_span_delta_ratio=float(t_over.get("update_span_delta_ratio", thresholds.update_span_delta_ratio))
        )
        b_over = (context.get("budgets") or {})
        budgets = DecisionBudgets(
            add_limit=int(b_over.get("add_limit", budgets.add_limit)),
            delete_limit=int(b_over.get("delete_limit", budgets.delete_limit)),
            update_limit=int(b_over.get("update_limit", budgets.update_limit)),
            keep_soft_limit=int(b_over.get("keep_soft_limit", budgets.keep_soft_limit)),
        )
        p_over = (context.get("policy") or {})
        diversity = DiversityPolicy(
            dir_diversity_k=int(p_over.get("dir_diversity_k", diversity.dir_diversity_k)),
            prefer_test_files=bool(p_over.get("prefer_test_files", diversity.prefer_test_files))
        )

    # ---- 子图基线信息 ----
    in_sg_ids = set(subgraph.iter_node_ids())

    # ---- 先按分数降序，方便后续挑选 ----
    candidates_sorted = sorted(candidates, key=lambda c: float(c.get("score") or 0.0), reverse=True)

    # ---- 1) 先生成 UPDATE/KEEP/DELETE/NOOP 对“已在子图”的节点的建议 ----
    ops_existing: List[MemOp] = []
    num_update, num_delete, num_keep = 0, 0, 0

    # 建立一个快速索引：candidate by id（因为 existing 可能未被 candidates 覆盖完全）
    by_id: Dict[str, Candidate] = {c["id"]: c for c in candidates_sorted if c.get("id")}

    for nid in in_sg_ids:
        # 仅对出现在候选集中的 existing 节点给出积极建议；否则标记 NOOP 以便可观测
        cand = by_id.get(nid)
        node = cand or subgraph.get_node(nid)

        score = float((cand or {}).get("score") or 0.0)
        reasons = list((cand or {}).get("reasons") or [])
        from_anchor = bool((cand or {}).get("from_anchor", False))

        # span 变化 → UPDATE（限制 update_limit）
        old_node = subgraph.get_node(nid)
        if _span_changed(old_node.get("span"), (cand or {}).get("span"), thresholds.update_span_delta_ratio):
            if num_update < budgets.update_limit:
                ops_existing.append(MemOp("UPDATE", node, confidence=0.9, reasons=reasons + ["span_changed"]))
                num_update += 1
                continue  # UPDATE 优先级最高，对同一节点不再给其它建议

        # KEEP / DELETE / NOOP
        if score >= thresholds.keep_threshold:
            if num_keep < budgets.keep_soft_limit:
                ops_existing.append(MemOp("KEEP", node, confidence=min(0.8, 0.5 + 0.25 * score), reasons=reasons + ["keep_threshold"]))
                num_keep += 1
            else:
                # 超过 keep_soft_limit 时不输出多余 KEEP，避免日志过长
                pass
        elif (score < thresholds.delete_threshold) and (not from_anchor) and (not _is_tfile(node)):
            if num_delete < budgets.delete_limit:
                ops_existing.append(MemOp("DELETE", node, confidence=0.7, reasons=reasons + ["below_delete_threshold"]))
                num_delete += 1
            else:
                ops_existing.append(MemOp("NOOP", node, confidence=0.4, reasons=reasons + ["delete_budget_exhausted"]))
        else:
            ops_existing.append(MemOp("NOOP", node, confidence=0.5, reasons=reasons or ["noop_existing_low_signal"]))

    # ---- 2) 对“不在子图”的节点挑选 ADD 集（目录多样性 + 全局上限 + t-file 偏好） ----
    # 过滤：尚未在子图、且达到 add_threshold
    new_pool: List[Candidate] = []
    for c in candidates_sorted:
        cid = c.get("id")
        if not cid or cid in in_sg_ids:
            continue
        base_score = float(c.get("score") or 0.0)
        if base_score < thresholds.add_threshold:
            continue
        # t-file 轻度偏好：仅用于排序（不影响阈值）
        boost = 0.15 if (diversity.prefer_test_files and _is_tfile(c)) else 0.0
        c2 = dict(c)
        c2["__sel_score"] = base_score + boost
        new_pool.append(c2)  # type: ignore[assignment]

    # 目录分桶后，每目录先取 dir_diversity_k 个（按 __sel_score 降序），再全局补齐至 add_limit
    by_dir: Dict[str, List[Candidate]] = defaultdict(list)
    for c in sorted(new_pool, key=lambda x: float(x.get("__sel_score") or 0.0), reverse=True):
        by_dir[_dirname(c.get("path"))].append(c)

    selected_add: List[Candidate] = []
    rr = {d: deque(lst[:diversity.dir_diversity_k]) for d, lst in by_dir.items()}
    while rr and len(selected_add) < budgets.add_limit:
        for d in list(rr.keys()):
            if rr[d]:
                selected_add.append(rr[d].popleft())
                if len(selected_add) >= budgets.add_limit:
                    break
            else:
                rr.pop(d, None)

    if len(selected_add) < budgets.add_limit:
        picked_ids = {c["id"] for c in selected_add}
        rest = [c for c in sorted(new_pool, key=lambda x: float(x.get("__sel_score") or 0.0), reverse=True)
                if c["id"] not in picked_ids]
        need = budgets.add_limit - len(selected_add)
        selected_add.extend(rest[:need])

    ops_add = [
        MemOp("ADD", c, confidence=min(0.95, 0.6 + 0.2 * float(c.get("score") or 0.0)),
              reasons=list(c.get("reasons") or []) + ["add_selected"])
        for c in selected_add
    ]

    # ---- 3) 汇总与排序（UPDATE > ADD > KEEP > DELETE > NOOP） ----
    priority = {"UPDATE": 4, "ADD": 3, "KEEP": 2, "DELETE": 1, "NOOP": 0}
    all_ops: List[MemOp] = ops_existing + ops_add
    all_ops.sort(key=lambda op: (priority.get(op["op"], 0), float(op.get("confidence") or 0.0)), reverse=True)
    return all_ops
