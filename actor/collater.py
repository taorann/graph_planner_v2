# -*- coding: utf-8 -*-
from __future__ import annotations
"""
actor/collater.py

Step 4.1：Collater（可运行，占位策略）
- collate(subgraph, plan, cfg) -> (chunks: List[DocChunk], meta: CollateMeta)

功能：
1) 读取 3.4 的线性化结果（WSD/FULL）
2) 依据 PlanTarget 提升相关片段权重
3) 估算 token 成本，按预算选择片段
4) 在超预算时，若 cfg.collate.enable_light_reorder=True，则进行“轻量重排”再选
5) 支持测试片段交错 (interleave_tests)
6) 遵守 per_file_max_chunks 限制

注意：
- 这里不做复杂的窗口裁剪；WSD 的窗口由 subgraph_store.linearize 决定
- “轻量重排”仅改变挑选顺序，并不会修改片段内容
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

from aci.schema import CollateMeta
from memory.types import DocChunk
from memory import subgraph_store


# ---------------- utils ----------------

def _is_tfile_path(path: str) -> bool:
    p = (path or "").lower()
    return ("test" in p) or ("/tests/" in p) or p.endswith("_test.py") or p.endswith("test.py")

def _overlap(a1: int, a2: int, b1: int, b2: int) -> bool:
    return max(a1, b1) <= min(a2, b2)

def _lines(ck: DocChunk) -> int:
    try:
        return int(ck["end"]) - int(ck["start"]) + 1
    except Exception:
        return 0

def _est_tokens(ck: DocChunk) -> int:
    # 文本可用时按字符/4估算；缺文本时按行*60 估算（与 3.4 保持一致）
    if ck.get("text") is not None:
        return int(len(ck.get("text") or "") / 4)
    return max(1, _lines(ck) * 60)

def _summarize(chunks: List[DocChunk]) -> CollateMeta:
    n = len(chunks)
    total_lines = 0
    max_lines = 0
    tfile_chunks = 0
    tokens = 0
    for ck in chunks:
        ln = _lines(ck)
        total_lines += ln
        max_lines = max(max_lines, ln)
        tokens += _est_tokens(ck)
        if _is_tfile_path(ck["path"]):
            tfile_chunks += 1

    avg = total_lines / max(1, n)
    return {
        "chunks": n,
        "total_lines": total_lines,
        "avg_lines": round(avg, 2),
        "max_lines": max_lines,
        "tfile_chunk_ratio": round(tfile_chunks / max(1, n), 4),
        "est_tokens": tokens,
        "reordered": False,
        "warnings": [],
    }


# ---------------- 打分 ----------------

@dataclass
class _ScoreCtx:
    plan_targets: List[Dict[str, Any]]
    file_degree: Dict[str, float]
    prefer_tests: bool

def _build_file_degree_index(subgraph) -> Dict[str, float]:
    """
    基于子图节点的 degree，近似出每个文件的“重要度”。用于加权排序。
    """
    sg = subgraph_store.wrap(subgraph)
    sums: Dict[str, int] = defaultdict(int)
    cnts: Dict[str, int] = defaultdict(int)
    for n in sg.nodes.values():
        p = n.get("path")
        if not p:
            continue
        d = int(n.get("degree") or 0)
        sums[p] += d
        cnts[p] += 1
    out: Dict[str, float] = {}
    for p, s in sums.items():
        out[p] = float(s) / max(1, cnts[p])
    return out

def _score_chunk(ck: DocChunk, ctx: _ScoreCtx) -> float:
    """
    启发式权重：
      +1.6  片段与 plan target 文件相同且窗口重叠
      +1.0  片段与 plan target 文件相同（无重叠）
      +0.6  非测试文件（或者 prefer_tests=False 时反向为测试文件加权）
      +0~0.8  文件 degree 的归一化权重
      +0~0.5  短片段奖励（越短越高）
    """
    w = 0.0
    path = ck["path"]
    # 与 PlanTarget 关系
    overlap_bonus = 0.0
    samefile_bonus = 0.0
    for t in ctx.plan_targets:
        if t["path"] == path:
            samefile_bonus = max(samefile_bonus, 1.0)
            if _overlap(int(t["start"]), int(t["end"]), int(ck["start"]), int(ck["end"])):
                overlap_bonus = max(overlap_bonus, 1.6)
    w += max(overlap_bonus, samefile_bonus)

    # 测试文件权重
    is_test = _is_tfile_path(path)
    if ctx.prefer_tests:
        w += (0.6 if is_test else 0.3)
    else:
        w += (0.6 if not is_test else 0.3)

    # 文件 degree（0~0.8）
    deg = float(ctx.file_degree.get(path, 0.0))
    deg_norm = min(1.0, deg / 12.0)  # 粗略归一化，12度及以上视为1.0
    w += 0.8 * deg_norm

    # 短片段奖励（0~0.5）
    ln = _lines(ck)
    short_bonus = max(0.0, 0.5 - 0.0005 * ln)  # 0 行 ~ 1000 行 -> 0.5 ~ 0.0
    w += short_bonus

    return w


# ---------------- 轻量重排（enable_light_reorder=True 时使用） ----------------

def _light_reorder_select(chunks: List[DocChunk],
                          scores: Dict[int, float],
                          budget_tokens: int,
                          max_chunks: int,
                          per_file_max_chunks: int) -> List[DocChunk]:
    """
    稳定重排：按分数从高到低选择，遵守 token 预算、总片段上限、单文件片段上限。
    """
    order = sorted(range(len(chunks)), key=lambda i: (scores.get(i, 0.0), -_lines(chunks[i])), reverse=True)
    kept: List[DocChunk] = []
    token_sum = 0
    per_file_count: Dict[str, int] = defaultdict(int)

    for i in order:
        ck = chunks[i]
        p = ck["path"]
        if per_file_count[p] >= per_file_max_chunks:
            continue
        tk = _est_tokens(ck)
        if token_sum + tk > budget_tokens:
            continue
        kept.append(ck)
        token_sum += tk
        per_file_count[p] += 1
        if len(kept) >= max_chunks:
            break

    # 如果因为单个大块导致一个都装不下，兜底放一个最高分的
    if not kept and chunks:
        kept = [chunks[order[0]]]

    return kept


# ---------------- 测试片段交错 ----------------

def _interleave_tests(primary: List[DocChunk], tests: List[DocChunk]) -> List[DocChunk]:
    """
    简单交错：每插入 3~5 个非测试片段，穿插 1 个测试片段；若测试片段不多则更稀疏。
    """
    if not tests:
        return list(primary)
    # 计算步长：尽量让测试片段比例不超过 ~25%
    step = max(3, min(5, len(primary) // max(1, len(tests))))
    out: List[DocChunk] = []
    ti = 0
    for i, ck in enumerate(primary):
        out.append(ck)
        if (i + 1) % step == 0 and ti < len(tests):
            out.append(tests[ti]); ti += 1
    # 余下的测试片段（如果还剩）
    while ti < len(tests):
        out.append(tests[ti]); ti += 1
    return out


# ---------------- 主流程 ----------------

def collate(subgraph,
            plan,      # aci.schema.Plan
            cfg        # infra.config.Config
            ) -> Tuple[List[DocChunk], CollateMeta]:
    """
    组装 CGM 需要的上下文（DocChunk[]）与统计（CollateMeta）。
    - 线性化：subgraph_store.linearize(subgraph, mode=cfg.collate.mode or cfg.mode)
    - 打分：考虑 plan target / 文件度量 / 是否测试文件 / 片段长度
    - 选择：预算内 + per_file_max_chunks + max_chunks
    - 轻量重排：启用则按分数排序选择；否则保持原顺序裁剪
    - 交错：启用 interleave_tests 则将测试片段以固定步长交错
    """
    mode = getattr(cfg, "collate", getattr(cfg, "mode", "wsd")).mode if hasattr(cfg, "collate") else getattr(cfg, "mode", "wsd")
    chunks = subgraph_store.linearize(subgraph, mode=mode)  # List[DocChunk]
    meta = _summarize(chunks)

    # 早退：预算内直接返回（也记录 meta）
    budget_tokens = getattr(cfg.collate, "budget_tokens", 40000)
    max_chunks = getattr(cfg.collate, "max_chunks", 64)
    per_file_max = getattr(cfg.collate, "per_file_max_chunks", 8)
    prefer_tests = getattr(cfg, "prefer_test_files", True)
    enable_reorder = getattr(cfg.collate, "enable_light_reorder", False)
    interleave_tests = getattr(cfg.collate, "interleave_tests", True)

    # 打分上下文
    file_degree = _build_file_degree_index(subgraph)
    plan_targets = [t.__dict__ if hasattr(t, "__dict__") else dict(t) for t in getattr(plan, "targets", [])]
    ctx = _ScoreCtx(plan_targets=plan_targets, file_degree=file_degree, prefer_tests=prefer_tests)
    scores = {i: _score_chunk(ck, ctx) for i, ck in enumerate(chunks)}

    # 预算校验
    if meta["est_tokens"] <= budget_tokens and len(chunks) <= max_chunks:
        # 可选：仍然进行“轻量重排”？我们保持默认不重排，尊重线性化顺序
        # 但要应用 per_file_max_chunks 限制与交错（不改变顺序）
        limited: List[DocChunk] = []
        per_file_count: Dict[str, int] = defaultdict(int)
        token_sum = 0
        for ck in chunks:
            p = ck["path"]
            if per_file_count[p] >= per_file_max:
                continue
            tk = _est_tokens(ck)
            if token_sum + tk > budget_tokens:
                continue
            limited.append(ck)
            token_sum += tk
            per_file_count[p] += 1
            if len(limited) >= max_chunks:
                break

        # 交错
        if interleave_tests:
            prim = [c for c in limited if not _is_tfile_path(c["path"])]
            tchs = [c for c in limited if _is_tfile_path(c["path"])]
            final = _interleave_tests(prim, tchs)
        else:
            final = limited

        meta = _summarize(final)  # 以最终片段重算 meta
        return final, meta

    # 超预算：按开关走重排或直接裁剪
    if enable_reorder:
        selected = _light_reorder_select(
            chunks=chunks,
            scores=scores,
            budget_tokens=budget_tokens,
            max_chunks=max_chunks,
            per_file_max_chunks=per_file_max,
        )
        meta = _summarize(selected)
        meta["reordered"] = True
        meta.setdefault("warnings", []).append(
            f"light_reorder_applied: est>{budget_tokens}"
        )
    else:
        # 不重排：按线性化原顺序裁剪（但仍遵守 per_file_max）
        selected: List[DocChunk] = []
        per_file_count: Dict[str, int] = defaultdict(int)
        token_sum = 0
        for ck in chunks:
            p = ck["path"]
            if per_file_count[p] >= per_file_max:
                continue
            tk = _est_tokens(ck)
            if token_sum + tk > budget_tokens:
                continue
            selected.append(ck)
            token_sum += tk
            per_file_count[p] += 1
            if len(selected) >= max_chunks:
                break
        meta = _summarize(selected)
        meta.setdefault("warnings", []).append(
            f"budget_exceeded_without_reorder: est>{budget_tokens}"
        )

    # 交错（对 selected 生效）
    if interleave_tests and selected:
        prim = [c for c in selected if not _is_tfile_path(c["path"])]
        tchs = [c for c in selected if _is_tfile_path(c["path"])]
        selected = _interleave_tests(prim, tchs)
        meta = _summarize(selected)
        meta["reordered"] = meta.get("reordered", False)  # 保持标志位

    return selected, meta
