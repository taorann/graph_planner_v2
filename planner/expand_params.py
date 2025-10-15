# your_project/planner/expand_params.py
# -*- coding: utf-8 -*-
"""
Step 5.3 扩展参数器（Hop & Terms Planner）
---------------------------------------
对外 API：
    plan_hop_and_terms(
        observation_pack: dict,
        anchors: list[dict],
        budget: dict | None = None
    ) -> dict
返回：
    {
      "terms": [str, ...],
      "hop": 1 | 2,
      "why": {...}   # 可选遥测字段
    }

职责：
- 在 5.1/5.2 的基础上，收敛并补全 terms；根据 token/子图密度/失败证据决定 hop。
- 规则优先级：失败帧函数名/文件名 > issue/top_assert 提取 > anchors 的文本字段。
- 预算意识：接近 token 软上限时收紧 terms，固定 hop=1。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict
import re

# ======== 观测契约（对齐蓝皮书字段） ========

class FailureFrame(TypedDict, total=False):
    path: str
    line: Optional[int]
    func: Optional[str]

class SubgraphStats(TypedDict, total=False):
    nodes: int
    files: int
    funcs: int
    classes: int
    last_expanded: bool

class CostStats(TypedDict, total=False):
    tokens: int
    elapsed_ms: int

class ObservationPack(TypedDict, total=False):
    issue: str
    top_assert: Optional[str]
    error_kind: Optional[str]
    failure_frame: FailureFrame
    subgraph_stats: SubgraphStats
    cost: CostStats
    cfg: dict  # 可选：透传策略配置

Anchor = Dict[str, Any]

# ======== 轻量策略配置 ========

@dataclass
class PolicyCfg:
    max_terms: int = 5
    max_hop: int = 2
    token_soft_limit: int = 16000
    tiny_subgraph_nodes: int = 1   # 认为“极小子图”的阈值
    dense_degree_threshold: int = 8 # 稠密图（用于抑制 hop）: 仅启发式

def _dig(cfg: Any, key: str, default: Any) -> Any:
    if not cfg:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, cfg.get(key.split(".")[-1], default))
    return getattr(cfg, key, default)

def _compose_cfg(obs: ObservationPack) -> PolicyCfg:
    cfg = obs.get("cfg", {}) or {}
    return PolicyCfg(
        max_terms=_dig(cfg, "policy.max_terms", _dig(cfg, "max_terms", 5)),
        max_hop=_dig(cfg, "policy.max_hop", _dig(cfg, "max_hop", 2)),
        token_soft_limit=_dig(cfg, "policy.token_soft_limit", _dig(cfg, "token_soft_limit", 16000)),
        tiny_subgraph_nodes=_dig(cfg, "policy.tiny_subgraph_nodes", _dig(cfg, "tiny_subgraph_nodes", 1)),
        dense_degree_threshold=_dig(cfg, "policy.dense_degree_threshold", _dig(cfg, "dense_degree_threshold", 8)),
    )

# ======== 文本解析 ========

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.:/#-]{2,}")
_STOP = {
    "the","and","for","with","from","this","that","when","then","into","onto",
    "not","none","null","true","false","to","in","on","at","of","by","as","is","are",
}

def _extract_terms_from_text(text: str, k: int) -> List[str]:
    if not text:
        return []
    seen, out = set(), []
    for m in _WORD_RE.finditer(text):
        w = m.group(0)
        if w.lower() in _STOP:
            continue
        if w not in seen:
            seen.add(w)
            out.append(w)
        if len(out) >= k:
            break
    return out

def _is_token_tight(cost: CostStats, limit: int) -> bool:
    try:
        tokens = int(cost.get("tokens", 0))
    except Exception:
        tokens = 0
    return tokens >= int(limit)

# ======== 5.3：主函数 ========

def plan_hop_and_terms(
    observation_pack: Dict[str, Any],
    anchors: List[Anchor],
    budget: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    - observation_pack: 与蓝皮书一致；可含 cfg（策略参数）。
    - anchors: 5.2 产出的锚点列表（kind/text/id）。
    - budget: 可选，若包含如 {"token_soft_limit": 12000} 会覆盖配置中的默认软上限。
    """
    obs: ObservationPack = observation_pack or {}
    pcfg = _compose_cfg(obs)

    # 允许 budget 覆盖软上限（如果外部把 collate 的 token 预算下放到这里）
    if budget and "token_soft_limit" in budget:
        pcfg.token_soft_limit = int(budget["token_soft_limit"])

    sub = obs.get("subgraph_stats") or {}
    fail = obs.get("failure_frame") or {}
    cost = obs.get("cost") or {}
    issue_text = (obs.get("issue") or "") + " " + (obs.get("top_assert") or "")

    sub_nodes = int(sub.get("nodes", 0) or 0)
    failure_path = fail.get("path")
    failure_func = fail.get("func")

    token_tight = _is_token_tight(cost, pcfg.token_soft_limit)

    # ---- Terms 聚合：失败帧 > issue/top_assert > anchors 文本 ----
    terms: List[str] = []

    # 1) 失败帧最强信号
    if failure_func:
        terms.append(failure_func)
    if failure_path:
        terms.append(failure_path.split("/")[-1])

    # 2) 从 issue/top_assert 追加
    extra = _extract_terms_from_text(issue_text, k=pcfg.max_terms)
    for t in extra:
        if t not in terms:
            terms.append(t)

    # 3) 从 anchors 的 text 兜底（避免无词可用）
    for a in anchors or []:
        t = a.get("text")
        if isinstance(t, str) and t and (t not in terms):
            # 只取看起来像符号/文件名的短文本，避免把整段路径全部塞入
            if len(t) <= 64:
                terms.append(t)

    # 截断 terms（token 紧张则更紧）
    max_terms = 3 if token_tight else pcfg.max_terms
    terms = terms[:max_terms]

    # ---- hop 决策：默认 1；子图极小 + 有失败定位 + 不紧张 ⇒ 放宽到 2（若允许）
    hop = 1
    if (pcfg.max_hop >= 2) and (not token_tight):
        if failure_path and sub_nodes <= pcfg.tiny_subgraph_nodes:
            hop = 2

    # 可根据“图稠密度”做抑制（这里只能用粗略：节点数较多就保守）
    if sub_nodes >= pcfg.dense_degree_threshold:
        hop = 1

    return {
        "terms": terms,
        "hop": hop,
        "why": {
            "token_tight": token_tight,
            "subgraph_nodes": sub_nodes,
            "failure_path": failure_path,
            "failure_func": failure_func,
            "max_terms": max_terms,
            "policy": {
                "max_hop": pcfg.max_hop,
                "tiny_subgraph_nodes": pcfg.tiny_subgraph_nodes,
                "dense_degree_threshold": pcfg.dense_degree_threshold,
                "token_soft_limit": pcfg.token_soft_limit,
            },
        },
    }

# ======== 自测（仅脚本直跑时执行） ========
if __name__ == "__main__":
    demo_obs = {
        "issue": "AssertionError in test_parser",
        "top_assert": "Expected token ')', got ']'",
        "failure_frame": {"path": "pkg/parser/core.py", "line": 128, "func": "parse_expr"},
        "subgraph_stats": {"nodes": 1, "files": 1, "funcs": 0, "classes": 0, "last_expanded": False},
        "cost": {"tokens": 9000, "elapsed_ms": 500},
        "cfg": {"policy.max_terms": 6, "policy.max_hop": 2},
    }
    demo_anchors = [
        {"kind": "file", "text": "pkg/parser/core.py"},
        {"kind": "function", "text": "parse_expr"},
    ]
    print(plan_hop_and_terms(demo_obs, demo_anchors))
