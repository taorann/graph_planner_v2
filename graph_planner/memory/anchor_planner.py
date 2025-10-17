# your_project/memory/anchor_planner.py
# -*- coding: utf-8 -*-
"""
Step 5.2 锚点生成器（Anchors Proposer）
--------------------------------------
签名（对外 API，与“表单”一致）：
    propose(observation_pack: dict) -> dict
返回：
    {
      "anchors": [ { "kind": "file|function|symbol|class|module|t-file", "text": "...", "id": optional } ],
      "should_expand": bool,
      "terms": [str, ...],
      "hop": 1|2
    }

职责：
- 围绕 failure_frame（path/func）优先产出锚点；
- 从 issue/top_assert 粗提 terms（去停用词、去重、限长）；
- 依据 token 软上限与子图规模决定 hop（默认 1，必要时放宽到 2）；
- should_expand 的判据与 Step 5.1 规则基线保持一致（仅就“是否需要扩展”作出建议）。

注意：
- 这里只“提议” anchors/terms/hop/should_expand；实际的 1-hop 扩展由 Step 3（mem_candidates / graph_adapter）
  执行，符合蓝皮书“职责分离”的约定。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict
import re

# ======== 观测数据契约（与蓝皮书字段名对齐） ========

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
    # 可选：透传 cfg（与 5.1 对齐；不写入对外 API）
    cfg: dict

# ======== 轻量策略配置（允许从 observation_pack["cfg"] 覆写） ========

@dataclass
class PolicyCfg:
    max_anchors: int = 3
    max_terms: int = 5
    max_hop: int = 2
    token_soft_limit: int = 16000  # 触发“收紧模式”的软上限
    tiny_subgraph_nodes: int = 1   # 判定“极小子图”的阈值

def _dig(cfg: Any, key: str, default: Any) -> Any:
    if not cfg:
        return default
    if isinstance(cfg, dict):
        # 支持 "policy.max_terms" 或 "max_terms" 两种键风格
        return cfg.get(key, cfg.get(key.split(".")[-1], default))
    return getattr(cfg, key, default)

def _compose_cfg(observation_pack: ObservationPack) -> PolicyCfg:
    cfg = observation_pack.get("cfg", {})
    return PolicyCfg(
        max_anchors=_dig(cfg, "policy.max_anchors", _dig(cfg, "max_anchors", 3)),
        max_terms=_dig(cfg, "policy.max_terms", _dig(cfg, "max_terms", 5)),
        max_hop=_dig(cfg, "policy.max_hop", _dig(cfg, "max_hop", 2)),
        token_soft_limit=_dig(cfg, "policy.token_soft_limit", _dig(cfg, "token_soft_limit", 16000)),
        tiny_subgraph_nodes=_dig(cfg, "policy.tiny_subgraph_nodes", _dig(cfg, "tiny_subgraph_nodes", 1)),
    )

# ======== 文本解析与启发式 ========

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
        wl = w.lower()
        if wl in _STOP:
            continue
        if w not in seen:
            seen.add(w)
            out.append(w)
        if len(out) >= k:
            break
    return out

def _is_token_tight(observation_pack: ObservationPack, limit: int) -> bool:
    cost = observation_pack.get("cost") or {}
    try:
        tokens = int(cost.get("tokens", 0))
    except Exception:
        tokens = 0
    return tokens >= int(limit)

# ======== 对外主函数 ========

def propose(observation_pack: Dict[str, Any]) -> Dict[str, Any]:
    """
    生成 anchors / terms / hop / should_expand 的规则提议。
    - 仅依赖 observation_pack，满足“表单”签名；
    - 1-hop 的实际展开交由 Step 3 处理。
    """
    obs: ObservationPack = observation_pack or {}
    pcfg = _compose_cfg(obs)

    sub = obs.get("subgraph_stats") or {}
    fail = obs.get("failure_frame") or {}
    issue_text = (obs.get("issue") or "") + " " + (obs.get("top_assert") or "")

    sub_nodes = int(sub.get("nodes", 0) or 0)
    last_expanded = bool(sub.get("last_expanded", False))
    failure_path = fail.get("path")
    failure_func = fail.get("func")

    token_tight = _is_token_tight(obs, pcfg.token_soft_limit)

    # ---- 1) 组装 anchors ----
    anchors: List[Dict[str, Any]] = []
    if failure_path:
        anchors.append({"kind": "file", "text": failure_path})
        if failure_func:
            anchors.append({"kind": "function", "text": failure_func})

    # 兜底：若没有失败定位，允许用 issue 文本中的“像路径/符号”的词作为 symbol/file 候选
    if not anchors:
        boot_terms = _extract_terms_from_text(issue_text, k=pcfg.max_terms)
        # 简单启发：含‘/’或‘.py/.js/.ts/.java/.go/.rs’的词视作 file 候选，其余当 symbol
        for t in boot_terms:
            tl = t.lower()
            if ("/" in t) or tl.endswith((".py",".js",".ts",".java",".go",".rs",".cpp",".c",".hpp",".h")):
                anchors.append({"kind": "file", "text": t})
            else:
                anchors.append({"kind": "symbol", "text": t})

    # 限制 anchors 个数
    anchors = anchors[:pcfg.max_anchors]

    # ---- 2) 组装 terms ----
    terms: List[str] = []
    if failure_func:
        terms.append(failure_func)
    if failure_path:
        terms.append(failure_path.split("/")[-1])
    # 补充 issue/top_assert 提取的关键词（不重复）
    extras = _extract_terms_from_text(issue_text, k=pcfg.max_terms)
    for t in extras:
        if t not in terms:
            terms.append(t)
        if len(terms) >= pcfg.max_terms:
            break

    # token 紧张时收紧 terms
    if token_tight and len(terms) > 3:
        terms = terms[:3]

    # ---- 3) 决定 hop ----
    hop = 1
    # 子图极小时可以更积极一点（若允许）
    if failure_path and sub_nodes <= pcfg.tiny_subgraph_nodes and pcfg.max_hop >= 2 and not token_tight:
        hop = 2

    # ---- 4) 是否需要扩展 ----
    # 规则：子图极小 或 刚无扩展且缺上下文 ⇒ True；若上一轮已扩展或 token 紧张 ⇒ False
    if sub_nodes <= pcfg.tiny_subgraph_nodes and not token_tight:
        should_expand = True
    elif last_expanded:
        should_expand = False
    else:
        # 若已有失败定位但子图不大，仍建议扩展一跳以纳入邻域
        should_expand = bool(failure_path) and (sub_nodes <= 2) and not token_tight

    # 返回与“表单”对齐的字段
    return {
        "anchors": anchors,
        "should_expand": should_expand,
        "terms": terms,
        "hop": hop,
    }

# ============ 简单自测（可选） ============
if __name__ == "__main__":
    demo = {
        "issue": "AssertionError in test_parser",
        "top_assert": "Expected token ')', got ']'",
        "failure_frame": {"path": "pkg/parser/core.py", "line": 128, "func": "parse_expr"},
        "subgraph_stats": {"nodes": 1, "files": 1, "funcs": 0, "classes": 0, "last_expanded": False},
        "cost": {"tokens": 8000, "elapsed_ms": 500},
        "cfg": {"policy.max_terms": 6, "policy.max_hop": 2},
    }
    print(propose(demo))
