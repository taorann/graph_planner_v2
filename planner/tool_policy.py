# your_project/planner/tool_policy.py
# -*- coding: utf-8 -*-
"""
Step 5.1 决策接口 & 规则基线
--------------------------------
产出一个可直接被 `PlannerAgent`（由 `scripts/run_rule_agent.py` 启动）调用的策略函数 `decide(state, cfg)`：
- 输入：state（ObservationPack + 运行态片段），cfg（来自 infra/config 的字典或对象）
- 输出：Decision（should_expand, anchors, terms, hop, next_tool, priority_tests, why）

设计要点（与蓝皮书一致）：
1) “先决策，后执行”：只选择下一步、锚点、术语与 hop，真正的扩展/执行交给 Step 2/3/ACI。
2) 预算意识：在 tokens/步数吃紧时，优先 hop=1、terms/anchors 截断，避免膨胀。
3) 贴近失败证据：优先使用 failure_frame.path/func 作为 anchor（若可用）。
4) 轻依赖：不强依赖外部图；若 Subgraph 稀疏或空，则从 issue 文本提取 terms 兜底。

参考脉络：
- SWE-agent 的 ACI 强调把“工具调用”抽象为受控接口（本模块仅决策，不直接操作 ACI）。[Yang et al., 2024]
- RepoGraph 显示仓库级结构对于定位/导航收益显著，因此优先围绕失败栈邻域做扩展。[Ouyang et al., 2024]
- CGM 将图结构融入模型输入，启发我们将 terms/anchors 与图节点语义对齐，减少噪声。[Tao et al., 2025]
- Memory-R1 通过 RL 学习 {ADD, UPDATE, DELETE, NOOP} 型操作头，后续可把 decide() 替换为带 RL 的小头。[Yan et al., 2025]
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union
import os
import re

# =========================
# 数据契约（与蓝皮书一致）
# =========================

class FailureFrame(TypedDict, total=False):
    path: str
    line: Optional[int]
    func: Optional[str]

class SubgraphStats(TypedDict, total=False):
    nodes: int
    files: int
    funcs: int
    classes: int
    last_expanded: bool  # 上一轮是否扩展过

class CostStats(TypedDict, total=False):
    tokens: int
    elapsed_ms: int

class ObservationPack(TypedDict, total=False):
    issue: str
    error_kind: Optional[str]
    top_assert: Optional[str]
    failure_frame: FailureFrame
    subgraph_stats: SubgraphStats
    cost: CostStats

class Anchor(TypedDict, total=False):
    # kind 参考：function|symbol|file|class|module|t-file
    kind: str
    # 二选一：text 或 id（id 给外部图/子图直连）
    text: Optional[str]
    id: Optional[str]

NextTool = Literal["expand", "view", "search", "edit", "test", "lint", "noop"]

class Decision(TypedDict, total=False):
    should_expand: bool
    anchors: List[Anchor]
    terms: List[str]
    hop: int  # 1 或 2（默认 1）
    next_tool: NextTool
    priority_tests: List[str]
    why: Dict[str, Any]  # 便于事件分析

# =========================
# 轻量配置读取（与 infra/config 对齐）
# =========================

@dataclass
class PolicyCfg:
    # 预算与上限（若 cfg 中没有对应字段，使用安全默认）
    max_anchors: int = 3
    max_terms: int = 5
    max_hop: int = 2
    token_soft_limit: int = 16000  # 软上限（触发“保守模式”）
    prefer_tfile_tests: bool = True

    # 工具选择阈值
    min_subgraph_nodes_for_view: int = 4
    consider_view_when_failure_located: bool = True

def _dig(cfg: Any, key: str, default: Any) -> Any:
    """从 dict/对象中容错读取配置。"""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)

def _compose_cfg(cfg: Any) -> PolicyCfg:
    return PolicyCfg(
        max_anchors=_dig(cfg, "policy.max_anchors", _dig(cfg, "max_anchors", 3)),
        max_terms=_dig(cfg, "policy.max_terms", _dig(cfg, "max_terms", 5)),
        max_hop=_dig(cfg, "policy.max_hop", _dig(cfg, "max_hop", 2)),
        token_soft_limit=_dig(cfg, "policy.token_soft_limit", _dig(cfg, "token_soft_limit", 16000)),
        prefer_tfile_tests=_dig(cfg, "policy.prefer_tfile_tests", _dig(cfg, "prefer_tfile_tests", True)),
        min_subgraph_nodes_for_view=_dig(cfg, "policy.min_subgraph_nodes_for_view", _dig(cfg, "min_subgraph_nodes_for_view", 4)),
        consider_view_when_failure_located=_dig(cfg, "policy.consider_view_when_failure_located", _dig(cfg, "consider_view_when_failure_located", True)),
    )

# =========================
# 规则基线的工具函数
# =========================

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.:/#-]{2,}")

def _extract_terms_from_text(text: str, k: int) -> List[str]:
    """
    从 issue/top_assert 中粗提 terms：
    - 允许文件/函数样式片段（带 / . #）
    - 去重、按出现顺序截断
    """
    if not text:
        return []
    seen = set()
    out: List[str] = []
    for m in _WORD_RE.finditer(text):
        w = m.group(0)
        # 过滤纯数字/过短 token
        if w.lower() in ("the", "and", "for", "with", "from", "this", "that", "when", "then"):
            continue
        if w not in seen:
            seen.add(w)
            out.append(w)
        if len(out) >= k:
            break
    return out

def _guess_test_names(failure_path: Optional[str]) -> List[str]:
    """
    依据失败路径猜测优先测试名；若含 test/tests，直接返回该测试文件；
    否则用同目录的 test* 作为候选（由外层 test runner 决定是否存在）。
    """
    if not failure_path:
        return []
    parts = failure_path.replace("\\", "/").split("/")
    filename = parts[-1] if parts else ""
    directory = "/".join(parts[:-1]) if len(parts) > 1 else ""
    out: List[str] = []
    if "test" in filename.lower():
        out.append(failure_path)
        return out
    if "tests" in directory.lower():
        # 失败发生在 tests 目录下的源码文件（少见），仍优先该目录
        out.append(failure_path)
        return out
    # 兜底：同目录的测试前缀（交给外部 runner 实际匹配）
    if directory:
        out.append(os.path.join(directory, "test_*"))
    return out

def _is_token_tight(cost: CostStats, limit: int) -> bool:
    tokens = int(cost.get("tokens", 0)) if isinstance(cost, dict) else 0
    return tokens >= int(limit)

# =========================
# 核心：规则型策略 decide()
# =========================

def decide(state: Dict[str, Any], cfg: Any = None) -> Decision:
    """
    规则基线：
    1) 若 subgraph 为空或极小：should_expand=True，hop=1，terms 从 issue/top_assert 提取；
    2) 若有 failure_frame.path：优先围绕该文件/函数扩展；若子图规模足够，可先 view；
    3) 若 tokens 逼近软上限：收紧 anchors/terms，固定 hop=1；
    4) 若上一轮已扩展（last_expanded=True）：改为 view 或 test（优先 t-file）以收敛闭环；
    """
    pcfg = _compose_cfg(cfg)

    obs: ObservationPack = state.get("observation", {}) or {}
    sub: SubgraphStats = obs.get("subgraph_stats", {}) or {}
    fail: FailureFrame = obs.get("failure_frame", {}) or {}
    cost: CostStats = obs.get("cost", {}) or {}
    issue_text = (obs.get("issue") or "") + " " + (obs.get("top_assert") or "")

    sub_nodes = int(sub.get("nodes", 0) or 0)
    last_expanded = bool(sub.get("last_expanded", False))

    failure_path = fail.get("path")
    failure_func = fail.get("func")

    # === 预算意识：收紧模式 ===
    token_tight = _is_token_tight(cost, pcfg.token_soft_limit)

    # === 锚点与 terms 初稿 ===
    anchors: List[Anchor] = []
    terms: List[str] = []

    if failure_path:
        # 失败路径优先：file 级 anchor
        anchors.append({"kind": "file", "text": failure_path})
        if failure_func:
            anchors.append({"kind": "function", "text": failure_func})
        # 从路径/函数名补齐 terms
        base_terms = [failure_func] if failure_func else []
        base_terms += [failure_path.split("/")[-1]]
        terms.extend([t for t in base_terms if t])
    else:
        # 无失败定位：从 issue 文本提取
        terms.extend(_extract_terms_from_text(issue_text, k=pcfg.max_terms))

    # 再补一轮：issue/top_assert 中的关键词兜底（不重复）
    if len(terms) < pcfg.max_terms and issue_text:
        extra = _extract_terms_from_text(issue_text, k=pcfg.max_terms - len(terms))
        for t in extra:
            if t not in terms:
                terms.append(t)

    # 截断 anchors/terms 以符合预算
    if token_tight:
        max_terms = min(pcfg.max_terms, 3)
        max_anchors = min(pcfg.max_anchors, 2)
        hop = 1
    else:
        max_terms = pcfg.max_terms
        max_anchors = pcfg.max_anchors
        hop = 1  # 默认 1；更积极的 2 交给 5.3
        # 简单放宽：若有 failure 定位且子图很小，可允许 hop=2
        if failure_path and sub_nodes <= 2 and pcfg.max_hop >= 2:
            hop = 2

    anchors = anchors[:max_anchors]
    terms = terms[:max_terms]

    # === 选择 next_tool ===
    # 优先级（启发式）：
    # - 子图极小 / 无证据：expand
    # - 刚扩展过：view（观察新片段）或 test（若刚应用过补丁）
    # - 有明确 failure 定位：view（便于人/模型理解上下文），否则 expand
    # - 若外层标记“刚打过补丁”（state["just_patched"]），则 test
    just_patched = bool(state.get("just_patched", False))

    if just_patched:
        next_tool: NextTool = "test"
    elif sub_nodes <= 1:
        next_tool = "expand"
    elif failure_path and pcfg.consider_view_when_failure_located and sub_nodes >= pcfg.min_subgraph_nodes_for_view and not last_expanded:
        next_tool = "view"
    elif last_expanded:
        next_tool = "view"
    else:
        # 证据一般：若 tokens 紧张则直接 test，否则 expand 以获取更多上下文
        next_tool = "test" if token_tight else "expand"

    # === 是否 should_expand ===
    should_expand = next_tool == "expand"

    # === 测试优先级（粗规则）：围绕失败路径与 t-file 方向 ===
    priority_tests = _guess_test_names(failure_path)
    # 可选：如果 cfg 指明更偏好 t-file，可在 5.5 再细化

    decision: Decision = {
        "should_expand": should_expand,
        "anchors": anchors,
        "terms": terms,
        "hop": hop,
        "next_tool": next_tool,
        "priority_tests": priority_tests,
        "why": {
            "token_tight": token_tight,
            "subgraph_nodes": sub_nodes,
            "last_expanded": last_expanded,
            "failure_path": failure_path,
            "failure_func": failure_func,
            "limits": {
                "max_terms": max_terms,
                "max_anchors": max_anchors,
                "max_hop": pcfg.max_hop,
            },
        },
    }
    return decision

# =============== 可选：简单自测 ===============
if __name__ == "__main__":
    fake_state = {
        "observation": {
            "issue": "[FAIL] test_parse: AssertionError in parser",
            "top_assert": "Expected token ')', got ']'",
            "failure_frame": {"path": "pkg/parser/core.py", "line": 128, "func": "parse_expr"},
            "subgraph_stats": {"nodes": 2, "files": 1, "funcs": 1, "classes": 0, "last_expanded": False},
            "cost": {"tokens": 9000, "elapsed_ms": 1200},
        },
        "just_patched": False,
    }
    print(decide(fake_state, cfg={"policy.max_terms": 6, "policy.max_hop": 2}))
