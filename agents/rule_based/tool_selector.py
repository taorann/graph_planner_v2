# graph_planner/agents/rule_based/tool_selector.py
# -*- coding: utf-8 -*-
"""
Step 5.4 工具选择策略（Tool Selection Policy）
--------------------------------------------
对外 API:
    choose_next_tool(state: dict) -> dict

返回:
    {
      "next_tool": "expand" | "view" | "search" | "edit" | "test" | "lint" | "noop",
      "priority_tests": [str, ...],
      "why": {...}   # 遥测说明，供事件落盘与训练用
    }

设计目标:
- 与 5.1/5.2/5.3 解耦：本模块只决定“下一步调用哪个工具”，不生成锚点/terms/hop。
- 预算感知：在 token 紧张或上下文已经充足时，避免继续扩展。
- 闭环节奏：扩展→查看/编辑→测试；支持 Lint/Noop 支路。
- RL Hook：保留 `rl_override()` 占位，后续可注入（不破坏接口）。

依赖输入(state):
- observation: ObservationPack（蓝皮书定义）
- just_patched: bool            # 刚应用过补丁
- last_lint_failed: bool        # 上次 lint 失败（可选）
- last_test_failed: bool        # 上次 test 失败（可选）
- collate_meta: dict            # 上次线性化统计（可选，含 est_tokens 等）
- cfg: dict 或 Config.to_dict() # 可选策略阈值
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict
import os

# --------- 契约类型（与蓝皮书） ---------

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

NextTool = str  # "expand" | "view" | "search" | "edit" | "test" | "lint" | "noop"

# --------- 策略参数 ---------

@dataclass
class ToolPolicyCfg:
    token_soft_limit: int = 16000      # 达到或超过则认为紧张
    min_nodes_to_view: int = 2         # 子图不小于该阈值，才有意义 “view”
    prefer_tfile_tests: bool = True    # 推测测试选择器时优先 tests 方向
    max_rounds_without_test: int = 2   # 连续多少轮没测过，强制转 test
    est_tokens_soft_limit: int = 18000 # collate_meta.est_tokens 的软上限

def _dig(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        # 支持 policy.* 或扁平键
        return cfg.get(key, cfg.get(key.split(".")[-1], default))
    return getattr(cfg, key, default)

def _compose_cfg(cfg_like: Any) -> ToolPolicyCfg:
    return ToolPolicyCfg(
        token_soft_limit = _dig(cfg_like, "policy.token_soft_limit", _dig(cfg_like, "token_soft_limit", 16000)),
        min_nodes_to_view = _dig(cfg_like, "policy.min_subgraph_nodes_for_view", _dig(cfg_like, "min_subgraph_nodes_for_view", 2)),
        prefer_tfile_tests = bool(_dig(cfg_like, "policy.prefer_tfile_tests", _dig(cfg_like, "prefer_tfile_tests", True))),
        max_rounds_without_test = int(_dig(cfg_like, "policy.max_rounds_without_test", _dig(cfg_like, "max_rounds_without_test", 2))),
        est_tokens_soft_limit = int(_dig(cfg_like, "policy.est_tokens_soft_limit", _dig(cfg_like, "est_tokens_soft_limit", 18000))),
    )

# --------- 辅助：优先测试名推断 ---------

def _guess_priority_tests(failure_path: Optional[str]) -> List[str]:
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
        out.append(failure_path)
        return out
    if directory:
        out.append(os.path.join(directory, "test_*"))
    return out

def _token_tight(cost: Dict[str, Any], soft_limit: int) -> bool:
    try:
        return int((cost or {}).get("tokens", 0)) >= int(soft_limit)
    except Exception:
        return False

def _context_heavy(collate_meta: Optional[Dict[str, Any]], limit: int) -> bool:
    if not isinstance(collate_meta, dict):
        return False
    est = collate_meta.get("est_tokens")
    try:
        return est is not None and int(est) >= int(limit)
    except Exception:
        return False

# --------- RL Hook（占位） ---------

def rl_override(proposal: Dict[str, Any], state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    预留：若要用 RL/小头覆盖策略，在这里读取 state（含观测/历史特征），
    输出与 proposal 同 schema 的 dict 即可。返回 None 则保持原提案。
    """
    return None

# --------- 对外主函数 ---------

def choose_next_tool(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    规则版工具选择器。
    - 优先级：
        1) 刚打完补丁 -> test
        2) 上次 lint 失败 -> lint
        3) 上次 test 失败但未改动 -> edit/view（根据是否有 failure 定位）
        4) 上下文不足（子图小 / 无失败定位） -> expand
        5) 上一轮刚扩展 -> view（先看再动）
        6) token 或 context 紧张 -> test（收敛闭环）
        7) 其他 -> expand 或 view（看子图大小和失败定位）
    """
    obs: ObservationPack = state.get("observation", {}) or {}
    sub = obs.get("subgraph_stats", {}) or {}
    fail = obs.get("failure_frame", {}) or {}
    cost = obs.get("cost", {}) or {}
    collate_meta = state.get("collate_meta")
    cfg_like = state.get("cfg")
    pcfg = _compose_cfg(cfg_like)

    # 快速信号
    just_patched = bool(state.get("just_patched", False))
    last_lint_failed = bool(state.get("last_lint_failed", False))
    last_test_failed = bool(state.get("last_test_failed", False))
    rounds_since_test = int(state.get("rounds_since_test", 0))

    sub_nodes = int(sub.get("nodes", 0) or 0)
    last_expanded = bool(sub.get("last_expanded", False))
    failure_path = fail.get("path")
    token_tight = _token_tight(cost, pcfg.token_soft_limit)
    context_heavy = _context_heavy(collate_meta, pcfg.est_tokens_soft_limit)

    # 1) 刚打补丁 -> test
    if just_patched:
        next_tool: NextTool = "test"
        return {
            "next_tool": next_tool,
            "priority_tests": _guess_priority_tests(failure_path),
            "why": {"rule": "just_patched -> test"}
        }

    # 2) lint 失败 -> lint
    if last_lint_failed:
        return {
            "next_tool": "lint",
            "priority_tests": [],
            "why": {"rule": "last_lint_failed -> lint"}
        }

    # 3) 强制测试频率（避免长期不测）
    if rounds_since_test >= pcfg.max_rounds_without_test:
        return {
            "next_tool": "test",
            "priority_tests": _guess_priority_tests(failure_path),
            "why": {"rule": f"rounds_since_test({rounds_since_test}) >= {pcfg.max_rounds_without_test} -> test"}
        }

    # 4) 上下文不足：子图极小 或 无失败定位 -> expand
    if (sub_nodes < pcfg.min_nodes_to_view) or (not failure_path):
        # 但若 token/context 已经紧张，则不要继续扩展，转 test 收敛
        if token_tight or context_heavy:
            return {
                "next_tool": "test",
                "priority_tests": _guess_priority_tests(failure_path),
                "why": {"rule": "context tight but small graph -> test"}
            }
        return {
            "next_tool": "expand",
            "priority_tests": [],
            "why": {"rule": "small graph or no failure_path -> expand", "sub_nodes": sub_nodes, "has_failure_path": bool(failure_path)}
        }

    # 5) 上一轮刚扩展 -> view（先看再动）
    if last_expanded:
        return {
            "next_tool": "view",
            "priority_tests": [],
            "why": {"rule": "last_expanded -> view"}
        }

    # 6) test 收敛条件：token/context 紧张 或 上次 test 失败
    if token_tight or context_heavy or last_test_failed:
        return {
            "next_tool": "test",
            "priority_tests": _guess_priority_tests(failure_path),
            "why": {
                "rule": "token/context tight or last_test_failed -> test",
                "token_tight": token_tight,
                "context_heavy": context_heavy,
                "last_test_failed": last_test_failed
            }
        }

    # 7) 常规：有失败定位 + 子图不小 -> 先 view；否则 expand
    if failure_path and (sub_nodes >= pcfg.min_nodes_to_view):
        proposal = {
            "next_tool": "view",
            "priority_tests": [],
            "why": {"rule": "failure located & graph large enough -> view", "sub_nodes": sub_nodes}
        }
    else:
        proposal = {
            "next_tool": "expand",
            "priority_tests": [],
            "why": {"rule": "fallback expand"}
        }

    # RL Hook（可选覆盖）
    override = rl_override(proposal, state)
    return override if isinstance(override, dict) else proposal
