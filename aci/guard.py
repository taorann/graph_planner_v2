# -*- coding: utf-8 -*-
from __future__ import annotations
"""
aci/guard.py

补丁护栏校验（兼容 cfg.guard.* 和 cfg.* 两种配置写法）
- enforce_patch_guard(patch, plan, cfg) -> None | raise GuardError
- sanitize_decision(decision, defaults) -> decision  （新增：预算清洗/标准化/上限裁剪）
"""
import os
from types import SimpleNamespace
from typing import Dict, Any, List, Optional
from aci.schema import Patch, Plan


# ---------- 常量 ----------

ALLOWED_TOOLS = {"expand", "view", "search", "edit", "test", "lint", "noop"}


# ---------- 异常 ----------

class GuardError(Exception):
    pass


# ---------- 基础工具 ----------

def _is_relative_repo_path(p: str) -> bool:
    if not p or os.path.isabs(p):
        return False
    norm = os.path.normpath(p).replace("\\", "/")
    if norm.startswith("../") or norm == "..":
        return False
    return True


def _in_any_target(edit: Dict[str, Any], plan: Plan) -> bool:
    ep, es, ee = edit["path"], int(edit["start"]), int(edit["end"])
    for t in plan.targets:
        if t.path == ep and es >= int(t.start) and ee <= int(t.end):
            return True
    return False


def _non_empty_new_text(edit: Dict[str, Any]) -> bool:
    nt = edit.get("new_text")
    return isinstance(nt, str) and len(nt) > 0


def _cfg_get_guard(cfg, key: str, default):
    """
    兼容两种写法：
      1) 扁平: cfg.max_edits_per_patch
      2) 分组: cfg.guard.max_edits_per_patch
    支持对象或 dict。
    """
    if cfg is None:
        return default
    # dict 风格
    if isinstance(cfg, dict):
        if key in cfg and cfg[key] is not None:
            return cfg[key]
        guard_dict = cfg.get("guard", {}) or {}
        return guard_dict.get(key, default)
    # 对象风格
    val = getattr(cfg, key, None)
    if val is not None:
        return val
    guard_ns = getattr(cfg, "guard", SimpleNamespace())
    return getattr(guard_ns, key, default)


def _cfg_get_policy(cfg, key: str, default):
    """
    读取 policy.* 上限（支持 dict/对象，扁平或分组）
    例如：policy.max_anchors / max_anchors
    """
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        if key in cfg and cfg[key] is not None:
            return cfg[key]
        # 支持 "policy.xxx"
        pol_dict = cfg.get("policy", {}) or {}
        if key in pol_dict and pol_dict[key] is not None:
            return pol_dict[key]
        # 扁平键的最后一段
        flat_key = key.split(".")[-1]
        if flat_key in cfg and cfg[flat_key] is not None:
            return cfg[flat_key]
        if flat_key in pol_dict and pol_dict[flat_key] is not None:
            return pol_dict[flat_key]
        return default
    # 对象
    val = getattr(cfg, key, None)
    if val is not None:
        return val
    pol = getattr(cfg, "policy", SimpleNamespace())
    val2 = getattr(pol, key, None)
    if val2 is not None:
        return val2
    # 再尝试扁平最后一段
    flat = key.split(".")[-1]
    val3 = getattr(cfg, flat, None)
    if val3 is not None:
        return val3
    val4 = getattr(pol, flat, None)
    return val4 if val4 is not None else default


def _policy_limits(cfg: Any) -> Dict[str, int]:
    """统一读取策略上限（带默认值）"""
    return {
        "max_anchors": int(_cfg_get_policy(cfg, "policy.max_anchors", 3)),
        "max_terms": int(_cfg_get_policy(cfg, "policy.max_terms", 5)),
        "max_hop": int(_cfg_get_policy(cfg, "policy.max_hop", 2)),
        "max_priority_tests": int(_cfg_get_policy(cfg, "policy.max_priority_tests", 16)),
    }


# ---------- 补丁护栏 ----------

def enforce_patch_guard(patch: Patch, plan: Plan, cfg) -> None:
    edits = patch.get("edits") or []
    if not isinstance(edits, list):
        raise GuardError("patch.edits must be a list")

    # 读阈值（有默认值）
    max_edits_per_patch = int(_cfg_get_guard(cfg, "max_edits_per_patch", 20))
    max_lines_per_edit = int(_cfg_get_guard(cfg, "max_lines_per_edit", 200))

    # 数量限制
    if max_edits_per_patch > 0 and len(edits) > max_edits_per_patch:
        raise GuardError(f"too many edits in one patch: {len(edits)} > {max_edits_per_patch}")

    for i, e in enumerate(edits):
        # 路径相对
        path_ok = _is_relative_repo_path(e.get("path", ""))
        if not path_ok:
            raise GuardError(f"edit[{i}]: path must be repo-root relative, got: {e.get('path')}")

        # 行号合法
        try:
            s = int(e.get("start"))
            nd = int(e.get("end"))
        except Exception:
            raise GuardError(f"edit[{i}]: start/end must be integers")
        if s < 1 or nd < 1 or s > nd:
            raise GuardError(f"edit[{i}]: invalid range start={s}, end={nd}")

        # 行窗约束
        if not _in_any_target(e, plan):
            raise GuardError(f"edit[{i}]: edit range not inside any PlanTarget window")

        # 禁纯删除
        if not _non_empty_new_text(e):
            raise GuardError(f"edit[{i}]: pure deletion is not allowed (empty new_text)")

        # 每个编辑的最大行数
        if max_lines_per_edit > 0:
            lines = nd - s + 1
            if lines > max_lines_per_edit:
                raise GuardError(f"edit[{i}]: too many lines changed ({lines} > {max_lines_per_edit})")


# ---------- 决策清洗（新增完整实现） ----------

def _normalize_anchors(anchors: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for a in anchors or []:
        if not isinstance(a, dict):
            continue
        kind = str(a.get("kind", "")).lower().strip() or "symbol"
        text = a.get("text")
        _id = a.get("id")
        if text is None and _id is None:
            continue
        out.append({"kind": kind, "text": text, "id": _id})
    return out


def _normalize_terms(terms: Optional[List[str]]) -> List[str]:
    out, seen = [], set()
    for t in terms or []:
        if not isinstance(t, str):
            continue
        tt = t.strip()
        if not tt:
            continue
        if tt not in seen:
            seen.add(tt)
            out.append(tt)
    return out


def _normalize_priority_tests(tests: Optional[List[str]]) -> List[str]:
    out, seen = [], set()
    for x in tests or []:
        if not isinstance(x, str):
            continue
        xx = x.strip()
        if not xx:
            continue
        if xx not in seen:
            seen.add(xx)
            out.append(xx)
    return out


def sanitize_decision(decision: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    """
    预算清洗与 Guard 对齐：
    - 合并 defaults（兜底 hop/next_tool 等）；
    - 规范化 anchors/terms/priority_tests；
    - 应用策略上限（max_anchors/max_terms/max_hop/max_priority_tests）；
    - next_tool 校验到合法集合；
    - 同步设置 expand 布尔（若未提供）。
    """
    # 先合并默认值
    out: Dict[str, Any] = dict(defaults or {})
    out.update(decision or {})

    cfg = out.get("cfg") or defaults.get("cfg")
    limits = _policy_limits(cfg)

    # anchors
    anchors = _normalize_anchors(out.get("anchors"))
    if len(anchors) > limits["max_anchors"]:
        anchors = anchors[:limits["max_anchors"]]
    out["anchors"] = anchors

    # terms
    terms = _normalize_terms(out.get("terms"))
    if len(terms) > limits["max_terms"]:
        terms = terms[:limits["max_terms"]]
    out["terms"] = terms

    # hop
    hop = out.get("hop", defaults.get("hop", 1))
    try:
        hop = int(hop)
    except Exception:
        hop = 1
    if hop < 1:
        hop = 1
    if hop > limits["max_hop"]:
        hop = limits["max_hop"]
    out["hop"] = hop

    # tool
    next_tool = str(out.get("next_tool", defaults.get("next_tool", "expand"))).lower()
    if next_tool not in ALLOWED_TOOLS:
        next_tool = "expand"
    out["next_tool"] = next_tool

    # priority tests
    tests = _normalize_priority_tests(out.get("priority_tests"))
    if len(tests) > limits["max_priority_tests"]:
        tests = tests[:limits["max_priority_tests"]]
    out["priority_tests"] = tests

    # expand 布尔（与“表单”一致；未提供时用 next_tool 推断）
    if "expand" not in out:
        out["expand"] = (next_tool == "expand")

    return out
