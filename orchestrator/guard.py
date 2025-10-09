# -*- coding: utf-8 -*-
from __future__ import annotations
"""
orchestrator/guard.py

补丁护栏校验（兼容 cfg.guard.* 和 cfg.* 两种配置写法）
- enforce_patch_guard(patch, plan, cfg) -> None | raise GuardError
- sanitize_decision(decision, defaults) -> decision
"""

import os
from types import SimpleNamespace
from typing import Dict, Any
from aci.schema import Patch, Plan


class GuardError(Exception):
    pass


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
    """
    val = getattr(cfg, key, None)
    if val is not None:
        return val
    guard_ns = getattr(cfg, "guard", SimpleNamespace())
    return getattr(guard_ns, key, default)


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


def sanitize_decision(decision: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(defaults or {})
    out.update(decision or {})
    return out
