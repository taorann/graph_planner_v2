# -*- coding: utf-8 -*-
from __future__ import annotations

"""
最小闭环（Step 4.3 集成版）：
Plan → Collate → CGM(本地) → Guard → ACI 应用 → Lint/Test → Feedback → 事件落盘
"""

import os
from typing import Any, Dict, List, Optional, Tuple

from infra.config import load, Config
from infra.telemetry import log_event, emit_metrics
from aci.schema import Plan, PlanTarget, Patch, Feedback, CollateMeta, CGMReqMeta
from aci import tools as aci_tools
from actor import cgm_adapter
from actor.collater import collate
from .guard import enforce_patch_guard, GuardError

# ========== 3.x 遗留：极简 Plan 生成（仍保留） ==========

def _make_plan(cfg: Config) -> Plan:
    """
    极简 Plan 生成：挑选一个可编辑文本文件，定位第一行非空行，做 1 行窗口。
    优先 .py，其次 .md/.txt。
    """
    roots = [os.getcwd()]
    candidates = _collect_candidates(roots, [".py"]) or _collect_candidates(roots, [".md", ".txt"])
    if not candidates:
        raise RuntimeError("No candidate text files (.py/.md/.txt) found to build a minimal plan.")

    path = candidates[0]
    start = _first_nonempty_line(path) or 1
    tgt = PlanTarget(path=os.path.relpath(path, os.getcwd()),
                     start=start, end=start,
                     id=f"auto::{os.path.basename(path)}::{start}",
                     confidence=1.0, why="ACI smoke plan (single-line)")
    return Plan(targets=[tgt], budget={"mode": cfg.mode}, priority_tests=[])


def _collect_candidates(roots: List[str], exts: List[str]) -> List[str]:
    out: List[str] = []
    for r in roots:
        for dirpath, _, files in os.walk(r):
            # 排除 .git/.aci/backups 目录
            parts = dirpath.split(os.sep)
            if any(p.startswith(".git") for p in parts) or ".aci" in parts:
                continue
            for fn in files:
                if any(fn.endswith(ext) for ext in exts):
                    out.append(os.path.join(dirpath, fn))
    out.sort()
    return out


def _first_nonempty_line(path: str) -> int | None:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f, start=1):
            if line.strip():
                return i
    return None

# ========== 4.3 主流程 ==========

def run_once(issue: Optional[str] = None) -> Dict[str, Any]:
    """
    端到端一轮：
      1) 生成 Plan
      2) Collate（线性化 + 评分 + 预算 + 轻量重排）
      3) 本地 CGM 生成补丁
      4) Guard 校验（行窗/路径/禁纯删除/数量/行数）
      5) ACI 应用（逐 edit）
      6) Lint & Test
      7) 事件与指标
    """
    cfg = load()

    # 1) 计划
    plan = _make_plan(cfg)

    # 2) collate：基于子图线性化 + 预算控制（这里暂时不显式维护子图，沿用 step3 的状态管理的话，可替换 subgraph 来源）
    # 如果你已经在 step3 中维护了 subgraph，可在此处加载：subgraph = subgraph_store.load(issue or "__default__")
    # 这里使用一个最小兜底：不传 subgraph 也能工作（collater 内部会处理）
    try:
        # 若你已有子图对象，请替换为实际 subgraph
        subgraph = {"nodes": {}, "edges": []}
        subgraph_linearized, collate_meta = collate(subgraph, plan, cfg)
    except Exception as e:
        # 退化到无上下文（CGM 本地版依然可行）
        subgraph = None
        subgraph_linearized, collate_meta = [], {"chunks": 0, "est_tokens": 0, "reordered": False, "warnings": [f"collate_failed: {e}"]}

    # 3) CGM：本地占位生成补丁（仍受 Plan 行窗约束）
    cgm_req_meta: CGMReqMeta = {
        "endpoint": "local-cgm",
        "model": "local-placeholder",
        "temperature": 0.0,
        "max_tokens": 0,
        "timeout_s": 0,
    }
    patch = _call_cgm(plan, cfg, subgraph_linearized=subgraph_linearized)

    # 4) Guard 校验
    try:
        enforce_patch_guard(patch, plan, cfg)
    except GuardError as ge:
        fb: Feedback = {"ok": False, "lint_report": {}, "test_report": {}, "diff_summary": str(ge), "changed_files": []}
        event = _pack_event(issue, plan, patch, fb, cfg,
                            extra={"collate_meta": collate_meta, "cgm_req_meta": cgm_req_meta})
        log_event(event)
        return {"ok": False, "error": f"GuardError: {ge}", "events_path": cfg.telemetry.events_path}

    # 5) ACI 应用
    apply_resp = _apply_patch_with_guard(patch, plan)
    changed_files = list({e["path"] for e in patch.get("edits", [])}) if apply_resp["success"] else []

    # 6) Lint & Test
    lint_report = {}
    if cfg.lint.enabled and changed_files:
        lint_resp = aci_tools.lint_check(changed_files)
        lint_report = {
            "ok": lint_resp["success"],
            "issues": lint_resp["data"].get("issues", 0),
            "framework_rc": lint_resp["data"].get("rc"),
        }

    test_resp = aci_tools.run_tests([])
    test_report = {
        "ok": test_resp["success"],
        "framework": test_resp["data"].get("framework"),
        "passed": test_resp["data"].get("passed") or test_resp["data"].get("ran"),
        "failed": test_resp["data"].get("failed"),
        "errors": test_resp["data"].get("errors"),
    }

    ok = apply_resp["success"] and test_resp["success"]
    fb: Feedback = {
        "ok": ok,
        "lint_report": lint_report,
        "test_report": test_report,
        "diff_summary": apply_resp["message"],
        "changed_files": changed_files,
    }

    # 7) 事件与指标
    event = _pack_event(issue, plan, patch, fb, cfg,
                        apply_resp=apply_resp, test_resp=test_resp,
                        extra={"collate_meta": collate_meta, "cgm_req_meta": cgm_req_meta})
    log_event(event)
    emit_metrics({
        "ok": ok,
        "files_changed": len(changed_files),
        "lint_ok": lint_report.get("ok", False) if lint_report else None,
        "tests_ok": test_resp["success"],
        "context_tokens": collate_meta.get("est_tokens", 0),
        "patch_changed_lines": apply_resp["data"].get("changed_lines", 0),
    })
    return {"ok": ok, "feedback": fb, "events_path": cfg.telemetry.events_path}


# ---------- helpers ----------

def _call_cgm(plan: Plan, cfg: Config, *, subgraph_linearized: Optional[List[Dict[str, Any]]] = None) -> Patch:
    return cgm_adapter.generate(subgraph_linearized=subgraph_linearized, plan=plan, constraints={"mode": cfg.mode})


def _apply_patch_with_guard(patch: Patch, plan: Plan):
    """
    实际应用补丁（顺序处理每个 edit），底层使用 ACI.edit_lines。
    """
    total_changed = 0
    total_size = 0
    logs: List[str] = []
    ok = True
    for e in patch.get("edits", []) or []:
        resp = aci_tools.edit_lines(e["path"], int(e["start"]), int(e["end"]), e["new_text"])
        logs.append(resp["message"])
        ok = ok and resp["success"]
        total_changed += int(resp["data"].get("changed_lines", 0)) if resp["success"] else 0
        total_size += len(e.get("new_text") or "")

    msg = f"Applied {len(patch.get('edits', []) or [])} edit(s), changed_lines={total_changed}, patch_size={total_size}"
    return {
        "success": ok,
        "message": msg,
        "data": {"changed_lines": total_changed, "edits": patch.get("edits", []), "patch_size": total_size},
        "logs": logs,
    }


def _pack_event(issue: Optional[str], plan: Plan, patch: Patch, feedback: Dict[str, Any],
                cfg: Config, apply_resp: Dict[str, Any] | None = None,
                test_resp: Dict[str, Any] | None = None,
                extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    base = {
        "issue_id": issue or "",
        "step": "step4-collate-cgm-guard",
        "mode": cfg.mode,
        "plan_targets": [t.__dict__ for t in plan.targets],
        "patch_summary": patch.get("summary", ""),
        "apply": apply_resp or {},
        "feedback": feedback,
        "config": cfg.to_dict(),
    }
    if extra:
        base.update(extra)
    return base
