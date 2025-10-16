# graph_planner/agents/rule_based/telemetry.py
# -*- coding: utf-8 -*-
"""
Step 5.7 Telemetry（只关心“第5步”的可观测）
------------------------------------------
对外 API:
    emit_step5_event(telemetry, *, issue_id: str | None,
                     decision: dict, anchors_info: dict | None,
                     hop_terms_info: dict | None, tool_sel_info: dict | None) -> None
- telemetry: 模块对象或具有 log_event(dict)->None 的实例（对齐 infra/telemetry.py）
- 其余参数：直接传 5.1/5.2/5.3/5.4 产物，用于聚合到一个事件里。
"""
from __future__ import annotations
from typing import Any, Dict, Optional

def _as_dict(x: Any) -> dict:
    return x if isinstance(x, dict) else (getattr(x, "to_dict", lambda: {})())

def emit_step5_event(
    telemetry: Any, *,
    issue_id: Optional[str],
    decision: Dict[str, Any],
    anchors_info: Optional[Dict[str, Any]],
    hop_terms_info: Optional[Dict[str, Any]],
    tool_sel_info: Optional[Dict[str, Any]],
) -> None:
    if telemetry is None or not hasattr(telemetry, "log_event"):
        return
    event = {
        "step": "step5-policy",
        "issue_id": issue_id,
        "decision": {
            "expand": bool(decision.get("expand", decision.get("should_expand", False))),
            "anchors": decision.get("anchors"),
            "terms": decision.get("terms"),
            "hop": decision.get("hop"),
            "next_tool": decision.get("next_tool"),
            "priority_tests": decision.get("priority_tests", []),
            "why": decision.get("why", {}),
        },
        "anchors_info": _as_dict(anchors_info or {}),
        "hop_terms_info": _as_dict(hop_terms_info or {}),
        "tool_sel_info": _as_dict(tool_sel_info or {}),
    }
    try:
        telemetry.log_event(event)
    except Exception:
        pass
