# -*- coding: utf-8 -*-
from __future__ import annotations

"""
遥测日志：事件写入 logs/events.jsonl
"""

import json
import os
import time
from typing import Any, Dict
from .config import load


def _ensure_log_dir(path: str) -> None:
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)


def log_event(event: Dict[str, Any]) -> None:
    """
    以 JSONL 形式追加事件。
    """
    cfg = load()
    path = cfg.telemetry.events_path
    _ensure_log_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def emit_metrics(metrics: Dict[str, Any]) -> None:
    """
    控制台友好打印（可接入 Prom/StatsD；本步采用打印）。
    """
    line = " | ".join(f"{k}={v}" for k, v in metrics.items())
    print(f"[metrics] {line}")
