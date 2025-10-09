# -*- coding: utf-8 -*-
"""
Step3 SMOKE: 端到端最小冒烟，检查 3.4 元数据是否写入 events。
python bin/step3_smoke.py
"""

import os, sys, json

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from orchestrator.loop import run_once
if __name__ == "__main__":
    r = run_once(issue="__smoke__")
    print("[SMOKE] ok =", r.get("ok"), "events_path =", r.get("events_path"))
    # 尝试读取最后一条事件
    events_path = r.get("events_path")
    if events_path and os.path.isfile(events_path):
        with open(events_path, "rb") as f:
            try:
                f.seek(-4096, os.SEEK_END)
            except Exception:
                f.seek(0)
            tail = f.read().decode("utf-8", "ignore").strip().splitlines()[-1]
        try:
            evt = json.loads(tail)
        except Exception:
            evt = {}
        has_precheck = "subgraph_precheck" in evt
        has_linearize = "linearize_meta" in evt
        print(f"[SMOKE] event has subgraph_precheck={has_precheck}, linearize_meta={has_linearize}")
    else:
        print("[SMOKE] events file not found (skip)")
