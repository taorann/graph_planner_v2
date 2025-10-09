# -*- coding: utf-8 -*-
"""
Step4 冒烟：Collate → 本地 CGM → Guard → ACI → Lint/Test
python bin/step4_smoke.py
"""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from orchestrator.loop import run_once
from infra.config import load

if __name__ == "__main__":
    cfg = load()
    print(f"[CFG] mode={cfg.mode} collate.budget_tokens={cfg.collate.budget_tokens} "
          f"enable_light_reorder={cfg.collate.enable_light_reorder}")
    res = run_once(issue="__smoke_step4__")
    ok = res.get("ok", False)
    print(f"[SMOKE] ok = {ok} events_path = {res.get('events_path')}")
    if not ok:
        sys.exit(1)
