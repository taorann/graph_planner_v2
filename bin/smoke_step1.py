#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Step-1 smoke test under a given working directory (default: current dir).
It temporarily chdirs into the target dir so orchestrator.loop will pick files there.
Usage:
  python bin/smoke_step1.py __aci_step1_demo
Env:
  ISSUE=<id or short text>   # optional, recorded in events.jsonl
"""
import os
import sys

def main():
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    target_dir = os.path.abspath(target_dir)
    if not os.path.isdir(target_dir):
        raise SystemExit(f"Not a directory: {target_dir}")
    prev = os.getcwd()
    os.chdir(target_dir)
    try:
        from orchestrator.loop import run_once
        res = run_once(issue=os.environ.get("ISSUE", ""))
        print(res)
    finally:
        os.chdir(prev)

if __name__ == "__main__":
    main()
