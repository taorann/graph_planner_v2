#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys

from datasets import load_dataset

from graph_planner.integrations.rllm.env import GraphPlannerRLLMEnv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test that loads a materialized *_verl.parquet and resets the repo env"
    )
    parser.add_argument("--parquet", required=True, help="Path to a *_verl.parquet file")
    args = parser.parse_args()

    dataset = load_dataset("parquet", data_files=args.parquet, split="train")
    if len(dataset) == 0:
        print("Empty dataset:", args.parquet)
        sys.exit(2)

    example = dataset[0]
    extra = example.get("extra_info")
    if isinstance(extra, str):
        extra = json.loads(extra)
    sandbox = (extra or {}).get("sandbox", {})
    print(
        "[SMOKE] sandbox:",
        {k: sandbox.get(k) for k in ["backend", "docker_image", "workdir"]},
    )

    entry = {
        "max_steps": 5,
        "sandbox": sandbox,
        "issue": {"id": "smoke-001", "title": "smoke test"},
    }
    env = GraphPlannerRLLMEnv.from_dict(entry)
    obs, info = env.reset()
    print("[SMOKE] reset ok; info keys:", list((info or {}).keys()))
    env.close()


if __name__ == "__main__":
    main()
