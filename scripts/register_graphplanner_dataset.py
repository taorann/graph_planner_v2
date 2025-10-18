#!/usr/bin/env python
"""Register Graph Planner RepoEnv tasks with rLLM's dataset registry."""

from __future__ import annotations

import argparse
from pathlib import Path

from graph_planner.infra.vendor import ensure_rllm_importable

ensure_rllm_importable()

from graph_planner.integrations.rllm.dataset import (
    GRAPH_PLANNER_DATASET_NAME,
    ensure_dataset_registered,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to a JSON or JSONL file describing RepoEnv tasks.",
    )
    parser.add_argument(
        "--name",
        default=GRAPH_PLANNER_DATASET_NAME,
        help="Registry name to use (default: %(default)s).",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split name (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = ensure_dataset_registered(name=args.name, split=args.split, path=args.dataset)
    verl_path = dataset.get_verl_data_path()
    if verl_path:
        print(f"Registered dataset stored at {dataset.get_data_path()}")
        print(f"Verl formatted copy located at {verl_path}")
    else:
        print(f"Registered dataset stored at {dataset.get_data_path()} (no Verl copy found)")


if __name__ == "__main__":
    main()
