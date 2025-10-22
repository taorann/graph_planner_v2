"""Download and convert R2E-Gym and SWE-bench datasets into Graph Planner format."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable, List, Mapping, MutableMapping, Optional

from graph_planner.datasets import (
    DatasetConversionResult,
    convert_r2e_entries,
    convert_swebench_entries,
    ensure_directory,
    write_jsonl,
)
from graph_planner.runtime.containers import (
    collect_docker_images,
    prepull_docker_images,
    write_docker_manifest,
)

LOGGER = logging.getLogger(__name__)


def _load_dataset(name: str, split: str, token: Optional[str] = None) -> Iterable[Mapping[str, Any]]:
    from datasets import load_dataset

    kwargs = {}
    if token:
        kwargs["token"] = token
    dataset = load_dataset(name, split=split, **kwargs)
    return (json.loads(json.dumps(row)) for row in dataset)  # deep copy, remove Arrow types


def _subset(rows: Iterable[Mapping[str, Any]], limit: Optional[int]) -> List[Mapping[str, Any]]:
    out: List[Mapping[str, Any]] = []
    for row in rows:
        out.append(row)
        if limit is not None and len(out) >= limit:
            break
    return out


def _write_manifest_and_maybe_prepull(
    *,
    output_dir: Path,
    result: DatasetConversionResult,
    manifest_name: str,
    prepull: bool,
    max_workers: Optional[int],
    retries: Optional[int],
    delay: Optional[int],
    pull_timeout: Optional[int],
) -> Path:
    collection = collect_docker_images(
        records=result.records,
        instance_paths=result.instance_paths,
    )
    manifest_path = output_dir / manifest_name
    write_docker_manifest(manifest_path, collection.images)
    LOGGER.info(
        "Docker manifest written to %s (%d images, %d missing metadata)",
        manifest_path,
        len(collection.images),
        collection.missing,
    )
    if prepull and collection.images:
        LOGGER.info(
            "Pre-pulling %d docker images using R2E helpers", len(collection.images)
        )
        prepull_docker_images(
            collection.images,
            max_workers=max_workers,
            retries=retries,
            delay=delay,
            pull_timeout=pull_timeout,
        )
    elif prepull:
        LOGGER.warning("Pre-pull requested but no docker images detected in dataset")
    return manifest_path


def _prepare_r2e(
    *,
    output_dir: Path,
    dataset: str,
    train_split: str,
    val_split: Optional[str],
    val_size: int,
    token: Optional[str],
    train_limit: Optional[int],
    val_limit: Optional[int],
) -> DatasetConversionResult:
    LOGGER.info("Downloading R2E dataset %s (split=%s)", dataset, train_split)
    train_rows = _subset(_load_dataset(dataset, train_split, token), train_limit)

    val_rows: List[Mapping[str, Any]]
    if val_split:
        LOGGER.info("Downloading R2E validation split %s", val_split)
        val_rows = _subset(_load_dataset(dataset, val_split, token), val_limit)
    elif val_size:
        LOGGER.info("Sampling %d entries from train split for validation", val_size)
        val_rows = train_rows[:val_size]
        train_rows = train_rows[val_size:]
    else:
        val_rows = []

    ensure_directory(output_dir)
    train_result = convert_r2e_entries(
        train_rows,
        output_dir=output_dir,
        dataset_name=dataset,
        split=train_split,
    )
    write_jsonl(output_dir / "train.jsonl", train_result.records)

    combined_records = list(train_result.records)
    combined_instances = list(train_result.instance_paths)

    if val_rows:
        val_result = convert_r2e_entries(
            val_rows,
            output_dir=output_dir,
            dataset_name=dataset,
            split=val_split or "val",
        )
        write_jsonl(output_dir / "val.jsonl", val_result.records)
        combined_records.extend(val_result.records)
        combined_instances.extend(val_result.instance_paths)
        train_result = DatasetConversionResult(
            records=combined_records,
            instance_paths=combined_instances,
            skipped=train_result.skipped + val_result.skipped,
        )

    return train_result


def _prepare_swebench(
    *,
    output_dir: Path,
    dataset: str,
    split: str,
    token: Optional[str],
    limit: Optional[int],
) -> DatasetConversionResult:
    LOGGER.info("Downloading SWE-bench dataset %s (split=%s)", dataset, split)
    rows = _subset(_load_dataset(dataset, split, token), limit)
    ensure_directory(output_dir)
    result = convert_swebench_entries(
        rows,
        output_dir=output_dir,
        dataset_name=dataset,
        split=split,
    )
    write_jsonl(output_dir / f"{split}.jsonl", result.records)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Graph Planner training/eval datasets.")
    parser.add_argument("--r2e-dataset", default="R2E-Gym/R2E-Gym-Lite")
    parser.add_argument("--r2e-train-split", default="train")
    parser.add_argument("--r2e-val-split", default=None)
    parser.add_argument("--r2e-val-size", type=int, default=512)
    parser.add_argument("--r2e-train-limit", type=int, default=None)
    parser.add_argument("--r2e-val-limit", type=int, default=None)
    parser.add_argument("--r2e-output", type=Path, default=Path("datasets/r2e_gym"))
    parser.add_argument("--skip-r2e", action="store_true")

    parser.add_argument("--swebench-dataset", default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--swebench-split", default="test")
    parser.add_argument("--swebench-limit", type=int, default=None)
    parser.add_argument("--swebench-output", type=Path, default=Path("datasets/swebench"))
    parser.add_argument("--skip-swebench", action="store_true")

    parser.add_argument("--prepull-containers", action="store_true")
    parser.add_argument("--prepull-max-workers", type=int, default=None)
    parser.add_argument("--prepull-retries", type=int, default=None)
    parser.add_argument("--prepull-delay", type=int, default=None)
    parser.add_argument("--prepull-timeout", type=int, default=None)

    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    token = args.hf_token

    if not args.skip_r2e:
        r2e_result = _prepare_r2e(
            output_dir=args.r2e_output,
            dataset=args.r2e_dataset,
            train_split=args.r2e_train_split,
            val_split=args.r2e_val_split,
            val_size=args.r2e_val_size,
            token=token,
            train_limit=args.r2e_train_limit,
            val_limit=args.r2e_val_limit,
        )
        LOGGER.info(
            "R2E tasks written: %d (instances: %d, skipped: %d)",
            len(r2e_result.records),
            len(r2e_result.instance_paths),
            r2e_result.skipped,
        )
        _write_manifest_and_maybe_prepull(
            output_dir=args.r2e_output,
            result=r2e_result,
            manifest_name="docker_images.txt",
            prepull=args.prepull_containers,
            max_workers=args.prepull_max_workers,
            retries=args.prepull_retries,
            delay=args.prepull_delay,
            pull_timeout=args.prepull_timeout,
        )

    if not args.skip_swebench:
        swe_result = _prepare_swebench(
            output_dir=args.swebench_output,
            dataset=args.swebench_dataset,
            split=args.swebench_split,
            token=token,
            limit=args.swebench_limit,
        )
        LOGGER.info(
            "SWE-bench tasks written: %d (instances: %d, skipped: %d)",
            len(swe_result.records),
            len(swe_result.instance_paths),
            swe_result.skipped,
        )
        _write_manifest_and_maybe_prepull(
            output_dir=args.swebench_output,
            result=swe_result,
            manifest_name=f"docker_images_{args.swebench_split}.txt",
            prepull=args.prepull_containers,
            max_workers=args.prepull_max_workers,
            retries=args.prepull_retries,
            delay=args.prepull_delay,
            pull_timeout=args.prepull_timeout,
        )


if __name__ == "__main__":
    main()
