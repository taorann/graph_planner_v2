"""Prepare SWE-bench validation/evaluation datasets for Graph Planner."""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Mapping, Optional, Sequence

from graph_planner.datasets import (
    DatasetConversionResult,
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


def _load_hf_dataset(
    name: str, split: str, token: Optional[str] = None
) -> tuple[Iterable[Mapping[str, Any]], str]:
    """Return dataset rows for ``split`` (falling back when unavailable).

    Hugging Face only exposes the ``test`` split for SWE-bench Verified, so we
    eagerly retry with alternative split names when the requested one does not
    exist.  The returned tuple includes the effective split name so callers can
    surface accurate logging and file naming.
    """

    from datasets import load_dataset

    kwargs: dict[str, Any] = {}
    if token:
        kwargs["token"] = token

    candidates = [split]
    lowered = split.lower()
    if lowered in {"validation", "val", "dev"}:
        candidates.extend(["dev", "validation", "val", "test"])
    elif lowered == "test":
        candidates.append("validation")

    last_error: Optional[Exception] = None
    for candidate in candidates:
        try:
            dataset = load_dataset(name, split=candidate, **kwargs)
        except ValueError as exc:
            if "Unknown split" not in str(exc):
                raise
            last_error = exc
            continue

        if candidate != split:
            LOGGER.warning(
                "Requested split %s not available in %s; falling back to %s",
                split,
                name,
                candidate,
            )
        rows = (json.loads(json.dumps(row)) for row in dataset)
        return rows, candidate

    if last_error is not None:
        raise last_error
    raise ValueError(f"Unable to load split {split!r} for dataset {name!r}")


def _subset(rows: Iterable[Mapping[str, Any]], limit: Optional[int]) -> List[Mapping[str, Any]]:
    output: List[Mapping[str, Any]] = []
    for row in rows:
        output.append(row)
        if limit is not None and len(output) >= limit:
            break
    return output


def _candidate_files(root: Path, split: str) -> List[Path]:
    split_key = split.lower()
    base_candidates: Sequence[Path] = (
        root / "data" / "verified" / f"{split_key}.jsonl",
        root / "data" / "verified" / f"{split_key}.json",
        root / "data" / "verified" / f"swe-bench-verified-{split_key}.jsonl",
        root / "data" / "verified" / f"swe-bench-verified-{split_key}.json",
        root / "data" / f"swe-bench-verified-{split_key}.jsonl",
        root / "data" / f"swe-bench-verified-{split_key}.json",
        root / "data" / f"{split_key}.jsonl",
        root / "data" / f"{split_key}.json",
        root / f"swe-bench-verified-{split_key}.jsonl",
        root / f"swe-bench-verified-{split_key}.json",
        root / f"{split_key}.jsonl",
        root / f"{split_key}.json",
    )

    discovered: List[Path] = []
    for candidate in base_candidates:
        if candidate.exists():
            discovered.append(candidate)

    if not discovered:
        for path in sorted(root.rglob("*.json*")):
            if split_key in path.name.lower():
                discovered.append(path)
    return discovered


def _iter_json_records(path: Path) -> Iterator[Mapping[str, Any]]:
    reader: Any
    if path.suffix == ".gz":
        reader = gzip.open(path, "rt", encoding="utf-8")
    else:
        reader = path.open("r", encoding="utf-8")

    with reader as handle:
        suffixes = path.suffixes
        if suffixes[-2:] in ([".json", ".gz"], [".jsonl", ".gz"]) or path.suffix in {".jsonl"}:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        else:
            data = json.load(handle)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, Mapping):
                        yield item
            elif isinstance(data, Mapping):
                if "instances" in data and isinstance(data["instances"], Sequence):
                    for item in data["instances"]:
                        if isinstance(item, Mapping):
                            yield item
                elif "tasks" in data and isinstance(data["tasks"], Sequence):
                    for item in data["tasks"]:
                        if isinstance(item, Mapping):
                            yield item


def _load_local_swebench(root: Path, split: str) -> List[Mapping[str, Any]]:
    if not root.exists():
        return []

    candidates = _candidate_files(root, split)
    records: List[Mapping[str, Any]] = []
    for path in candidates:
        records.extend(_iter_json_records(path))
    return records


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
    missing = getattr(collection, "missing", 0)
    build_only = getattr(collection, "build_only", 0)
    LOGGER.info(
        "Docker manifest written to %s (%d images, %d build-only, %d missing metadata)",
        manifest_path,
        len(collection.images),
        build_only,
        missing,
    )
    if prepull and collection.images:
        LOGGER.info("Pre-pulling %d docker images using R2E helpers", len(collection.images))
        prepull_docker_images(
            collection.images,
            max_workers=max_workers,
            retries=retries,
            delay=delay,
            pull_timeout=pull_timeout,
        )
    elif prepull and build_only:
        LOGGER.warning(
            "Pre-pull requested but only build-only SWE-bench images were discovered;"
            " run swebench.harness.prepare_images to build them locally."
        )
    elif prepull:
        LOGGER.warning("Pre-pull requested but no docker images detected in dataset")
    return manifest_path


def _prepare_swebench(
    *,
    output_dir: Path,
    dataset: str,
    split: str,
    token: Optional[str],
    limit: Optional[int],
    dataset_path: Optional[Path],
) -> DatasetConversionResult:
    rows: List[Mapping[str, Any]] = []
    requested_split = split
    effective_split = split
    if dataset_path:
        rows = _subset(_load_local_swebench(dataset_path, split), limit)
        if rows:
            LOGGER.info(
                "Loaded %d entries for split %s from local SWE-bench repository %s",
                len(rows),
                split,
                dataset_path,
            )
    if not rows:
        LOGGER.info("Downloading SWE-bench dataset %s (split=%s)", dataset, split)
        hf_rows, effective_split = _load_hf_dataset(dataset, split, token)
        rows = _subset(hf_rows, limit)
        if effective_split != requested_split:
            LOGGER.warning(
                "Using split %s from %s after failing to locate %s",
                effective_split,
                dataset,
                requested_split,
            )

    ensure_directory(output_dir)
    result = convert_swebench_entries(
        rows,
        output_dir=output_dir,
        dataset_name=dataset_path.as_posix() if dataset_path else dataset,
        split=effective_split,
    )
    output_file = output_dir / f"{effective_split}.jsonl"
    write_jsonl(output_file, result.records)
    LOGGER.info("Wrote %d SWE-bench records to %s", len(result.records), output_file)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare SWE-bench validation/test datasets for Graph Planner."
    )
    parser.add_argument("--swebench-dataset", default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--swebench-split", default="validation")
    parser.add_argument("--swebench-limit", type=int, default=None)
    parser.add_argument("--swebench-output", type=Path, default=Path("datasets/swebench"))
    parser.add_argument(
        "--swebench-path",
        type=Path,
        default=Path("graph_planner/SWE-bench"),
        help="Path to a local SWE-bench checkout (used if available).",
    )
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

    if args.skip_swebench:
        LOGGER.info("Skipping SWE-bench dataset preparation (requested via --skip-swebench)")
        return

    token = args.hf_token
    split = args.swebench_split
    # Normalise alias values to huggingface names
    if split.lower() in {"val", "validation"}:
        split = "validation"
    elif split.lower() in {"dev"}:
        split = "validation"

    dataset_path: Optional[Path] = args.swebench_path
    if dataset_path and not dataset_path.exists():
        dataset_path = None

    swe_result = _prepare_swebench(
        output_dir=args.swebench_output,
        dataset=args.swebench_dataset,
        split=split,
        token=token,
        limit=args.swebench_limit,
        dataset_path=dataset_path,
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
        manifest_name=f"docker_images_{split}.txt",
        prepull=args.prepull_containers,
        max_workers=args.prepull_max_workers,
        retries=args.prepull_retries,
        delay=args.prepull_delay,
        pull_timeout=args.prepull_timeout,
    )


if __name__ == "__main__":
    main()
