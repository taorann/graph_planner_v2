"""Utilities for converting external SWE datasets into Graph Planner JSONL files."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, MutableMapping, Sequence

SANITISE_PATTERN = re.compile(r"[^a-zA-Z0-9_.-]+")


def sanitize_identifier(value: str) -> str:
    """Return a filesystem-safe identifier derived from ``value``."""

    if not value:
        raise ValueError("identifier must be a non-empty string")
    cleaned = SANITISE_PATTERN.sub("_", value)
    return cleaned.strip("._") or "task"


def ensure_directory(path: Path) -> None:
    """Ensure ``path`` exists as a directory."""

    path.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write ``rows`` to ``path`` in JSONL format using UTF-8 encoding."""

    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


@dataclass
class DatasetConversionResult:
    """Metadata emitted by dataset conversion helpers."""

    records: List[MutableMapping[str, object]]
    instance_paths: List[Path]


def _deepcopy_entry(entry: Mapping[str, object]) -> MutableMapping[str, object]:
    return json.loads(json.dumps(entry))


def _normalise_issue_text(*candidates: object, fallback: str) -> str:
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback


def _extract_first(entry: Mapping[str, object], *paths: Sequence[str]) -> object:
    for path in paths:
        value: object = entry
        for key in path:
            if isinstance(value, Mapping):
                value = value.get(key)  # type: ignore[assignment]
            else:
                value = None
                break
        if value not in (None, ""):
            return value
    return None


def convert_r2e_entries(
    entries: Iterable[Mapping[str, object]],
    *,
    output_dir: Path,
    dataset_name: str = "R2E-Gym/R2E-Gym-Lite",
    split: str = "train",
    default_max_steps: int = 40,
) -> DatasetConversionResult:
    """Convert R2E-Gym style entries into Graph Planner tasks."""

    ensure_directory(output_dir)
    instance_dir = output_dir / "instances"
    ensure_directory(instance_dir)

    records: List[MutableMapping[str, object]] = []
    ds_paths: List[Path] = []

    for raw in entries:
        entry = _deepcopy_entry(raw)
        gym_config = entry.get("gym_config")
        if not isinstance(gym_config, Mapping):
            gym_config = entry.get("ds") if isinstance(entry.get("ds"), Mapping) else entry
        gym_config = _deepcopy_entry(gym_config)  # type: ignore[arg-type]

        task_id = _extract_first(
            entry,
            ("task_id",),
            ("instance_id",),
            ("id",),
            ("task", "task_id"),
            ("task", "instance_id"),
            ("gym_config", "task_id"),
        )
        if not isinstance(task_id, str):
            raise ValueError("Unable to determine task identifier from R2E entry")
        task_id = task_id.strip() or "task"
        safe_id = sanitize_identifier(task_id)

        docker_image = _extract_first(
            gym_config,
            ("docker_image",),
            ("image_name",),
        )
        if not isinstance(docker_image, str):
            raise ValueError(f"Entry {task_id!r} did not contain a docker image")

        repo_name = _extract_first(
            gym_config,
            ("repo_name",),
            ("repo",),
        )
        if not isinstance(repo_name, str):
            repo_name = None

        max_steps = _extract_first(
            entry,
            ("max_steps",),
            ("task", "max_steps"),
            ("gym_config", "max_steps"),
        )
        if isinstance(max_steps, int):
            steps = max_steps
        else:
            try:
                steps = int(max_steps)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                steps = default_max_steps
        if steps <= 0:
            steps = default_max_steps

        issue_title = _normalise_issue_text(
            entry.get("issue_title"),
            _extract_first(entry, ("task", "title")),
            _extract_first(entry, ("task", "problem_statement_title")),
            fallback=f"R2E task {task_id}",
        )
        issue_body = _normalise_issue_text(
            entry.get("issue_body"),
            entry.get("problem_statement"),
            _extract_first(entry, ("task", "problem_statement")),
            _extract_first(entry, ("task", "description")),
            fallback=f"Resolve the issue described by {issue_title}.",
        )

        ds_path = instance_dir / f"{safe_id}.json"
        ds_paths.append(ds_path)
        ensure_directory(ds_path.parent)
        ds_path.write_text(json.dumps(gym_config, ensure_ascii=False, sort_keys=True), encoding="utf-8")

        r2e_relative = Path(os_path_relpath(ds_path, output_dir))

        record: MutableMapping[str, object] = {
            "task_id": task_id,
            "max_steps": steps,
            "issue": {
                "id": task_id,
                "title": issue_title,
                "body": issue_body,
            },
            "sandbox": {
                "docker_image": docker_image,
                "workdir": str(gym_config.get("repo_path", "/repo")),
                "mounts": {},
                "env": {},
                "backend": "repoenv",
                "r2e_ds_json": str(r2e_relative),
            },
            "data_source": dataset_name,
            "split": split,
        }
        if repo_name:
            record["repo"] = repo_name
        records.append(record)

    return DatasetConversionResult(records=records, instance_paths=ds_paths)


def convert_swebench_entries(
    entries: Iterable[Mapping[str, object]],
    *,
    output_dir: Path,
    dataset_name: str = "princeton-nlp/SWE-bench_Verified",
    split: str = "test",
    default_max_steps: int = 40,
) -> DatasetConversionResult:
    """Convert SWE-bench style entries into Graph Planner tasks."""

    ensure_directory(output_dir)
    instance_dir = output_dir / "instances"
    ensure_directory(instance_dir)

    records: List[MutableMapping[str, object]] = []
    ds_paths: List[Path] = []

    for raw in entries:
        entry = _deepcopy_entry(raw)
        instance_id = _extract_first(
            entry,
            ("instance_id",),
            ("task_id",),
            ("id",),
        )
        if not isinstance(instance_id, str):
            raise ValueError("Unable to determine instance identifier from SWE-bench entry")
        instance_id = instance_id.strip() or "instance"
        safe_id = sanitize_identifier(instance_id)

        docker_image = _extract_first(
            entry,
            ("docker_image",),
            ("image_name",),
        )
        if not isinstance(docker_image, str):
            raise ValueError(f"Entry {instance_id!r} did not contain a docker image")

        repo_name = _extract_first(entry, ("repo",), ("repo_name",))
        if not isinstance(repo_name, str):
            repo_name = None

        swe_ds = {
            "instance_id": instance_id,
            "docker_image": docker_image,
            "repo": repo_name,
            "base_commit": _extract_first(entry, ("base_commit",), ("commit",)),
            "test_patch": entry.get("test_patch"),
            "patch": entry.get("patch"),
            "tests": entry.get("tests"),
        }

        ds_path = instance_dir / f"{safe_id}.json"
        ds_paths.append(ds_path)
        ds_path.write_text(json.dumps(swe_ds, ensure_ascii=False, sort_keys=True), encoding="utf-8")

        max_steps = entry.get("max_steps")
        if isinstance(max_steps, int):
            steps = max_steps
        else:
            try:
                steps = int(max_steps)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                steps = default_max_steps
        if steps <= 0:
            steps = default_max_steps

        issue_title = _normalise_issue_text(
            entry.get("title"),
            entry.get("problem_statement"),
            fallback=f"SWE-bench task {instance_id}",
        )
        issue_body = _normalise_issue_text(
            entry.get("problem_statement"),
            entry.get("description"),
            fallback=f"Resolve the issue described by {issue_title}.",
        )

        swe_relative = Path(os_path_relpath(ds_path, output_dir))

        record: MutableMapping[str, object] = {
            "task_id": instance_id,
            "max_steps": steps,
            "issue": {
                "id": instance_id,
                "title": issue_title,
                "body": issue_body,
            },
            "sandbox": {
                "docker_image": docker_image,
                "workdir": "/repo",
                "mounts": {},
                "env": {},
                "backend": "repoenv",
                "r2e_ds_json": str(swe_relative),
            },
            "data_source": dataset_name,
            "split": split,
        }
        if repo_name:
            record["repo"] = repo_name
        records.append(record)

    return DatasetConversionResult(records=records, instance_paths=ds_paths)


def os_path_relpath(path: Path, start: Path) -> str:
    """Wrapper so we can mock os.path.relpath in tests without importing os globally."""

    from os import path as osp

    return osp.relpath(path, start)
