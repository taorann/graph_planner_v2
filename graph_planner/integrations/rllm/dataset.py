"""Utilities for registering Graph Planner tasks with rLLM's dataset registry."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from ...infra.vendor import ensure_rllm_importable

ensure_rllm_importable()

try:
    from rllm.data.dataset import Dataset, DatasetRegistry
except ImportError as _exc:  # pragma: no cover - optional dependency
    Dataset = None  # type: ignore[assignment]
    DatasetRegistry = None  # type: ignore[assignment]
    _IMPORT_ERROR = _exc
else:
    _IMPORT_ERROR = None

GRAPH_PLANNER_DATASET_NAME = "graph_planner_repoenv"


def _coerce_path(value: str | os.PathLike[str], *, base_dir: Path | None = None) -> str:
    """Normalise a filesystem path for rLLM consumption."""
    raw = Path(value)
    if not raw.is_absolute():
        if base_dir is None:
            base_dir = Path.cwd()
        raw = (base_dir / raw).resolve()
    return str(raw)


def _normalise_entry(entry: Dict[str, Any], *, base_dir: Path | None = None) -> Dict[str, Any]:
    payload = json.loads(json.dumps(entry))  # deep copy without NumPy types
    sandbox = payload.get("sandbox") or {}
    ds_path = sandbox.get("r2e_ds_json")
    if ds_path:
        sandbox["r2e_ds_json"] = _coerce_path(ds_path, base_dir=base_dir)
    mounts = sandbox.get("mounts")
    if isinstance(mounts, dict):
        resolved: Dict[str, str] = {}
        for host, container in mounts.items():
            resolved[_coerce_path(host, base_dir=base_dir)] = container
        sandbox["mounts"] = resolved
    payload["sandbox"] = sandbox
    return payload


def load_task_entries(path: str | os.PathLike[str]) -> List[Dict[str, Any]]:
    """Load Graph Planner RL tasks from a JSON/JSONL file."""
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Task descriptor not found: {path_obj}")
    base_dir = path_obj.parent
    if path_obj.suffix == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path_obj.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    elif path_obj.suffix == ".json":
        with path_obj.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            rows = [payload]
        else:
            rows = list(payload)
    else:
        raise ValueError(f"Unsupported dataset extension: {path_obj.suffix}")
    return [_normalise_entry(row, base_dir=base_dir) for row in rows]


def register_dataset_from_file(
    *,
    name: str = GRAPH_PLANNER_DATASET_NAME,
    split: str = "train",
    path: str | os.PathLike[str],
) -> Dataset:
    """Register a dataset file with rLLM's registry and return the Dataset handle."""
    if DatasetRegistry is None or Dataset is None:
        raise ImportError("rLLM is required for dataset registration") from _IMPORT_ERROR
    entries = load_task_entries(path)
    return DatasetRegistry.register_dataset(name=name, data=entries, split=split)


def ensure_dataset_registered(
    *,
    name: str = GRAPH_PLANNER_DATASET_NAME,
    split: str = "train",
    path: str | os.PathLike[str],
) -> Dataset:
    """Idempotently register the dataset if the Verl artefacts are missing."""
    if DatasetRegistry is None or Dataset is None:
        raise ImportError("rLLM is required for dataset registration") from _IMPORT_ERROR
    dataset = register_dataset_from_file(name=name, split=split, path=path)
    return dataset
