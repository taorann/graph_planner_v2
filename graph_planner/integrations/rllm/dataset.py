"""Graph Planner rLLM 数据集注册工具。

English summary
    Normalises dataset descriptors, resolves relative paths and plugs them into
    rLLM's dataset registry so PPO training can reuse planner tasks.
"""

from __future__ import annotations

import json
import os
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List

from ...infra.vendor import ensure_rllm_importable

ensure_rllm_importable()

LOGGER = logging.getLogger(__name__)

try:
    from rllm.rllm.data.dataset import Dataset, DatasetRegistry  # type: ignore[attr-defined]
except ModuleNotFoundError:
    try:
        from rllm.data.dataset import Dataset, DatasetRegistry  # type: ignore[attr-defined]
    except ModuleNotFoundError as _exc:  # pragma: no cover - optional dependency
        Dataset = None  # type: ignore[assignment]
        DatasetRegistry = None  # type: ignore[assignment]
        _IMPORT_ERROR = _exc
    else:
        _IMPORT_ERROR = None
else:
    _IMPORT_ERROR = None

GRAPH_PLANNER_DATASET_NAME = "graph_planner_repoenv"
GRAPH_PLANNER_CGM_DATASET_NAME = "graph_planner_cgm"


def _unique(seq: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for item in seq:
        if item and item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def resolve_task_file(
    path: str | os.PathLike[str],
    *,
    split: str | None = None,
) -> Path:
    """Resolve a dataset descriptor path, applying split-based fallbacks when missing."""

    original = Path(path).expanduser()
    if original.exists() and original.is_file():
        return original.resolve()

    candidates: List[Path] = []
    attempted: List[Path] = []

    def add_candidate(candidate: Path) -> None:
        candidate = candidate.expanduser()
        if candidate not in candidates:
            candidates.append(candidate)

    add_candidate(original)

    base_dirs: List[Path] = []
    if original.is_dir():
        base_dirs.append(original)
    else:
        base_dirs.append(original.parent)

    suffixes = _unique(
        [
            original.suffix if original.suffix else "",
            ".jsonl",
            ".json",
        ]
    )

    stems = []
    base_stem = original.stem if original.suffix else original.name
    stems.append(base_stem)
    if split:
        stems.append(split)
        if split.endswith("_verified"):
            stems.append(split[: -len("_verified")])
        else:
            stems.append(f"{split}_verified")
        stems.append(split.replace("-", "_"))
        stems.append(split.replace("_", "-"))
    stems = _unique(stems)

    for directory in base_dirs:
        for stem in stems:
            for suffix in suffixes:
                candidate = directory / f"{stem}{suffix}" if suffix else directory / stem
                add_candidate(candidate)

    for candidate in candidates:
        attempted.append(candidate)
        if candidate.is_file():
            resolved = candidate.resolve()
            if resolved != original.resolve(strict=False):
                LOGGER.info(
                    "Resolved dataset path %s -> %s using split '%s'",
                    original,
                    resolved,
                    split or "auto",
                )
            return resolved

    attempted_str = "\n  - ".join(str(p) for p in attempted[:8])
    raise FileNotFoundError(
        (
            f"Task descriptor not found: {original}. Checked candidates:\n  - {attempted_str}"
        )
    )


def _coerce_path(value: str | os.PathLike[str], *, base_dir: Path | None = None) -> str:
    """将相对路径解析为绝对路径字符串，方便 rLLM 访问。"""
    raw = Path(value)
    if not raw.is_absolute():
        if base_dir is None:
            base_dir = Path.cwd()
        raw = (base_dir / raw).resolve()
    return str(raw)


def _normalise_entry(entry: Dict[str, Any], *, base_dir: Path | None = None) -> Dict[str, Any]:
    """深拷贝任务条目并解析 sandbox/mounts 等路径，同时输出平铺字段。"""

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

    issue = payload.get("issue") or {}
    data_source = payload.get("data_source") or "graph_planner"
    summary: Dict[str, Any] = {
        "task_id": payload.get("task_id"),
        "issue_id": issue.get("id"),
        "max_steps": payload.get("max_steps"),
        "data_source": data_source,
        "raw_entry_json": json.dumps(payload, ensure_ascii=False),
    }
    if "language" in payload:
        summary["language"] = payload["language"]
    if "repo" in payload:
        summary["repo"] = payload["repo"]
    return summary


def load_task_entries(path: str | os.PathLike[str]) -> List[Dict[str, Any]]:
    """从 JSON/JSONL 文件中加载 Graph Planner 任务条目。"""
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
    """将任务文件注册到 rLLM，并返回对应 ``Dataset`` 句柄。"""
    if DatasetRegistry is None or Dataset is None:
        raise ImportError("rLLM is required for dataset registration") from _IMPORT_ERROR
    resolved = resolve_task_file(path, split=split)
    entries = load_task_entries(resolved)
    return DatasetRegistry.register_dataset(name=name, data=entries, split=split)


def ensure_dataset_registered(
    *,
    name: str = GRAPH_PLANNER_DATASET_NAME,
    split: str = "train",
    path: str | os.PathLike[str],
) -> Dataset:
    """确保数据集已注册（若缺失则重新注册）。"""
    if DatasetRegistry is None or Dataset is None:
        raise ImportError("rLLM is required for dataset registration") from _IMPORT_ERROR
    dataset = register_dataset_from_file(name=name, split=split, path=path)
    return dataset


__all__ = [
    "GRAPH_PLANNER_DATASET_NAME",
    "GRAPH_PLANNER_CGM_DATASET_NAME",
    "resolve_task_file",
    "load_task_entries",
    "register_dataset_from_file",
    "ensure_dataset_registered",
]
