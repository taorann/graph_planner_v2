"""Graph Planner rLLM 数据集注册工具。

English summary
    Normalises dataset descriptors, resolves relative paths and plugs them into
    rLLM's dataset registry so PPO training can reuse planner tasks.
"""

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
GRAPH_PLANNER_CGM_DATASET_NAME = "graph_planner_cgm"


def _coerce_path(value: str | os.PathLike[str], *, base_dir: Path | None = None) -> str:
    """将相对路径解析为绝对路径字符串，方便 rLLM 访问。"""
    raw = Path(value)
    if not raw.is_absolute():
        if base_dir is None:
            base_dir = Path.cwd()
        raw = (base_dir / raw).resolve()
    return str(raw)


def _normalise_entry(entry: Dict[str, Any], *, base_dir: Path | None = None) -> Dict[str, Any]:
    """深拷贝任务条目并解析 sandbox/mounts 等路径。"""
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
    entries = load_task_entries(path)
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
    "load_task_entries",
    "register_dataset_from_file",
    "ensure_dataset_registered",
]
