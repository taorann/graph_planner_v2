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


JSON_SUFFIXES = {".json", ".jsonl"}
PREFERRED_INSTANCE_FILENAMES = (
    "metadata.jsonl",
    "metadata.json",
    "instance.jsonl",
    "instance.json",
    "task.jsonl",
    "task.json",
)


SPLIT_ALIASES: dict[str, list[str]] = {
    "validation": ["val", "dev"],
    "val": ["validation", "dev"],
    "dev": ["validation", "val"],
    "train": ["training"],
    "training": ["train"],
    "test": ["evaluation", "eval"],
    "evaluation": ["test", "eval"],
    "eval": ["test", "evaluation"],
}


def _discover_available_descriptors(base_dirs: Iterable[Path]) -> dict[str, Path]:
    """Return a mapping of available dataset stems to their source paths."""

    available: dict[str, Path] = {}
    for directory in base_dirs:
        for candidate in sorted(directory.glob("*.json*")):
            if candidate.is_file() and candidate.suffix in JSON_SUFFIXES:
                available.setdefault(candidate.stem, candidate)

        instances_root = directory / "instances"
        if not instances_root.is_dir():
            continue

        for entry in sorted(instances_root.iterdir()):
            if entry.is_dir():
                available.setdefault(entry.name, entry)
            elif entry.is_file() and entry.suffix in JSON_SUFFIXES:
                available.setdefault(entry.stem, entry)

    return available


def _iter_instance_sources(instance_dir: Path) -> List[Path]:
    sources: List[Path] = []
    seen: set[Path] = set()

    for candidate in sorted(instance_dir.iterdir()):
        if candidate.is_file() and candidate.suffix in JSON_SUFFIXES:
            if candidate not in seen:
                sources.append(candidate)
                seen.add(candidate)
            continue

        if not candidate.is_dir():
            continue

        chosen: List[Path] = []
        for filename in PREFERRED_INSTANCE_FILENAMES:
            preferred = candidate / filename
            if preferred.exists():
                chosen.append(preferred)
                break

        if not chosen:
            chosen.extend(
                sub
                for sub in sorted(candidate.glob("*.json*"))
                if sub.is_file() and sub.suffix in JSON_SUFFIXES
            )

        for entry in chosen:
            if entry not in seen:
                sources.append(entry)
                seen.add(entry)

    return sources


def _materialise_instance_split(base_dir: Path, instance_dir: Path, stem: str) -> Path:
    """Materialise a dataset split from SWE-bench ``instances`` directories."""

    safe_stem = stem.replace(os.sep, "_")
    cache_path = base_dir / f".auto_{safe_stem}.jsonl"

    sources = _iter_instance_sources(instance_dir)
    if not sources:
        raise FileNotFoundError(
            f"No JSON/JSONL files found in SWE-bench instances directory: {instance_dir}"
        )

    def _should_refresh() -> bool:
        if not cache_path.exists():
            return True
        cache_mtime = cache_path.stat().st_mtime
        for candidate in sources:
            if candidate.stat().st_mtime > cache_mtime:
                return True
        return False

    if _should_refresh():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as handle:
            for file_path in sources:
                if file_path.suffix == ".jsonl":
                    with file_path.open("r", encoding="utf-8") as src:
                        for line in src:
                            line = line.strip()
                            if not line:
                                continue
                            handle.write(line)
                            handle.write("\n")
                else:
                    with file_path.open("r", encoding="utf-8") as src:
                        payload = json.load(src)
                    if isinstance(payload, list):
                        for row in payload:
                            handle.write(json.dumps(row, ensure_ascii=False))
                            handle.write("\n")
                    else:
                        handle.write(json.dumps(payload, ensure_ascii=False))
                        handle.write("\n")
        LOGGER.info(
            "Materialised dataset split '%s' from %s into %s (%d sources)",
            stem,
            instance_dir,
            cache_path,
            len(sources),
        )

    return cache_path


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

        split_lower = split.lower()
        stems.extend(SPLIT_ALIASES.get(split_lower, []))
    stems = _unique(stems)

    for directory in base_dirs:
        for stem in stems:
            for suffix in suffixes:
                candidate = directory / f"{stem}{suffix}" if suffix else directory / stem
                add_candidate(candidate)

        instances_root = directory / "instances"
        if instances_root.is_dir():
            for stem in stems:
                for suffix in suffixes:
                    inst_candidate = (
                        instances_root / f"{stem}{suffix}"
                        if suffix
                        else instances_root / stem
                    )
                    if inst_candidate.is_file():
                        add_candidate(inst_candidate)
                instance_dir = instances_root / stem
                if instance_dir.is_dir():
                    try:
                        materialised = _materialise_instance_split(directory, instance_dir, stem)
                    except FileNotFoundError:
                        continue
                    add_candidate(materialised)

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
    available = _discover_available_descriptors(base_dirs)

    message_parts = [
        f"Task descriptor not found: {original}. Checked candidates:\n  - {attempted_str}",
    ]

    if available:
        available_list = "\n  - ".join(str(path) for path in list(available.values())[:8])
        message_parts.append(f"Available dataset descriptors:\n  - {available_list}")

        if split:
            split_lower = split.lower()
            available_keys = {key.lower(): key for key in available.keys()}
            if split_lower not in available_keys:
                # Prefer a matching alias if present, otherwise highlight the first candidate.
                alias_candidates = [split_lower]
                alias_candidates.extend(SPLIT_ALIASES.get(split_lower, []))
                suggestion_key = None
                for alias in alias_candidates:
                    if alias in available_keys:
                        suggestion_key = available_keys[alias]
                        break

                if suggestion_key is None and "test" in available_keys:
                    suggestion_key = available_keys["test"]

                if suggestion_key is None and available:
                    suggestion_key = next(iter(available.keys()))

                if suggestion_key:
                    suggestion_path = available[suggestion_key]
                    message_parts.append(
                        "Suggestion: update --dataset/--dataset-split to use "
                        f"'{suggestion_key}' (e.g. {suggestion_path})."
                    )

    raise FileNotFoundError("\n".join(message_parts))


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
