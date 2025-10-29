from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd

# unify parquet writer: pyarrow first, else fastparquet
_PARQUET_ENGINE = "pyarrow"
try:
    import pyarrow  # noqa: F401
except Exception:  # pragma: no cover - optional dependency
    _PARQUET_ENGINE = "fastparquet"

DATASETS_ROOT = Path("rllm/rllm/data/datasets").resolve()

_SANDBOX_KEYS = {
    "backend",
    "docker_image",
    "mounts",
    "workdir",
    "env",
    "r2e_ds_json",
    "repo_name",
    "issue_id",
    "issue_title",
}

GRAPH_PLANNER_DATASET_NAME = "graph_planner_repoenv"
GRAPH_PLANNER_CGM_DATASET_NAME = "graph_planner_cgm"


@dataclass
class RegisteredDataset:
    name: str
    split: str
    src_path: Path
    out_parquet: Path
    num_rows: int

    def get_verl_data_path(self) -> str:
        return str(self.out_parquet)

    # Compatibility helpers matching the legacy Dataset API used by scripts/tests.
    def get_data_path(self) -> str:
        return str(self.out_parquet)


def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _as_str_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    # JSON-serializable, with stable types
    def _clean(v):
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        if isinstance(v, (list, tuple)):
            return [_clean(x) for x in v]
        if isinstance(v, dict):
            return {str(k): _clean(v) for k, v in v.items()}
        return str(v)

    return {str(k): _clean(v) for k, v in d.items()}


def _infer_split_from_path(p: Union[str, Path]) -> str:
    s = str(p).lower()
    if re.search(r"(val|valid)", s):
        return "val"
    if "test" in s:
        return "val"  # treat test as val by default
    return "train"


def _load_jsonl(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def _load_json(path: Path) -> pd.DataFrame:
    obj = json.loads(Path(path).read_text())
    if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
        return pd.DataFrame(obj["data"])
    if isinstance(obj, list):
        return pd.DataFrame(obj)
    # fallback: wrap
    return pd.DataFrame([obj])


def _maybe_make_prompt(row: dict) -> str:
    # Prefer explicit fields
    for k in ["prompt", "instruction", "query", "task", "message", "input"]:
        if isinstance(row.get(k), str) and row[k].strip():
            return row[k]
    # Construct a stable prompt from repo/issue
    repo = row.get("repo") or row.get("repo_name") or row.get("repository")
    issue_id = row.get("issue_id") or row.get("id")
    title = row.get("issue_title") or row.get("title")
    parts = ["[CODE-REPAIR]"]
    if repo:
        parts.append(f"repo={repo}")
    if issue_id:
        parts.append(f"issue_id={issue_id}")
    if title:
        parts.append(f"title={title}")
    return "\n".join(parts)


def _extract_sandbox(row: dict) -> Dict[str, Any]:
    # Accept both flattened "sandbox.*" fields or nested dict under "sandbox"
    sb = {}
    if isinstance(row.get("sandbox"), dict):
        for k, v in row["sandbox"].items():
            if k in _SANDBOX_KEYS:
                sb[k] = v
    # flattened fallbacks
    for k in list(row.keys()):
        if k.startswith("sandbox."):
            sb[k.split(".", 1)[1]] = row[k]
        elif k in _SANDBOX_KEYS and k not in sb:
            sb[k] = row[k]

    # normalize typical fields
    sb.setdefault("backend", "docker")
    # mounts: dict[str, str]
    m = sb.get("mounts", {})
    if isinstance(m, list):
        # allow list of "src:dst"
        md = {}
        for it in m:
            if isinstance(it, str) and ":" in it:
                src, dst = it.split(":", 1)
                md[src] = dst
        sb["mounts"] = md
    elif not isinstance(m, dict):
        sb["mounts"] = {}
    # env: dict[str,str]
    if not isinstance(sb.get("env"), dict):
        sb["env"] = {}
    # workdir
    if not isinstance(sb.get("workdir"), str):
        sb["workdir"] = "/workspace"
    return sb


def _normalize_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = {
        "prompt": [],
        "extra_info": [],  # JSON string (stable across parquet engines)
    }

    for _, r in df.iterrows():
        row = r.to_dict()
        prompt = _maybe_make_prompt(row)

        sb = _extract_sandbox(row)
        extra = {
            "sandbox": sb,
            # pass-through useful identifiers (optional)
            "repo": row.get("repo") or row.get("repo_name"),
            "issue_id": row.get("issue_id") or row.get("id"),
            "issue_title": row.get("issue_title") or row.get("title"),
        }
        out["prompt"].append(prompt)
        out["extra_info"].append(json.dumps(_as_str_dict(extra), ensure_ascii=False))
    return pd.DataFrame(out)


def ensure_dataset_registered(
    name: str = GRAPH_PLANNER_DATASET_NAME,
    path: Union[str, Path, None] = None,
    split: Optional[str] = None,
) -> RegisteredDataset:
    """Materialize a *_verl.parquet dataset from JSON/JSONL/Parquet and return its path."""

    if path is None:
        raise ValueError("path must be provided when registering a dataset")

    src = Path(path).resolve()
    if split is None:
        split = _infer_split_from_path(src)

    # load
    if src.suffix.lower() in [".jsonl", ".jl"]:
        df = _load_jsonl(src)
    elif src.suffix.lower() == ".json":
        df = _load_json(src)
    elif src.suffix.lower() in [".parquet", ".pq"]:
        # already parquet: still normalize to guarantee schema/prompt/extra_info
        df = pd.read_parquet(src)
    else:
        raise ValueError(f"Unsupported dataset suffix: {src.suffix}")

    norm = _normalize_rows(df)

    out_dir = DATASETS_ROOT / name
    out_path = out_dir / f"{split}_verl.parquet"
    _ensure_dir(out_path)
    norm.to_parquet(out_path, index=False, engine=_PARQUET_ENGINE)

    return RegisteredDataset(
        name=name,
        split=split,
        src_path=src,
        out_parquet=out_path,
        num_rows=len(norm),
    )


# ---------------------------------------------------------------------------
# Legacy helper compatibility
# ---------------------------------------------------------------------------


def resolve_task_file(
    path: str | os.PathLike[str],
    *,
    split: str | None = None,
) -> Path:
    """Resolve dataset file path. Simplified compatibility shim."""

    candidate = Path(path).expanduser()
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"Task descriptor not found: {path}")


def load_task_entries(path: str | os.PathLike[str]) -> List[Dict[str, Any]]:
    resolved = resolve_task_file(path)
    if resolved.suffix.lower() in {".jsonl", ".jl"}:
        df = _load_jsonl(resolved)
    elif resolved.suffix.lower() == ".json":
        df = _load_json(resolved)
    elif resolved.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(resolved)
    else:
        raise ValueError(f"Unsupported dataset suffix: {resolved.suffix}")
    return df.to_dict(orient="records")


def register_dataset_from_file(
    *,
    name: str = GRAPH_PLANNER_DATASET_NAME,
    split: str = "train",
    path: str | os.PathLike[str],
) -> RegisteredDataset:
    return ensure_dataset_registered(name=name, split=split, path=path)


__all__ = [
    "GRAPH_PLANNER_DATASET_NAME",
    "GRAPH_PLANNER_CGM_DATASET_NAME",
    "RegisteredDataset",
    "ensure_dataset_registered",
    "resolve_task_file",
    "load_task_entries",
    "register_dataset_from_file",
]

