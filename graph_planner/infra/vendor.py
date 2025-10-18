"""Helpers for wiring third-party dependencies bundled as submodules."""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Set


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _expand(path: Path) -> Path:
    try:
        return path.expanduser().resolve()
    except Exception:  # pragma: no cover - defensive
        return path


def _iter_candidate_roots() -> Iterator[Path]:
    """Return deterministic locations that may contain the vendored ``rllm``."""

    env_hint = os.environ.get("GRAPH_PLANNER_RLLM_PATH")
    if env_hint:
        for chunk in env_hint.split(os.pathsep):
            chunk = chunk.strip()
            if chunk:
                yield _expand(Path(chunk))

    root = _repo_root()

    # The repository vendors rLLM as a git submodule under ``rllm/`` by default.
    default_submodule = root / "rllm"
    if default_submodule.exists():
        yield _expand(default_submodule)

    # Fallback: allow placing the package next to the repository root for
    # development workflows where rLLM is checked out separately.
    parent_rllm = root.parent / "rllm"
    if parent_rllm.exists():
        yield _expand(parent_rllm)

    yield root


def _iter_possible_sys_paths(base: Path) -> Iterator[Path]:
    if not base.exists() or not base.is_dir():
        return

    yield base

    try:
        for child in base.iterdir():
            name = child.name.lower()
            if "rllm" in name:
                yield child
            elif name in {"src", "source"}:
                try:
                    for inner in child.iterdir():
                        if "rllm" in inner.name.lower():
                            yield inner
                except Exception:  # pragma: no cover - filesystem race
                    continue
    except Exception:  # pragma: no cover - filesystem race
        return


def _normalise_path(path: Path) -> Path | None:
    try:
        resolved = path.expanduser().resolve()
    except Exception:  # pragma: no cover - defensive
        return None
    if (resolved / "rllm" / "__init__.py").is_file():
        return resolved
    if (resolved / "rllm" / "__init__.pyi").is_file():  # pragma: no cover - mypy stub
        return resolved
    src = resolved / "src"
    if (src / "rllm" / "__init__.py").is_file():
        return src
    return None


def _has_rllm_modules() -> bool:
    """Return True if both ``rllm`` and its agent submodules are resolvable."""

    try:
        base_spec = importlib.util.find_spec("rllm")
        if base_spec is None:
            return False
        agent_spec = importlib.util.find_spec("rllm.agents.agent")
        env_spec = importlib.util.find_spec("rllm.environments.base.base_env")
    except ModuleNotFoundError:  # pragma: no cover - defensive guard
        return False
    return agent_spec is not None and env_spec is not None


@lru_cache(maxsize=1)
def ensure_rllm_importable() -> bool:
    """Ensure the bundled rLLM repository is importable."""

    if _has_rllm_modules():
        return True

    visited: Set[Path] = set()
    for root in _iter_candidate_roots():
        for candidate in _iter_possible_sys_paths(root):
            normalized = _normalise_path(candidate)
            if normalized is None or normalized in visited:
                continue
            visited.add(normalized)
            sys.path.insert(0, str(normalized))
            for key in list(sys.modules):
                if key == "rllm" or key.startswith("rllm."):
                    sys.modules.pop(key, None)
            importlib.invalidate_caches()
            if _has_rllm_modules():
                return True
    return False


__all__ = ["ensure_rllm_importable"]
