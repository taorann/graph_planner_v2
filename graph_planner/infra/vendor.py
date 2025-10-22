"""Helpers for wiring third-party dependencies bundled as submodules."""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from functools import lru_cache
from types import ModuleType
from pathlib import Path
from typing import Iterator, Set, Tuple

# Namespace packages (PEP 420) do not expose ``__file__`` which means callers
# cannot simply rely on ``Path(pkg.__file__)`` to locate resources such as YAML
# configs.  We therefore keep track of the resolved search roots once the rLLM
# package has been imported successfully.
_RLLM_PACKAGE_ROOTS: Tuple[Path, ...] = ()


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
            if "rllm" in name or "verl" in name:
                yield child
            elif name in {"src", "source"}:
                try:
                    for inner in child.iterdir():
                        lower = inner.name.lower()
                        if "rllm" in lower or "verl" in lower:
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
    verl_pkg = resolved / "verl"
    if (verl_pkg / "__init__.py").is_file():
        return verl_pkg
    src_verl = src / "verl"
    if (src_verl / "__init__.py").is_file():
        return src_verl
    return None


def _has_rllm_modules() -> bool:
    """Return True if both ``rllm`` and its agent submodules are resolvable."""

    try:
        base_spec = importlib.util.find_spec("rllm")
        if base_spec is None:
            return False
        agent_spec = None
        env_spec = None
        for candidate in ("rllm.rllm.agents.agent", "rllm.agents.agent"):
            try:
                agent_spec = importlib.util.find_spec(candidate)
            except ModuleNotFoundError:
                continue
            if agent_spec is not None:
                break
        for candidate in ("rllm.rllm.environments.base.base_env", "rllm.environments.base.base_env"):
            try:
                env_spec = importlib.util.find_spec(candidate)
            except ModuleNotFoundError:
                continue
            if env_spec is not None:
                break
    except ModuleNotFoundError:  # pragma: no cover - defensive guard
        return False
    verl_core = _repo_root() / "rllm" / "verl" / "verl" / "trainer" / "ppo" / "core_algos.py"
    return agent_spec is not None and env_spec is not None and verl_core.exists()


def _ensure_verl_stub() -> None:
    """Register a lightweight ``verl`` namespace if the package is absent."""

    try:
        spec = importlib.util.find_spec("verl.trainer.ppo.core_algos")
    except ModuleNotFoundError:
        spec = None
    if spec is not None:
        return

    root = _repo_root()
    candidate = root / "rllm" / "verl" / "verl"
    if not (candidate / "trainer").exists():
        return

    stub = ModuleType("verl")
    stub.__path__ = [str(candidate)]  # type: ignore[attr-defined]
    sys.modules["verl"] = stub


def _ensure_tensordict_stub() -> None:
    """Provide a minimal ``tensordict`` placeholder required by Verl."""

    if "tensordict" in sys.modules:
        return
    try:
        if importlib.util.find_spec("tensordict") is not None:
            return
    except ModuleNotFoundError:
        pass

    class _TensorDict(dict):
        def __init__(self, source=None, batch_size=None, **kwargs):
            super().__init__(source or {})
            self.batch_size = batch_size or (1,)

    stub = ModuleType("tensordict")
    stub.TensorDict = _TensorDict  # type: ignore[attr-defined]
    sys.modules["tensordict"] = stub


def _capture_rllm_package_roots() -> None:
    """Discover and cache the on-disk roots for the ``rllm`` package."""

    global _RLLM_PACKAGE_ROOTS
    if _RLLM_PACKAGE_ROOTS:
        return

    try:
        import importlib

        pkg = importlib.import_module("rllm")
    except ModuleNotFoundError:  # pragma: no cover - defensive guard
        return

    roots: list[Path] = []
    for entry in getattr(pkg, "__path__", []):
        path = Path(entry).resolve()
        if not path.exists():
            continue
        roots.append(path)

        nested = path / "rllm"
        if nested.exists():
            roots.append(nested)

    if roots:
        # Preserve order while removing duplicates.
        deduped = []
        seen: Set[Path] = set()
        for candidate in roots:
            if candidate in seen:
                continue
            seen.add(candidate)
            deduped.append(candidate)
        _RLLM_PACKAGE_ROOTS = tuple(deduped)


@lru_cache(maxsize=1)
def ensure_rllm_importable() -> bool:
    """Ensure the bundled rLLM repository is importable."""

    if _has_rllm_modules():
        _ensure_verl_stub()
        _ensure_tensordict_stub()
        _capture_rllm_package_roots()
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
                _ensure_verl_stub()
                _ensure_tensordict_stub()
                _capture_rllm_package_roots()
                return True
    return False


def iter_rllm_package_roots() -> Tuple[Path, ...]:
    """Return cached package roots for the vendored ``rllm`` module."""

    ensure_rllm_importable()
    return _RLLM_PACKAGE_ROOTS


def find_in_rllm(*segments: str) -> Path:
    """Locate a path relative to any discovered rLLM package root."""

    for root in iter_rllm_package_roots():
        candidate = root.joinpath(*segments)
        if candidate.exists():
            try:
                return candidate.resolve()
            except OSError:  # pragma: no cover - filesystem permission issue
                return candidate
    raise FileNotFoundError(
        "Unable to locate rLLM resource: " + "/".join(segments)
    )


__all__ = ["ensure_rllm_importable", "iter_rllm_package_roots", "find_in_rllm"]
