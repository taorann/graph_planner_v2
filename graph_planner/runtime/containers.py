"""Helpers for collecting and pre-pulling dataset container images.

This module centralises the logic for discovering docker images from
Graph Planner datasets and invoking the upstream R2E-Gym pre-pull
utilities so training runs can reuse warm containers.  All functions are
side-effect free except :func:`prepull_docker_images`, which delegates to
the R2E helpers when available.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Mapping, MutableMapping, Optional, Sequence

from . import _ensure_r2e_on_path

LOGGER = logging.getLogger(__name__)


@dataclass
class DockerImageCollection:
    """Represents docker images discovered from dataset metadata."""

    images: List[str]
    missing: int = 0
    inspected: int = 0
    build_only: int = 0


@dataclass
class ContainerPrepResult:
    """Return payload emitted by :func:`prepull_docker_images`."""

    images: List[str]
    invoked: bool


def _extract_image(payload: Mapping[str, object] | None) -> Optional[tuple[str, bool]]:
    if not isinstance(payload, Mapping):
        return None

    candidate = None
    requires_build = False
    sandbox = payload.get("sandbox")
    if isinstance(sandbox, Mapping):
        requires_build = bool(sandbox.get("requires_build"))
        candidate = sandbox.get("docker_image")
    if not candidate:
        candidate = payload.get("docker_image")
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip(), requires_build
    return None


def _load_json(path: Path) -> Optional[MutableMapping[str, object]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        LOGGER.warning("Instance file %s was missing when collecting docker images", path)
    except json.JSONDecodeError as exc:
        LOGGER.warning("Failed to decode %s: %s", path, exc)
    return None


def collect_docker_images(
    *,
    records: Iterable[Mapping[str, object]] | None = None,
    instance_paths: Iterable[Path] | None = None,
    jsonl_paths: Iterable[Path] | None = None,
) -> DockerImageCollection:
    """Collect unique docker images from dataset artefacts.

    Parameters
    ----------
    records:
        Parsed dataset rows, typically returned by ``load_task_entries`` or
        conversion utilities.
    instance_paths:
        Paths to per-instance JSON files that may contain docker metadata.
    jsonl_paths:
        Dataset JSONL files to scan when ``records`` are unavailable.

    Returns
    -------
    DockerImageCollection
        Unique docker images plus bookkeeping counts used for diagnostics.
    """

    seen: set[str] = set()
    images: List[str] = []
    missing = 0
    build_only = 0
    inspected = 0

    def _register(value: Optional[tuple[str, bool]]) -> None:
        nonlocal missing, build_only
        if not value:
            missing += 1
            return
        image, requires_build = value
        if requires_build:
            build_only += 1
            return
        if image not in seen:
            seen.add(image)
            images.append(image)

    if records is not None:
        for row in records:
            inspected += 1
            _register(_extract_image(row))

    if instance_paths is not None:
        for path in instance_paths:
            payload = _load_json(path)
            if payload is None:
                continue
            inspected += 1
            _register(_extract_image(payload))

    if jsonl_paths is not None:
        for jsonl in jsonl_paths:
            if not jsonl.exists():
                continue
            with jsonl.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    inspected += 1
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError as exc:
                        LOGGER.warning("Skipping malformed JSON entry in %s: %s", jsonl, exc)
                        missing += 1
                        continue
                    _register(_extract_image(payload))

    return DockerImageCollection(images=images, missing=missing, inspected=inspected, build_only=build_only)


def write_docker_manifest(path: Path, images: Sequence[str]) -> Path:
    """Persist ``images`` to ``path`` in a newline-separated manifest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for image in images:
            handle.write(image)
            handle.write("\n")
    return path


def load_docker_manifest(path: Path) -> List[str]:
    """Load a newline-separated manifest of docker images."""

    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _resolve_pre_pull() -> Callable[..., None]:
    """Resolve the upstream pre-pull helper from R2E-Gym."""

    _ensure_r2e_on_path()

    try:
        from r2egym.repo_analysis.validate_docker_and_hf import pre_pull_docker_images as repo_pre_pull

        def _call_repo(images: Sequence[str], **kwargs: object) -> None:
            params = {k: v for k, v in kwargs.items() if k in {"max_workers", "retries", "delay", "pull_timeout"} and v is not None}
            repo_pre_pull(list(images), **params)  # type: ignore[arg-type]

        return _call_repo
    except Exception:  # pragma: no cover - fallback path
        pass

    try:
        from r2egym.agenthub.run.edit import prepull_docker_images as agent_pre_pull

        def _call_agent(images: Sequence[str], **kwargs: object) -> None:
            params = {k: v for k, v in kwargs.items() if k == "max_workers" and v is not None}
            dataset_rows = [{"docker_image": image} for image in images]
            agent_pre_pull(dataset_rows, **params)

        return _call_agent
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError("Unable to import R2E docker helpers") from exc


def prepull_docker_images(
    images: Sequence[str],
    *,
    max_workers: Optional[int] = None,
    retries: Optional[int] = None,
    delay: Optional[int] = None,
    pull_timeout: Optional[int] = None,
) -> ContainerPrepResult:
    """Pre-pull docker images using the R2E-Gym helpers.

    Parameters
    ----------
    images:
        Docker images to pre-pull.
    max_workers, retries, delay, pull_timeout:
        Optional overrides forwarded to the R2E helper when supported.
    """

    unique = []
    seen: set[str] = set()
    for image in images:
        if image and image not in seen:
            seen.add(image)
            unique.append(image)

    if not unique:
        return ContainerPrepResult(images=[], invoked=False)

    helper = _resolve_pre_pull()
    helper(unique, max_workers=max_workers, retries=retries, delay=delay, pull_timeout=pull_timeout)
    return ContainerPrepResult(images=unique, invoked=True)


__all__ = [
    "ContainerPrepResult",
    "DockerImageCollection",
    "collect_docker_images",
    "load_docker_manifest",
    "prepull_docker_images",
    "write_docker_manifest",
]

