"""Runtime package bootstrap helpers."""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_r2e_on_path() -> None:
    """Ensure the vendored R2E-Gym package is importable.

    The repository includes the upstream ``R2E-Gym`` sources as a sibling
    directory.  Python does not automatically add this location to
    ``sys.path`` which means importing :mod:`r2egym` would fail unless the
    caller tweaks ``PYTHONPATH`` manually.  Centralising the path adjustment
    here keeps all call sites zero-config: as soon as anything imports the
    :mod:`runtime` package we make the modules visible.
    """

    repo_root = Path(__file__).resolve().parents[1]
    candidate = repo_root / "R2E-Gym" / "src"
    if candidate.is_dir():
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


_ensure_r2e_on_path()

__all__ = []
