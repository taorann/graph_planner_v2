"""Utilities for normalising and hashing CGM patch payloads."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Mapping, Sequence

from ..agents.common.contracts import normalize_newlines

__all__ = ["patch_id"]


def _iter_edits(obj: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    """Yield edit dictionaries from a CGM patch-like mapping."""

    if not isinstance(obj, Mapping):  # pragma: no cover - defensive
        return []

    if "edits" in obj and isinstance(obj.get("edits"), Sequence):
        edits = obj.get("edits")  # type: ignore[assignment]
        return [edit for edit in edits if isinstance(edit, Mapping)]

    patch = obj.get("patch") if isinstance(obj.get("patch"), Mapping) else None
    if patch and isinstance(patch.get("edits"), Sequence):
        return [edit for edit in patch["edits"] if isinstance(edit, Mapping)]

    return []


def patch_id(obj: Mapping[str, Any]) -> str:
    """Return a deterministic SHA256 fingerprint for a CGM patch.

    The hash is computed from the target path and each edit tuple
    ``(path, start, end, new_text)`` after normalising newline conventions.
    This allows the repair pipeline to detect duplicate applications across
    retries regardless of cosmetic whitespace or JSON ordering differences.
    """

    path = ""
    records: list[tuple[str, int, int, str]] = []
    for edit in _iter_edits(obj):
        try:
            edit_path = str(edit.get("path", "")).strip()
            start = int(edit.get("start", 0))
            end = int(edit.get("end", 0))
            new_text = normalize_newlines(str(edit.get("new_text", "")))
        except Exception:  # pragma: no cover - defensive
            continue
        if not edit_path:
            continue
        path = path or edit_path
        records.append((edit_path, start, end, new_text))

    if not records:
        payload = "".encode("utf-8")
    else:
        records.sort()
        payload = "\u241f".join(
            f"{p}\u241e{start}\u241e{end}\u241e{new}" for p, start, end, new in records
        ).encode("utf-8")

    if path:
        payload += f"\u2400{path}".encode("utf-8")

    return hashlib.sha256(payload).hexdigest()
