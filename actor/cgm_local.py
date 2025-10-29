"""Local (non-RPC) CGM patch generator used as fallback by CGMService.
   NOTE: This file MUST NOT perform any Ray RPC; it must stay purely local."""

from __future__ import annotations
from typing import Any, Mapping, Dict

# Try the legacy in-repo implementation first
_impl = None
try:
    # If present in this repo
    from graph_planner.agents.rule_based import cgm_adapter as _impl  # type: ignore
except Exception:
    pass

if _impl is None:
    try:
        # Fallback to a top-level cgm_adapter.py shipped in this repo
        import cgm_adapter as _impl  # type: ignore
    except Exception:
        _impl = None  # type: ignore


def _to_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj  # type: ignore[return-value]
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()  # type: ignore[return-value]
        except Exception:
            pass
    return {"summary": "cgm-invalid-patch", "edits": []}


def generate(
    *,
    subgraph_linearized: Any,
    plan: Any,
    constraints: Mapping[str, Any] | None = None,
    snippets: Any = None,
    plan_text: str | None = None,
    issue: Any = None,
) -> Dict[str, Any]:
    """Return a patch dict using only local logic (no Ray/vLLM)."""
    if _impl is None:
        # Last-resort empty patch to keep pipeline alive
        return {"summary": "cgm-disabled", "edits": []}

    try:
        patch = _impl.generate(
            subgraph_linearized=subgraph_linearized,
            plan=plan,
            constraints=dict(constraints or {}),
            snippets=snippets,
            plan_text=plan_text,
            issue=issue,
        )
        return _to_dict(patch)
    except Exception:
        # Defensive: never crash the training loop
        return {"summary": "cgm-exception", "edits": []}


__all__ = ["generate"]
