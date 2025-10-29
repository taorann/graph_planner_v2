"""Bridge between planner environments and CGM generation services."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

try:  # pragma: no cover - optional runtime dependency
    import ray
except Exception:  # pragma: no cover - ray may be optional when running locally
    ray = None  # type: ignore[assignment]

from graph_planner.agents.rule_based import cgm_adapter as _local_cgm_adapter


def _ray_available() -> bool:
    return ray is not None and getattr(ray, "is_initialized", lambda: False)()


def _resolve_actor(name: str):  # pragma: no cover - thin Ray helper
    if not _ray_available():
        return None
    try:
        return ray.get_actor(name)
    except Exception:
        return None


def generate(
    collated: Mapping[str, Any],
    plan: Any,
    constraints: Optional[Mapping[str, Any]],
    *,
    run_id: str,
    timeout_s: float = 60.0,
    retry: int = 1,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Generate a patch via the CGM service, falling back to the local stub."""

    plan_struct = kwargs.get("plan_struct")
    plan_text = kwargs.get("plan_text")
    issue = kwargs.get("issue")

    actor_name = f"CGMService::{run_id}" if run_id else ""
    actor = _resolve_actor(actor_name) if actor_name else None
    request = {
        "collated": collated,
        "plan": plan,
        "constraints": dict(constraints or {}),
        "plan_struct": plan_struct.to_dict() if hasattr(plan_struct, "to_dict") else plan_struct,
        "plan_text": plan_text,
    }

    if actor is not None:
        last_exc: Exception | None = None
        attempts = max(0, int(retry)) + 1
        for _ in range(attempts):
            fut = actor.generate_patch.remote(request)
            try:
                return ray.get(fut, timeout=timeout_s)
            except ray.exceptions.GetTimeoutError as exc:  # type: ignore[attr-defined]
                last_exc = exc
            except Exception:
                raise
        assert last_exc is not None
        raise last_exc

    # Fall back to the legacy/local CGM adapter when no remote service is present.
    plan_struct_obj = plan_struct
    if plan_struct_obj is None and hasattr(plan, "targets"):
        plan_struct_obj = plan
    if plan_struct_obj is None:
        from aci.schema import Plan  # lazy import to avoid circular dependencies

        plan_struct_obj = Plan(targets=[], budget={}, priority_tests=[])

    return _local_cgm_adapter.generate(
        subgraph_linearized=collated.get("chunks"),
        plan=plan_struct_obj,
        constraints=dict(constraints or {}),
        snippets=collated.get("snippets"),
        plan_text=plan_text,
        issue=issue,
    )


__all__ = ["generate"]
