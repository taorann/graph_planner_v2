"""Bridge between planner environments and CGM generation services."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import uuid

try:  # pragma: no cover - optional runtime dependency
    import ray
except Exception:  # pragma: no cover - ray may be optional when running locally
    ray = None  # type: ignore[assignment]

from graph_planner.agents.rule_based import cgm_adapter as _local_cgm_adapter


def _ray_available() -> bool:
    return ray is not None and getattr(ray, "is_initialized", lambda: False)()


def _resolve_actor(name: str):  # pragma: no cover - thin Ray helper
    if not name or not _ray_available():
        return None
    try:
        return ray.get_actor(name)
    except Exception:
        return None


def _normalize_collated(
    collated: Mapping[str, Any] | None,
    *,
    subgraph_linearized: Any | None = None,
    snippets: Any | None = None,
) -> Dict[str, Any]:
    if collated is None:
        data: Dict[str, Any] = {}
        if subgraph_linearized is not None:
            data["chunks"] = subgraph_linearized
        if snippets is not None:
            data["snippets"] = snippets
        return data
    return dict(collated)


def generate(
    collated: Mapping[str, Any] | None = None,
    plan: Any | None = None,
    constraints: Optional[Mapping[str, Any]] = None,
    *,
    run_id: str | None = None,
    timeout_s: float = 60.0,
    retry: int = 1,
    plan_struct: Any | None = None,
    plan_text: Any | None = None,
    issue: Any | None = None,
    subgraph_linearized: Any | None = None,
    snippets: Any | None = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Generate a patch via the CGM service, falling back to the local stub."""

    # Preserve legacy keyword spellings.
    if plan is None and "plan" in kwargs:
        plan = kwargs["plan"]
    if constraints is None and "constraints" in kwargs:
        constraints = kwargs["constraints"]
    if plan_struct is None and "plan_struct" in kwargs:
        plan_struct = kwargs["plan_struct"]
    if plan_text is None and "plan_text" in kwargs:
        plan_text = kwargs["plan_text"]
    if issue is None and "issue" in kwargs:
        issue = kwargs["issue"]
    if snippets is None and "snippets" in kwargs:
        snippets = kwargs["snippets"]
    if subgraph_linearized is None and "subgraph_linearized" in kwargs:
        subgraph_linearized = kwargs["subgraph_linearized"]

    collated_payload = _normalize_collated(
        collated,
        subgraph_linearized=subgraph_linearized,
        snippets=snippets,
    )
    constraints_dict = dict(constraints or {})

    actor_name = f"CGMService::{run_id}" if run_id else ""
    actor = _resolve_actor(actor_name)
    if actor is not None:
        request = {
            "req_id": str(uuid.uuid4()),
            "collated": collated_payload,
            "plan": plan,
            "constraints": constraints_dict,
            "plan_struct": plan_struct.to_dict() if hasattr(plan_struct, "to_dict") else plan_struct,
            "plan_text": plan_text,
            "issue": issue,
        }
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
        if last_exc is not None:
            raise last_exc

    # Fall back to the legacy/local CGM adapter when no remote service is present.
    plan_struct_obj = plan_struct
    if plan_struct_obj is None and hasattr(plan, "targets"):
        plan_struct_obj = plan
    if plan_struct_obj is None:
        try:
            from aci.schema import Plan  # lazy import to avoid circular dependencies
        except Exception:  # pragma: no cover - optional dependency
            Plan = None  # type: ignore[assignment]
        if Plan is not None:
            plan_struct_obj = Plan(targets=[], budget={}, priority_tests=[])

    return _local_cgm_adapter.generate(
        subgraph_linearized=collated_payload.get("chunks"),
        plan=plan_struct_obj,
        constraints=constraints_dict,
        snippets=collated_payload.get("snippets"),
        plan_text=plan_text,
        issue=issue,
    )


__all__ = ["generate"]
