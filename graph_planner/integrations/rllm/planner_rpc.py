from __future__ import annotations

from typing import List, Optional

try:
    import ray
except Exception:  # ray optional in non-training contexts
    ray = None  # type: ignore


def generate(prompts: List[str], fallback_engine=None, **kw) -> List[str]:
    """
    Preferred: call the shared Ray actor 'planner_engine'.
    Fallback: use the provided local engine (vLLM/HF wrapper) with the same signature.
    Returns a plain list[str] (texts).
    """
    actor = None
    if ray is not None and getattr(ray, "is_initialized", lambda: False)():
        try:
            actor = ray.get_actor("planner_engine")
        except Exception:
            actor = None

    if actor is not None:
        return ray.get(actor.generate.remote(prompts, **kw))

    if fallback_engine is None:
        raise RuntimeError("No shared planner actor and no fallback engine available.")

    outs = fallback_engine.generate(prompts, **kw)
    if outs and hasattr(outs[0], "outputs"):
        return [o.outputs[0].text for o in outs]
    return outs
