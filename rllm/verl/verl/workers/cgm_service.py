"""Concurrent CGM service wrapper around a vLLM engine."""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from typing import Any, Dict, Iterable, Mapping

import ray

try:  # pragma: no cover - optional local fallback
    from aci.schema import Plan
except Exception:  # pragma: no cover - safety net when ACI not installed
    Plan = None  # type: ignore[assignment]

try:  # pragma: no cover - local CGM fallback helper (MUST be local, NOT RPC)
    from actor import cgm_local as _local_cgm_adapter
except Exception:  # pragma: no cover - optional dependency outside planner
    _local_cgm_adapter = None  # type: ignore[assignment]


class _LocalCGMEngine:
    """Fallback engine that mirrors the local CGM adapter implementation."""

    def __init__(self) -> None:
        self.group = "cgm-local"

    def _ensure_plan(self, plan_payload: Any, plan_struct: Any) -> Any:
        if plan_struct is not None:
            payload = plan_struct
        else:
            payload = plan_payload
        if Plan is None:
            return payload
        if isinstance(payload, Plan):
            return payload
        if isinstance(payload, Mapping):
            try:
                return Plan(**payload)
            except Exception:
                return payload
        return payload

    def _call_local_cgm(self, req: Mapping[str, Any]) -> Dict[str, Any]:
        if _local_cgm_adapter is None:
            return {"summary": "cgm-disabled", "edits": []}

        collated = req.get("collated") or {}
        plan = self._ensure_plan(req.get("plan"), req.get("plan_struct"))
        constraints = req.get("constraints") or {}
        snippets = collated.get("snippets")
        plan_text = req.get("plan_text") or req.get("plan")
        issue = req.get("issue")

        patch = _local_cgm_adapter.generate(
            subgraph_linearized=collated.get("chunks"),
            plan=plan,
            constraints=constraints,
            snippets=snippets,
            plan_text=plan_text,
            issue=issue,
        )

        if hasattr(patch, "to_dict"):
            return patch.to_dict()  # type: ignore[return-value]
        if isinstance(patch, Mapping):
            return dict(patch)
        return {"summary": "cgm-invalid-patch", "edits": []}

    def generate(self, prompts: Iterable[Any]) -> list[Dict[str, Any]]:
        responses: list[Dict[str, Any]] = []
        for prompt in prompts:
            if isinstance(prompt, Mapping):
                req = dict(prompt)
            elif isinstance(prompt, str):
                try:
                    req = json.loads(prompt)
                except json.JSONDecodeError:
                    req = {"plan": prompt, "collated": {}, "constraints": {}}
            else:
                req = {"plan": prompt, "collated": {}, "constraints": {}}
            responses.append(self._call_local_cgm(req))
        return responses


@ray.remote(max_concurrency=8)
class CGMService:
    """Aggregate concurrent patch requests into small vLLM batches."""

    def __init__(
        self,
        vllm_engine: Any,
        max_batch: int = 8,
        max_wait_ms: int = 10,
        vllm_cfg: dict | None = None,
    ) -> None:
        raw_cfg = vllm_cfg or {}
        if isinstance(raw_cfg, Mapping):
            raw_cfg = dict(raw_cfg)
        else:
            try:
                raw_cfg = dict(raw_cfg)
            except Exception:
                raw_cfg = {}
        self.vllm_cfg: dict[str, Any] = {str(k): raw_cfg[k] for k in raw_cfg}
        self._visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        try:
            self._ngpus = len([tok for tok in self._visible.split(",") if tok != ""])
        except Exception:
            self._ngpus = 1
        if self._ngpus <= 0:
            self._ngpus = 1
        candidate = vllm_engine
        if candidate is None or not hasattr(candidate, "generate") or (
            hasattr(candidate, "has_runtime") and not candidate.has_runtime()
        ):
            print("[cgm] Falling back to local CGM engine (no runtime attached)")
            self.engine = _LocalCGMEngine()
        else:
            self.engine = candidate
        self._apply_vllm_cfg()
        self.q: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self.max_batch = max_batch
        self.max_wait_ms = max_wait_ms
        self.lock = threading.Lock()
        self.batching = False

    def _apply_vllm_cfg(self) -> None:
        cfg = self.vllm_cfg
        tp_default = max(1, self._ngpus)

        def _as_int(key: str, default: int) -> int:
            value = cfg.get(key, default)
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        def _as_float(key: str, default: float) -> float:
            value = cfg.get(key, default)
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        tp = _as_int("tp", tp_default)
        gmu = _as_float("gpu_memory_utilization", 0.60)
        mml = _as_int("max_model_len", 8192)
        mns = _as_int("max_num_seqs", 2)
        kvd = str(cfg.get("kv_cache_dtype", "fp16"))
        eager = bool(cfg.get("enforce_eager", True))

        print(
            f"[cgm/vllm] tp={tp} gmu={gmu} max_model_len={mml} max_num_seqs={mns} "
            f"kv_cache_dtype={kvd} eager={eager} vis={self._visible}"
        )

        engine = getattr(self, "engine", None)
        if engine is None:
            return

        attr_map = {
            "tp": tp,
            "tensor_parallel_size": tp,
            "gpu_memory_utilization": gmu,
            "max_model_len": mml,
            "max_num_seqs": mns,
            "kv_cache_dtype": kvd,
            "enforce_eager": eager,
        }
        for attr, value in attr_map.items():
            if hasattr(engine, attr):
                try:
                    setattr(engine, attr, value)
                except Exception:
                    continue
        if hasattr(engine, "set_max_num_seqs"):
            try:
                engine.set_max_num_seqs(mns)
            except Exception:
                pass

    def generate_patch(self, req: Dict[str, Any]) -> Dict[str, Any]:
        ev = threading.Event()
        slot: Dict[str, Any] = {"req": req, "ev": ev, "out": None, "err": None}
        self.q.put(slot)
        self._kick()
        ev.wait()
        if slot["err"] is not None:
            raise slot["err"]
        return slot["out"]

    def _kick(self) -> None:
        with self.lock:
            if self.batching:
                return
            self.batching = True
        # Inline execution keeps batching on the actor thread while preserving the
        # structure that would allow async submission in the future if desired.
        self._batch_once()

    def _batch_once(self) -> None:
        try:
            batch = []
            try:
                first = self.q.get(timeout=0.001)
                batch.append(first)
            except queue.Empty:
                return

            deadline = time.time() + self.max_wait_ms / 1000.0
            while len(batch) < self.max_batch and time.time() < deadline:
                remaining = max(0.0, deadline - time.time())
                if remaining <= 0:
                    break
                try:
                    batch.append(self.q.get(timeout=remaining))
                except queue.Empty:
                    break

            print(f"[CGMService] batching={len(batch)} group={getattr(self.engine, 'group', 'cgm')}")
            prompts = [self._build_prompt(slot["req"]) for slot in batch]
            generations = self.engine.generate(prompts)
            patches = [self._parse_patch(gen) for gen in generations]
            for slot, patch in zip(batch, patches, strict=False):
                slot["out"] = patch
                slot["ev"].set()
        except Exception as exc:  # pragma: no cover - defensive path
            for slot in batch:
                slot["err"] = exc
                slot["ev"].set()
        finally:
            with self.lock:
                self.batching = False

    def _build_prompt(self, req: Dict[str, Any]) -> str:
        """Encode request as JSON string for vLLM; fallback will json.loads."""

        return json.dumps(req, ensure_ascii=False)

    def _parse_patch(self, payload: Any) -> Dict[str, Any]:
        """Parse CGM output payload into a structured patch dict."""

        if isinstance(payload, Mapping):
            return dict(payload)
        if hasattr(payload, "to_dict"):
            return payload.to_dict()  # type: ignore[return-value]
        if isinstance(payload, str):
            try:
                data = json.loads(payload)
                if isinstance(data, Mapping):
                    return dict(data)
            except json.JSONDecodeError:
                pass
        return {"summary": "cgm-invalid-patch", "edits": []}

