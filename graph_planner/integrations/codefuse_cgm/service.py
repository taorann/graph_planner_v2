"""HTTP service exposing local CodeFuse-CGM generation."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from aci.schema import Plan, PlanTarget

from ...agents.rule_based.cgm_adapter import _LocalCGMRuntime
from .inference import CGMGenerationConfig, CodeFuseCGMGenerator

LOGGER = logging.getLogger("graph_planner.cgm.service")


class GenerateRequest(BaseModel):
    """Request payload accepted by the CGM service."""

    issue: Optional[Dict[str, Any]] = None
    plan: Dict[str, Any]
    plan_text: Optional[str] = None
    subgraph: Optional[Sequence[Mapping[str, Any]]] = None
    snippets: Optional[Sequence[Mapping[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
    model_overrides: Optional[Dict[str, Any]] = Field(default=None, alias="model_config")

    model_config = ConfigDict(extra="allow")


@dataclass
class _RuntimeBundle:
    runtime: _LocalCGMRuntime
    lock: asyncio.Lock


def _plan_from_payload(payload: Mapping[str, Any]) -> Plan:
    if not isinstance(payload, Mapping):
        raise ValueError("plan must be an object")
    targets_raw = payload.get("targets")
    if not isinstance(targets_raw, Sequence) or not targets_raw:
        raise ValueError("plan.targets must be a non-empty array")

    targets: list[PlanTarget] = []
    for idx, entry in enumerate(targets_raw):
        if not isinstance(entry, Mapping):
            continue
        path = entry.get("path")
        start = entry.get("start")
        end = entry.get("end", start)
        if path is None or start is None:
            continue
        try:
            target = PlanTarget(
                path=str(path),
                start=int(start),
                end=int(end if end is not None else start),
                id=str(entry.get("id") or f"target-{idx}"),
                confidence=float(entry.get("confidence", 1.0)),
                why=str(entry.get("why", "")),
            )
        except Exception:
            continue
        targets.append(target)

    if not targets:
        raise ValueError("plan.targets produced no valid entries")

    budget = payload.get("budget") if isinstance(payload.get("budget"), Mapping) else {}
    priority = payload.get("priority_tests")
    if isinstance(priority, Sequence):
        priority_tests = [str(item) for item in priority]
    else:
        priority_tests = []

    return Plan(targets=targets, budget=dict(budget), priority_tests=priority_tests)


def _extract_constraints(metadata: Optional[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    if not isinstance(metadata, Mapping):
        return None
    constraints = metadata.get("constraints")
    if isinstance(constraints, Mapping):
        return constraints
    return None


def _apply_model_config(
    bundle: _RuntimeBundle, model_config: Optional[Mapping[str, Any]]
) -> Dict[str, Any]:
    if not isinstance(model_config, Mapping):
        return {}
    cfg = bundle.runtime.generator.config
    original: Dict[str, Any] = {}

    if "temperature" in model_config:
        original["temperature"] = cfg.temperature
        cfg.temperature = float(model_config["temperature"])
        original.setdefault("do_sample", cfg.do_sample)
        cfg.do_sample = cfg.temperature > 0
    if "top_p" in model_config:
        original["top_p"] = cfg.top_p
        cfg.top_p = float(model_config["top_p"])
    if "max_tokens" in model_config:
        original["max_new_tokens"] = cfg.max_new_tokens
        cfg.max_new_tokens = int(model_config["max_tokens"])
    if "num_return_sequences" in model_config:
        original["num_return_sequences"] = cfg.num_return_sequences
        cfg.num_return_sequences = int(model_config["num_return_sequences"])

    return original


def _restore_model_config(bundle: _RuntimeBundle, snapshot: Mapping[str, Any]) -> None:
    if not snapshot:
        return
    cfg = bundle.runtime.generator.config
    for key, value in snapshot.items():
        setattr(cfg, key, value)
    if "temperature" in snapshot and "do_sample" in snapshot:
        cfg.do_sample = bool(snapshot["do_sample"])


def create_app(bundle: _RuntimeBundle, *, route: str = "/generate") -> FastAPI:
    app = FastAPI()

    @app.get("/healthz")
    async def healthcheck() -> Dict[str, Any]:
        return {"ok": True}

    @app.post(route)
    async def generate(request: GenerateRequest) -> JSONResponse:
        try:
            plan = _plan_from_payload(request.plan)
        except ValueError as exc:  # pragma: no cover - defensive parsing
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        metadata = request.metadata or {}
        constraints = _extract_constraints(metadata)

        async with bundle.lock:
            overrides = _apply_model_config(bundle, request.model_overrides)
            try:
                patch = bundle.runtime.generate_patch(
                    issue=request.issue,
                    plan=plan,
                    plan_text=request.plan_text,
                    subgraph_linearized=request.subgraph,
                    snippets=request.snippets,
                    constraints=constraints,
                )
            except Exception as exc:  # pragma: no cover - runtime failure
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            finally:
                _restore_model_config(bundle, overrides)

        if not patch or not patch.get("edits"):
            raise HTTPException(status_code=502, detail="CGM produced no edits")

        response: Dict[str, Any] = {"patch": patch}
        summary = patch.get("summary")
        if summary:
            response["summary"] = summary
        if constraints:
            response.setdefault("metadata", {})["constraints"] = dict(constraints)
        return JSONResponse(response)

    return app


def _parse_device_map(raw: Optional[str]) -> Any:
    if raw is None:
        return None
    candidate = raw.strip()
    if not candidate:
        return None
    if candidate.lower() in {"auto", "balanced", "balanced_low_0", "sequential"}:
        return candidate
    try:
        return json.loads(candidate)
    except Exception:
        return candidate


def _build_generator(args: argparse.Namespace) -> CodeFuseCGMGenerator:
    config = CGMGenerationConfig(
        model_name_or_path=str(args.model),
        tokenizer_name_or_path=str(args.tokenizer or args.model),
        max_length=int(args.max_input_tokens),
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        do_sample=float(args.temperature) > 0,
        num_return_sequences=int(args.num_return_sequences),
        device=args.device,
        device_map=_parse_device_map(args.device_map),
        torch_dtype=args.dtype,
        load_in_8bit=bool(args.load_in_8bit),
        load_in_4bit=bool(args.load_in_4bit),
        trust_remote_code=bool(args.trust_remote_code),
        attn_implementation=args.attn_implementation,
    )
    return CodeFuseCGMGenerator(config)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path or identifier of the CGM checkpoint")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path (defaults to --model)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30001)
    parser.add_argument("--route", default="/generate", help="Route used for generation requests")
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--num-return-sequences", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--dtype", default=None, help="Optional torch dtype, e.g. float16")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--log-level", default="info")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="[%(levelname)s] %(message)s")
    LOGGER.info("Loading CGM model from %s", args.model)
    generator = _build_generator(args)
    runtime = _LocalCGMRuntime(generator=generator)
    bundle = _RuntimeBundle(runtime=runtime, lock=asyncio.Lock())
    app = create_app(bundle, route=args.route)
    LOGGER.info("Starting CGM service on %s:%s%s", args.host, args.port, args.route)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
