"""HTTP client wrapper for the CodeFuse CGM patch generator."""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional
from urllib import request, error

from aci.schema import Patch, Plan


class CodeFuseCGMError(RuntimeError):
    """Raised when the CodeFuse CGM service returns an error."""


def build_cgm_payload(
    *,
    issue: Optional[Dict[str, Any]],
    plan: Plan,
    plan_text: Optional[str],
    subgraph_linearized: Optional[Iterable[Dict[str, Any]]],
    snippets: Optional[Iterable[Dict[str, Any]]],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Package planner context for the CGM service.

    Parameters
    ----------
    issue:
        Issue metadata from the environment (ID, title, body, etc.).
    plan:
        Structured repair plan produced by the planner agent.
    plan_text:
        Natural language description of the repair plan.
    subgraph_linearized:
        Linearised code graph context emitted by :mod:`memory.subgraph_store`.
    snippets:
        Code snippets selected for editing, typically taken from
        ``PlannerEnv`` observations.
    extra:
        Optional metadata forwarded to the remote CGM endpoint.
    """

    payload: Dict[str, Any] = {
        "issue": issue or {},
        "plan": plan.to_dict(),
        "plan_text": plan_text or "",
        "subgraph": list(subgraph_linearized or []),
        "snippets": list(snippets or []),
    }
    if extra:
        payload.update(extra)
    return payload


class CodeFuseCGMClient:
    """Thin wrapper around the CodeFuse CGM HTTP API."""

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout_s: int = 60,
    ) -> None:
        if not endpoint:
            raise ValueError("endpoint must be provided")
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_patch(
        self,
        *,
        issue: Optional[Dict[str, Any]],
        plan: Plan,
        plan_text: Optional[str],
        subgraph_linearized: Optional[Iterable[Dict[str, Any]]],
        snippets: Optional[Iterable[Dict[str, Any]]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Patch:
        payload = build_cgm_payload(
            issue=issue,
            plan=plan,
            plan_text=plan_text,
            subgraph_linearized=subgraph_linearized,
            snippets=snippets,
            extra=dict(metadata or {}),
        )
        model_config = self._build_model_config()
        if model_config:
            payload["model_config"] = model_config
        response = self._post_json(payload)
        patch = self._extract_patch(response)
        if not patch.get("summary"):
            patch["summary"] = response.get("summary") or "codefuse-cgm"
        return patch

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_model_config(self) -> Dict[str, Any]:
        config: Dict[str, Any] = {}
        if self.model:
            config.setdefault("model", self.model)
        if self.temperature is not None:
            config.setdefault("temperature", self.temperature)
        if self.max_tokens is not None:
            config.setdefault("max_tokens", self.max_tokens)
        return config

    def _post_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(self.endpoint, data=data)
        req.add_header("Content-Type", "application/json")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read()
                if not raw:
                    raise CodeFuseCGMError("empty response body from CGM service")
                return json.loads(raw.decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise CodeFuseCGMError(f"CGM HTTPError {exc.code}: {body}") from exc
        except error.URLError as exc:
            raise CodeFuseCGMError(f"CGM URLError: {exc.reason}") from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise CodeFuseCGMError(f"CGM request failed: {exc}") from exc

    def _extract_patch(self, response: Dict[str, Any]) -> Patch:
        if not isinstance(response, dict):
            raise CodeFuseCGMError("invalid CGM response: expected dict")
        patch_obj = response.get("patch") or {}
        if not isinstance(patch_obj, dict):
            raise CodeFuseCGMError("CGM response missing 'patch' object")
        edits = patch_obj.get("edits") or []
        if not isinstance(edits, list):
            raise CodeFuseCGMError("CGM response 'edits' must be a list")
        normalized_edits = []
        for entry in edits:
            if not isinstance(entry, dict):
                continue
            try:
                normalized_edits.append(
                    {
                        "path": str(entry["path"]),
                        "start": int(entry.get("start", entry.get("line", 1))),
                        "end": int(entry.get("end", entry.get("start", entry.get("line", 1)))),
                        "new_text": self._ensure_newline(str(entry["new_text"])),
                    }
                )
            except Exception:
                continue
        summary = patch_obj.get("summary") or response.get("summary")
        patch: Patch = {"edits": normalized_edits}
        if summary:
            patch["summary"] = str(summary)
        return patch

    @staticmethod
    def _ensure_newline(text: str) -> str:
        return text if text.endswith("\n") else text + "\n"
