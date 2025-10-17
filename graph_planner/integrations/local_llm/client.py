"""Minimal HTTP client for locally deployed OpenAI-compatible chat models."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, Optional
from urllib import error, request


class LocalLLMError(RuntimeError):
    """Raised when a local LLM endpoint returns an unexpected response."""


class LocalLLMClient:
    """Thin wrapper around a locally hosted chat completion endpoint.

    The client assumes the target service exposes an OpenAI-compatible
    ``/v1/chat/completions`` style API.  This is the default for popular
    self-hosting stacks such as vLLM, llama.cpp's server mode, TGI and
    the rLLM reference launcher.  The class keeps the implementation
    lightweight so it can be vendored into training scripts without
    pulling heavyweight HTTP dependencies.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        api_key: Optional[str] = None,
        timeout_s: int = 60,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        default_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not endpoint:
            raise ValueError("endpoint must be provided for LocalLLMClient")
        if not model:
            raise ValueError("model must be provided for LocalLLMClient")
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.timeout_s = int(timeout_s)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.default_metadata = dict(default_metadata or {})

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: Any) -> "LocalLLMClient":
        """Create a client from ``infra.config.PlannerModelCfg``."""

        enabled = bool(getattr(cfg, "enabled", False))
        if not enabled:
            raise RuntimeError("planner model is disabled in the configuration")
        endpoint = getattr(cfg, "endpoint", None)
        if not endpoint:
            raise ValueError("planner model endpoint must be configured when enabled")
        api_key_env = getattr(cfg, "api_key_env", None)
        api_key = os.environ.get(api_key_env) if api_key_env else None
        return cls(
            endpoint=endpoint,
            model=getattr(cfg, "model", None) or "",
            api_key=api_key,
            timeout_s=int(getattr(cfg, "timeout_s", 60)),
            temperature=getattr(cfg, "temperature", None),
            max_tokens=getattr(cfg, "max_tokens", None),
            top_p=getattr(cfg, "top_p", None),
            default_metadata=dict(getattr(cfg, "metadata", {}) or {}),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def chat(
        self,
        messages: Iterable[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Send a chat completion request and return the model response text."""

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
        }
        params: Dict[str, Any] = {}
        temp = temperature if temperature is not None else self.temperature
        if temp is not None:
            params["temperature"] = float(temp)
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        if max_tok is not None:
            params["max_tokens"] = int(max_tok)
        top_p_val = top_p if top_p is not None else self.top_p
        if top_p_val is not None:
            params["top_p"] = float(top_p_val)
        if params:
            payload.update(params)
        if extra:
            payload.update(extra)
        if self.default_metadata:
            payload.setdefault("metadata", {}).update(self.default_metadata)

        raw = self._post_json(payload)
        return self._extract_message(raw)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _post_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(self.endpoint, data=data)
        req.add_header("Content-Type", "application/json")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with request.urlopen(req, timeout=self.timeout_s) as resp:
                body = resp.read()
        except error.HTTPError as exc:  # pragma: no cover - network failure
            raise LocalLLMError(f"local LLM HTTPError {exc.code}: {exc.reason}") from exc
        except error.URLError as exc:  # pragma: no cover - network failure
            raise LocalLLMError(f"local LLM URLError: {exc.reason}") from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise LocalLLMError(f"local LLM request failed: {exc}") from exc

        if not body:
            raise LocalLLMError("local LLM endpoint returned an empty body")
        try:
            return json.loads(body.decode("utf-8"))
        except Exception as exc:
            raise LocalLLMError(f"failed to decode local LLM response: {exc}") from exc

    def _extract_message(self, response: Dict[str, Any]) -> str:
        if not isinstance(response, dict):
            raise LocalLLMError("local LLM response must be a JSON object")
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LocalLLMError("local LLM response missing 'choices'")
        choice = choices[0] or {}
        message = choice.get("message") or {}
        content = message.get("content")
        if content is None:
            raise LocalLLMError("local LLM response missing message content")
        return str(content)
