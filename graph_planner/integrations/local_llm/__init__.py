"""Utilities for interacting with locally hosted chat/completion models."""

from .client import LocalLLMClient, LocalLLMError

__all__ = ["LocalLLMClient", "LocalLLMError"]
