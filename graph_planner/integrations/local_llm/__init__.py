"""Utilities for interacting with locally hosted chat/completion models."""

from .client import LocalLLMClient, LocalLLMError


def build_planner_client(cfg):
    """Create a planner client from configuration.

    若配置提供 ``model_path``，则使用 Hugging Face 权重直接加载本地模型；
    否则退回到 OpenAI 兼容的 HTTP 客户端。
    """

    model_path = getattr(cfg, "model_path", None)
    if model_path:
        from .hf import HuggingFaceChatClient

        return HuggingFaceChatClient.from_config(cfg)
    return LocalLLMClient.from_config(cfg)


__all__ = ["LocalLLMClient", "LocalLLMError", "build_planner_client"]
