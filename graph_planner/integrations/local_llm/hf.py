"""基于 Hugging Face 的本地对话模型客户端。

English summary
    Wraps a causal LM checkpoint with a chat-style interface so planner agents
    can run locally without remote services.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _resolve_dtype(name: Optional[str]) -> Optional[torch.dtype]:
    """Translate a user supplied dtype string into a ``torch.dtype``."""

    if not name:
        return None
    lowered = name.strip().lower()
    try:
        return getattr(torch, lowered)
    except AttributeError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Unsupported torch dtype: {name}") from exc


def _primary_device(model: AutoModelForCausalLM, fallback: str) -> torch.device:
    """Infer the device that should host tokenizer inputs for generation."""

    if hasattr(model, "device") and model.device is not None:
        return torch.device(model.device)
    if hasattr(model, "hf_device_map"):
        device_map = getattr(model, "hf_device_map") or {}
        if isinstance(device_map, Mapping) and device_map:
            first = next(iter(device_map.values()))
            if isinstance(first, (list, tuple)) and first:
                first = first[0]
            return torch.device(first)
    try:
        param = next(model.parameters())
    except StopIteration:  # pragma: no cover - model without parameters
        return torch.device(fallback)
    return torch.device(param.device)


@dataclass
class HuggingFaceChatConfig:
    """Configuration for :class:`HuggingFaceChatClient`."""

    model_name_or_path: str
    tokenizer_name_or_path: Optional[str] = None
    device: Optional[str] = None
    max_length: int = 4096
    max_new_tokens: int = 1024
    temperature: float = 0.2
    top_p: float = 0.95
    do_sample: bool = True
    device_map: Optional[Any] = None
    torch_dtype: Optional[str] = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    trust_remote_code: bool = False
    attn_implementation: Optional[str] = None
    model_kwargs: Mapping[str, Any] = field(default_factory=dict)


class HuggingFaceChatClient:
    """Simple chat wrapper around a causal language model."""

    def __init__(self, config: HuggingFaceChatConfig) -> None:
        """加载 tokenizer/模型并放置到配置指定的设备上。"""

        self.config = config
        explicit_device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = _resolve_dtype(config.torch_dtype)

        tok_path = config.tokenizer_name_or_path or config.model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": bool(config.trust_remote_code),
        }
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        if config.device_map is not None:
            model_kwargs["device_map"] = config.device_map
        if config.load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        if config.load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        if config.attn_implementation:
            model_kwargs["attn_implementation"] = config.attn_implementation
        if config.model_kwargs:
            model_kwargs.update(dict(config.model_kwargs))

        self.model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **model_kwargs)
        if config.device_map is None:
            self.device = torch.device(explicit_device)
            self.model.to(self.device)
        else:
            self.device = _primary_device(self.model, explicit_device)
        self.model.eval()

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: Any) -> "HuggingFaceChatClient":
        """从通用配置对象读取路径与采样参数。"""

        return cls(
            HuggingFaceChatConfig(
                model_name_or_path=getattr(cfg, "model_path"),
                tokenizer_name_or_path=getattr(cfg, "tokenizer_path", None),
                device=getattr(cfg, "device", None),
                max_length=int(getattr(cfg, "max_input_tokens", 4096)),
                max_new_tokens=int(getattr(cfg, "max_tokens", 1024)),
                temperature=float(getattr(cfg, "temperature", 0.0)),
                top_p=float(getattr(cfg, "top_p", 0.95)),
                do_sample=bool(getattr(cfg, "temperature", 0.0) and getattr(cfg, "temperature", 0.0) > 0),
                device_map=getattr(cfg, "device_map", None),
                torch_dtype=getattr(cfg, "torch_dtype", None),
                load_in_8bit=bool(getattr(cfg, "load_in_8bit", False)),
                load_in_4bit=bool(getattr(cfg, "load_in_4bit", False)),
                trust_remote_code=bool(getattr(cfg, "trust_remote_code", False)),
                attn_implementation=getattr(cfg, "attn_implementation", None),
            )
        )

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
        """执行一次对话生成并返回助手回复文本。"""

        del extra  # metadata is not used for local generation

        temp = temperature if temperature is not None else self.config.temperature
        top_p_val = top_p if top_p is not None else self.config.top_p
        max_new_tokens = max_tokens if max_tokens is not None else self.config.max_new_tokens

        do_sample = (temp or 0.0) > 0 or self.config.do_sample

        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt_text = self.tokenizer.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = "\n\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"

        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": float(temp) if temp is not None else None,
            "top_p": float(top_p_val) if top_p_val is not None else None,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}

        with torch.no_grad():
            output = self.model.generate(**encoded, **generation_kwargs)

        prompt_len = encoded["input_ids"].shape[-1]
        completion = output[0][prompt_len:]
        text = self.tokenizer.decode(completion, skip_special_tokens=True)
        return text.strip()


__all__ = ["HuggingFaceChatClient", "HuggingFaceChatConfig"]
