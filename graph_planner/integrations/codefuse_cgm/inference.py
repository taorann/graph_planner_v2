"""本地加载 CodeFuse CGM 模型进行推理的封装。

English summary
    Thin wrapper around Hugging Face causal LMs so downstream code can load a
    CGM checkpoint with graph-aware prompt formatting and generate patches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


def _resolve_dtype(name: Optional[str]) -> Optional[torch.dtype]:
    """Translate configuration dtype strings to ``torch.dtype`` values."""

    if not name:
        return None
    lowered = name.strip().lower()
    try:
        return getattr(torch, lowered)
    except AttributeError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported torch dtype: {name}") from exc


def _primary_device(model: AutoModelForCausalLM, fallback: str) -> torch.device:
    """Best-effort detection of the device that should host the inputs."""

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

from .data import CGMExample, GraphLinearizer, SnippetFormatter
from .formatting import ConversationEncoder


@dataclass
class CGMGenerationConfig:
    """Configuration controlling decoding behaviour."""

    model_name_or_path: str
    tokenizer_name_or_path: Optional[str] = None
    max_length: int = 8192
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.9
    do_sample: bool = False
    num_return_sequences: int = 1
    device: Optional[str] = None
    device_map: Optional[Any] = None
    torch_dtype: Optional[str] = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    trust_remote_code: bool = False
    attn_implementation: Optional[str] = None
    model_kwargs: Mapping[str, Any] = field(default_factory=dict)


class CodeFuseCGMGenerator:
    """Generate patches from a locally hosted CGM model."""

    def __init__(self, config: CGMGenerationConfig) -> None:
        """初始化 tokenizer、模型与上下文格式化工具。"""

        self.config = config
        explicit_device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = _resolve_dtype(config.torch_dtype)

        tok_path = config.tokenizer_name_or_path or config.model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path)
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

        self.encoder = ConversationEncoder(self.tokenizer, max_length=config.max_length)
        self.linearizer = GraphLinearizer()
        self.snippet_formatter = SnippetFormatter()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, example: CGMExample) -> List[str]:
        """Generate patch candidates for ``example``."""

        encoded = self.encoder.encode_prompt(
            prompt=example.prompt,
            plan_text=example.plan,
            graph_text=example.graph_text(linearizer=self.linearizer),
            snippets_text=example.snippets_text(formatter=self.snippet_formatter),
            issue_text=example.issue_text,
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        generated = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=self.config.do_sample,
            num_return_sequences=self.config.num_return_sequences,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        prompt_length = input_ids.shape[-1]
        sequences: List[str] = []
        for seq in generated:
            completion = seq[prompt_length:]
            text = self.tokenizer.decode(completion, skip_special_tokens=True)
            sequences.append(text.strip())
        return sequences


__all__ = [
    "CGMGenerationConfig",
    "CodeFuseCGMGenerator",
]

