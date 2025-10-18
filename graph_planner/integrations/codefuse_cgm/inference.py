"""本地加载 CodeFuse CGM 模型进行推理的封装。

English summary
    Thin wrapper around Hugging Face causal LMs so downstream code can load a
    CGM checkpoint with graph-aware prompt formatting and generate patches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

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


class CodeFuseCGMGenerator:
    """Generate patches from a locally hosted CGM model."""

    def __init__(self, config: CGMGenerationConfig) -> None:
        """初始化 tokenizer、模型与上下文格式化工具。"""

        self.config = config
        self.device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))

        tok_path = config.tokenizer_name_or_path or config.model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path)
        self.model.to(self.device)
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

