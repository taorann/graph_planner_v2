"""本地 CodeFuse CGM 训练循环封装。

English summary
    Offers a compact supervised fine-tuning loop (dataset, collator, trainer)
    so developers can reproduce the upstream CGM training flow without pulling
    the original repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import math

import torch
from torch.utils.data import DataLoader

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler,
)

from .data import CGMExample, CodeFuseCGMDataset, GraphLinearizer, SnippetFormatter
from .formatting import ConversationEncoder


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CGMTrainingConfig:
    """Collection of hyper-parameters for supervised fine-tuning."""

    model_name_or_path: str
    dataset_path: str
    output_dir: str
    graph_root: Optional[str] = None
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_epochs: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    eval_dataset_path: Optional[str] = None
    logging_steps: int = 50
    eval_steps: Optional[int] = None
    save_steps: Optional[int] = None
    max_length: int = 8192
    num_workers: int = 0
    device: Optional[str] = None


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------


class CGMBatchCollator:
    """Collate :class:`CGMExample` objects into tensors."""

    def __init__(
        self,
        encoder: ConversationEncoder,
        *,
        linearizer: Optional[GraphLinearizer] = None,
        snippet_formatter: Optional[SnippetFormatter] = None,
        pad_token_id: Optional[int] = None,
    ) -> None:
        """记录格式化工具并确定 padding token。"""

        self.encoder = encoder
        self.linearizer = linearizer or GraphLinearizer()
        self.snippet_formatter = snippet_formatter or SnippetFormatter()
        self.pad_token_id = pad_token_id if pad_token_id is not None else encoder.tokenizer.pad_token_id

    def __call__(self, batch: Iterable[CGMExample]) -> dict[str, torch.Tensor]:
        """将批量样本编码并进行动态 padding。"""

        input_ids, attention_masks, labels = [], [], []
        max_len = 0

        for example in batch:
            encoded = self.encoder.encode_example(
                prompt=example.prompt,
                response=example.response,
                plan_text=example.plan,
                graph_text=example.graph_text(linearizer=self.linearizer),
                snippets_text=example.snippets_text(formatter=self.snippet_formatter),
                issue_text=example.issue_text,
            )
            max_len = max(max_len, encoded["input_ids"].shape[-1])
            input_ids.append(encoded["input_ids"])
            attention_masks.append(encoded["attention_mask"])
            labels.append(encoded["labels"])

        def _pad(tensors: list[torch.Tensor], value: int) -> torch.Tensor:
            """内部辅助函数：右侧补齐张量并堆叠。"""

            padded = []
            for tensor in tensors:
                if tensor.shape[-1] < max_len:
                    pad_width = max_len - tensor.shape[-1]
                    tensor = torch.nn.functional.pad(tensor, (0, pad_width), value=value)
                padded.append(tensor)
            return torch.stack(padded, dim=0)

        batch_dict = {
            "input_ids": _pad(input_ids, self.pad_token_id),
            "attention_mask": _pad(attention_masks, 0),
            "labels": _pad(labels, -100),
        }
        return batch_dict


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class CodeFuseCGMTrainer:
    """Minimal training loop tailored for CGM supervised fine-tuning."""

    def __init__(self, config: CGMTrainingConfig) -> None:
        """根据配置初始化模型、数据集与优化器。"""

        self.config = config
        self.device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path)
        self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, use_fast=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.encoder = ConversationEncoder(
            tokenizer=self.tokenizer,
            max_length=config.max_length,
        )

        self.linearizer = GraphLinearizer()
        self.snippet_formatter = SnippetFormatter()

        self.train_dataset = CodeFuseCGMDataset(
            config.dataset_path,
            graph_root=Path(config.graph_root) if config.graph_root else None,
        )
        self.eval_dataset = (
            CodeFuseCGMDataset(
                config.eval_dataset_path,
                graph_root=Path(config.graph_root) if config.graph_root else None,
            )
            if config.eval_dataset_path
            else None
        )

        self.collator = CGMBatchCollator(
            self.encoder,
            linearizer=self.linearizer,
            snippet_formatter=self.snippet_formatter,
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        total_train_steps = math.ceil(
            len(self.train_dataset)
            / (config.per_device_batch_size * config.gradient_accumulation_steps)
        ) * config.num_epochs

        warmup_steps = int(total_train_steps * config.warmup_ratio)
        self.scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_train_steps,
        )

    # ------------------------------------------------------------------
    def train(self) -> None:
        """执行完整的监督训练流程并定期评估/保存检查点。"""

        cfg = self.config
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=cfg.per_device_batch_size,
            shuffle=True,
            collate_fn=self.collator,
            num_workers=cfg.num_workers,
        )
        eval_loader = (
            DataLoader(
                self.eval_dataset,
                batch_size=cfg.per_device_batch_size,
                shuffle=False,
                collate_fn=self.collator,
                num_workers=cfg.num_workers,
            )
            if self.eval_dataset
            else None
        )

        global_step = 0
        completed_epochs = 0
        best_eval_loss: Optional[float] = None

        self.model.train()
        for epoch in range(cfg.num_epochs):
            completed_epochs = epoch + 1
            for batch_idx, batch in enumerate(train_loader):
                loss = self._training_step(batch)
                if (global_step + 1) % cfg.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                global_step += 1

                if cfg.logging_steps and global_step % cfg.logging_steps == 0:
                    print(f"[train] step={global_step} loss={loss:.4f}")  # noqa: T201

                if cfg.eval_steps and eval_loader and global_step % cfg.eval_steps == 0:
                    eval_loss = self.evaluate(eval_loader)
                    print(  # noqa: T201
                        f"[eval] step={global_step} loss={eval_loss:.4f}"
                    )
                    if best_eval_loss is None or eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        self.save_checkpoint(Path(cfg.output_dir) / "best")

                if cfg.save_steps and global_step % cfg.save_steps == 0:
                    self.save_checkpoint(Path(cfg.output_dir) / f"step-{global_step}")

        # Final checkpoint after training completes
        self.save_checkpoint(Path(cfg.output_dir) / "final")
        if eval_loader:
            eval_loss = self.evaluate(eval_loader)
            print(f"[eval] final loss={eval_loss:.4f}")  # noqa: T201

        print(  # noqa: T201
            f"Training completed: epochs={completed_epochs} steps={global_step}"
        )

    # ------------------------------------------------------------------
    def _training_step(self, batch: dict[str, torch.Tensor]) -> float:
        """执行一次前向 + 反向传播，返回平均化后的 loss。"""

        self.model.train()
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss / self.config.gradient_accumulation_steps
        loss.backward()
        return float(loss.item())

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        """在评估集上计算平均损失。"""

        self.model.eval()
        losses = []
        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            losses.append(outputs.loss.item())
        self.model.train()
        return float(sum(losses) / max(len(losses), 1))

    # ------------------------------------------------------------------
    def save_checkpoint(self, output_dir: Path) -> None:
        """保存当前模型与 tokenizer。"""

        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)


__all__ = [
    "CGMTrainingConfig",
    "CGMBatchCollator",
    "CodeFuseCGMTrainer",
]

