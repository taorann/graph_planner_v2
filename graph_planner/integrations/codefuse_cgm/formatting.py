"""CGM 上下文序列化工具。

English summary
    Provides drop-in replacements for the fragmented preprocessing utilities in
    ``CodeFuse-CGM`` by exposing well documented helpers that turn graph nodes
    and snippets into linearised text and then compose chat-style prompts for
    training or inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

import json

from transformers import PreTrainedTokenizerBase


# ---------------------------------------------------------------------------
# Graph formatting
# ---------------------------------------------------------------------------


GraphDict = Mapping[str, object]


def load_graph_document(path: Path | str | None) -> Optional[GraphDict]:
    """读取子图 JSON 文档，兼容 ``None``/缺失文件。"""

    if path is None:
        return None
    graph_path = Path(path)
    if not graph_path.exists():
        return None
    with graph_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _pick(value: Mapping[str, object], *keys: str) -> Optional[str]:
    """按优先级返回首个非空字符串字段。"""

    for key in keys:
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _pick_list(value: Mapping[str, object], *keys: str) -> Sequence[str]:
    """提取字符串序列字段并进行类型转换。"""

    for key in keys:
        candidate = value.get(key)
        if isinstance(candidate, Sequence) and not isinstance(candidate, str):
            return [str(item) for item in candidate]
    return []


def _ellipsis(text: str, max_chars: int) -> str:
    """在超长时追加省略号以控制字段长度。"""

    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


@dataclass
class GraphLinearizer:
    """Render graph nodes into a structured, readable text block.

    输入数据假定来自上游 `serialize_subgraph.py` 生成的 JSON（节点、边
    列表），字段内容的抽取与截断策略则参考了
    ``cgm/data/preprocess.py`` 中的 ``getJavaSentence``、``getPythonSentence``
    以及 ``graph2embedding``。这样可以在保留原始图结构的同时，输出
    更适合语言模型阅读的文本。
    """

    max_nodes: int = 32
    max_chars_per_field: int = 512

    def linearize(self, graph: Optional[GraphDict]) -> str:
        """将 ``serialize_subgraph`` 产出的节点列表转换为多段文本。"""

        if not graph:
            return ""

        nodes = graph.get("nodes")
        if not isinstance(nodes, Sequence):
            return ""

        sections: List[str] = []
        for node in nodes[: self.max_nodes]:
            if not isinstance(node, Mapping):
                continue
            name = _pick(node, "name", "label", "title", "id", "nodeId") or "(anonymous)"
            node_type = _pick(node, "nodeType", "type")
            header = f"- {name}"
            if node_type:
                header += f" [{node_type}]"

            body_parts: List[str] = []
            summary = _pick(
                node,
                "summary",
                "docstring",
                "comment",
                "description",
                "signature",
            )
            if summary:
                body_parts.append(_ellipsis(summary, self.max_chars_per_field))

            text = _pick(node, "text", "code", "content", "body")
            if text:
                body_parts.append(_ellipsis(text, self.max_chars_per_field))

            anchors = _pick_list(node, "anchors", "anchor", "keywords")
            if anchors:
                body_parts.append("Anchors: " + ", ".join(anchors[:6]))

            if body_parts:
                sections.append(header + "\n" + "\n".join(f"    {line}" for line in body_parts))
            else:
                sections.append(header)

        return "\n".join(sections)


# ---------------------------------------------------------------------------
# Snippet formatting
# ---------------------------------------------------------------------------


@dataclass
class SnippetFormatter:
    """Serialise candidate code snippets in a deterministic order."""

    max_snippets: int = 5
    max_lines_per_snippet: int = 40

    def format(self, snippets: Optional[Sequence[Mapping[str, object]]]) -> str:
        """将候选片段序列转换为 ``path:start-end`` + 代码正文格式。"""

        if not snippets:
            return ""
        blocks: List[str] = []
        for entry in snippets[: self.max_snippets]:
            if not isinstance(entry, Mapping):
                continue
            path = _pick(entry, "path", "abs_path") or "unknown"
            start = entry.get("start") or entry.get("line")
            end = entry.get("end") or start
            header = f"{path}:{start}-{end}"
            lines = entry.get("snippet") or entry.get("lines")
            if isinstance(lines, Sequence) and not isinstance(lines, str):
                normalized = []
                for raw in lines[: self.max_lines_per_snippet]:
                    normalized.append(str(raw))
                blocks.append(header + "\n" + "\n".join(normalized))
            else:
                body = _pick(entry, "text", "content")
                if body:
                    blocks.append(header + "\n" + _ellipsis(body, 1024))
                else:
                    blocks.append(header)
        return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Conversation encoding
# ---------------------------------------------------------------------------


DEFAULT_SYSTEM_PROMPT = (
    "You are CodeFuse-CGM, a graph-aware assistant that generates precise "
    "code patches.  Use the provided issue description, planner plan and "
    "graph context to infer the necessary edits.  Reply with the patch diff "
    "or a detailed fix strategy when code changes are not possible."
)


@dataclass
class ConversationEncoder:
    """Compose chat prompts for CGM training and inference."""

    tokenizer: PreTrainedTokenizerBase
    max_length: int = 8192
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    def build_user_message(
        self,
        *,
        prompt: str,
        plan_text: Optional[str],
        graph_text: str,
        snippets_text: str,
        issue_text: Optional[str],
    ) -> str:
        """拼装包含 Issue/Plan/Graph/Snippets 的用户消息文本。"""

        sections: List[str] = []
        if issue_text:
            sections.append(f"[Issue]\n{issue_text.strip()}")
        sections.append(f"[Instruction]\n{prompt.strip()}")
        if plan_text:
            sections.append(f"[Plan]\n{plan_text.strip()}")
        if graph_text:
            sections.append(f"[Subgraph]\n{graph_text}")
        if snippets_text:
            sections.append(f"[Snippets]\n{snippets_text}")
        return "\n\n".join(sections)

    def _apply_chat_template(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        add_generation_prompt: bool,
    ) -> MutableMapping[str, object]:
        """调用 tokenizer 聊天模板或退化为 ``role: content`` 拼接。"""

        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )

        # Fallback: join messages manually when the tokenizer does not define a
        # chat template.  We fall back to a simple "role: message" format.
        text_blocks = []
        for msg in messages:
            text_blocks.append(f"{msg['role'].upper()}: {msg['content']}")
        if add_generation_prompt:
            text_blocks.append("ASSISTANT:")
        encoded = self.tokenizer(
            "\n\n".join(text_blocks),
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        return encoded

    def encode_prompt(
        self,
        *,
        prompt: str,
        plan_text: Optional[str],
        graph_text: str,
        snippets_text: str,
        issue_text: Optional[str],
    ) -> MutableMapping[str, object]:
        """仅编码提示部分，用于推理阶段生成输入张量。"""

        user_message = self.build_user_message(
            prompt=prompt,
            plan_text=plan_text,
            graph_text=graph_text,
            snippets_text=snippets_text,
            issue_text=issue_text,
        )
        messages = [{"role": "user", "content": user_message}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        return self._apply_chat_template(messages, add_generation_prompt=True)

    def encode_example(
        self,
        *,
        prompt: str,
        response: str,
        plan_text: Optional[str],
        graph_text: str,
        snippets_text: str,
        issue_text: Optional[str],
    ) -> MutableMapping[str, object]:
        """编码单条监督样本并对提示 token 打上 ``-100`` 标签。"""

        user_message = self.build_user_message(
            prompt=prompt,
            plan_text=plan_text,
            graph_text=graph_text,
            snippets_text=snippets_text,
            issue_text=issue_text,
        )
        messages: List[Mapping[str, str]] = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": response.strip()},
        ]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        full = self._apply_chat_template(messages, add_generation_prompt=False)
        prompt_only = self._apply_chat_template(messages[:-1], add_generation_prompt=True)

        input_ids = full["input_ids"].squeeze(0)
        attention_mask = full["attention_mask"].squeeze(0)
        prompt_len = prompt_only["input_ids"].shape[-1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_length": prompt_len,
        }

