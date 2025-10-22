"""本地 CodeFuse CGM 数据集与样本结构定义。

English summary
    The module exposes structured containers and loaders so CGM training and
    inference code can consume prompts, graph metadata, snippets and plan text
    without depending on the original ``CodeFuse-CGM`` scripts.  It mirrors the
    upstream semantics while providing clearer typing and validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Mapping, Optional, Sequence

import json

from .formatting import GraphDict, GraphLinearizer, SnippetFormatter, load_graph_document


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


def _to_path(base: Optional[Path], value: Optional[str]) -> Optional[Path]:
    """将相对路径解析为绝对 ``Path``。

    Parameters
    ----------
    base:
        作为解析参考的目录。若为空则直接使用 ``value``。
    value:
        记录中的路径字段，允许为空或相对路径。
    """

    if not value:
        return None
    path = Path(value)
    if base and not path.is_absolute():
        path = base / path
    return path


def _ensure_mapping(value: object) -> Mapping[str, object]:
    """确保 ``value`` 为映射，便于后续字段访问。"""

    if isinstance(value, Mapping):
        return value
    return {}


def _ensure_sequence(value: object) -> Sequence[Mapping[str, object]]:
    """过滤非映射元素后返回序列副本。"""

    if isinstance(value, Sequence) and not isinstance(value, str):
        return [item for item in value if isinstance(item, Mapping)]  # type: ignore[list-item]
    return []


@dataclass
class CGMExample:
    """Single training/inference example for the CGM reader."""

    prompt: str
    response: str
    graph: Optional[GraphDict]
    plan: Optional[str]
    issue: Mapping[str, object]
    snippets: Sequence[Mapping[str, object]]
    metadata: Mapping[str, object]

    def graph_text(self, *, linearizer: GraphLinearizer) -> str:
        """使用给定 ``linearizer`` 生成子图文本。"""

        return linearizer.linearize(self.graph)

    def snippets_text(self, *, formatter: SnippetFormatter) -> str:
        """利用 ``formatter`` 将候选片段格式化为文本。"""

        return formatter.format(self.snippets)

    @property
    def issue_text(self) -> Optional[str]:
        """返回问题描述（优先正文，其次标题）。"""

        description = self.issue.get("body") or self.issue.get("description")
        if isinstance(description, str) and description.strip():
            return description.strip()
        title = self.issue.get("title")
        if isinstance(title, str) and title.strip():
            return title.strip()
        return None


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------


def _iter_jsonl(path: Path) -> Iterator[Mapping[str, object]]:
    """按行读取 JSONL 文件并产出映射对象。"""

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _iter_json(path: Path) -> Iterator[Mapping[str, object]]:
    """读取 JSON 文件，兼容 ``{"data": [...]} `` 与纯列表格式。"""

    payload = json.loads(path.read_text("utf-8"))
    if isinstance(payload, Mapping):
        data = payload.get("data") or payload.get("examples")
        if isinstance(data, Sequence):
            for item in data:
                if isinstance(item, Mapping):
                    yield item
        return
    if isinstance(payload, Sequence):
        for item in payload:
            if isinstance(item, Mapping):
                yield item


def _detect_prompt(record: Mapping[str, object]) -> str:
    """从多种候选字段中提取 prompt。"""

    for key in ("prompt", "question", "input"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _detect_response(record: Mapping[str, object]) -> str:
    """提取参考回答或目标输出文本。"""

    for key in ("answer", "response", "output"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _detect_plan(record: Mapping[str, object]) -> Optional[str]:
    """提取可选的计划描述字段。"""

    for key in ("plan", "plan_text", "repair_plan"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _detect_issue(record: Mapping[str, object]) -> Mapping[str, object]:
    """返回问题元数据映射，默认空字典。"""

    issue = record.get("issue") or {}
    if isinstance(issue, Mapping):
        return issue
    return {}


def _detect_snippets(record: Mapping[str, object]) -> Sequence[Mapping[str, object]]:
    """解析候选代码片段列表。"""

    for key in ("snippets", "chunks", "candidate_snippets"):
        value = record.get(key)
        seq = _ensure_sequence(value)
        if seq:
            return seq
    return []


def _detect_metadata(record: Mapping[str, object]) -> Mapping[str, object]:
    """读取附加元信息字段。"""

    meta = record.get("metadata") or {}
    if isinstance(meta, Mapping):
        return meta
    return {}


def _detect_graph(record: Mapping[str, object], base_dir: Optional[Path]) -> Optional[GraphDict]:
    """从记录中获取内嵌图或加载外部图文件。"""

    graph = record.get("graph")
    if isinstance(graph, Mapping):
        return graph
    if isinstance(graph, str):
        return load_graph_document(_to_path(base_dir, graph))

    repo_file = record.get("repo")
    if isinstance(repo_file, str):
        candidate = _to_path(base_dir, repo_file)
        if candidate and candidate.exists():
            return load_graph_document(candidate)
    return None


class CodeFuseCGMDataset:
    """Load CGM training examples from JSON/JSONL files."""

    def __init__(self, path: Path | str, *, graph_root: Optional[Path] = None) -> None:
        """读取给定路径的样本文件，并可选指定图文件根目录。"""

        self.path = Path(path)
        self.graph_root = graph_root
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self._records: List[Mapping[str, object]] = list(self._load_records())

    # Public API ---------------------------------------------------------
    def __len__(self) -> int:  # pragma: no cover - trivial
        """数据集中样本数量。"""

        return len(self._records)

    def __getitem__(self, idx: int) -> CGMExample:
        """根据索引返回标准化后的 :class:`CGMExample`。"""

        record = self._records[idx]
        prompt = _detect_prompt(record)
        response = _detect_response(record)
        plan = _detect_plan(record)
        issue = _detect_issue(record)
        snippets = _detect_snippets(record)
        metadata = _detect_metadata(record)
        graph = _detect_graph(record, self.graph_root)

        if not prompt:
            raise ValueError(f"record[{idx}] is missing a prompt")
        if not response:
            raise ValueError(f"record[{idx}] is missing an answer/response")

        return CGMExample(
            prompt=prompt,
            response=response,
            graph=graph,
            plan=plan,
            issue=issue,
            snippets=snippets,
            metadata=metadata,
        )

    # Internal helpers ---------------------------------------------------
    def _load_records(self) -> Iterable[Mapping[str, object]]:
        """根据文件后缀选择 JSON/JSONL 解析方式。"""

        suffix = self.path.suffix.lower()
        if suffix == ".jsonl":
            yield from _iter_jsonl(self.path)
            return
        if suffix == ".json":
            yield from _iter_json(self.path)
            return
        raise ValueError(f"Unsupported dataset format: {self.path.suffix}")


__all__ = [
    "CGMExample",
    "CodeFuseCGMDataset",
    "GraphLinearizer",
    "SnippetFormatter",
]

