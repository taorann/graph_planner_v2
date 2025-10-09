# -*- coding: utf-8 -*-
from __future__ import annotations

"""
统一数据结构与校验（AciResp / Plan / Patch / Feedback / CollateMeta / CGMReqMeta）
Unified schemas & validation for ACI.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, TypedDict, Literal
from datetime import datetime

# ---------- ACI unified response ----------

class AciResp(TypedDict, total=False):
    success: bool                 # 操作是否成功
    message: str                  # 人类可读提示
    data: Dict[str, Any]          # 结构化数据
    logs: List[str]               # 原始日志行
    metrics: Dict[str, Any]       # 计量（耗时、修改行数等）
    ts: str                       # ISO 时间戳
    elapsed_ms: int               # 本次操作耗时（毫秒）


def validate_aci_resp(obj: Dict[str, Any]) -> None:
    """
    轻量校验（必需键与类型）。
    Lightweight validation for AciResp.
    """
    required = ("success", "message", "data", "logs", "metrics", "ts", "elapsed_ms")
    missing = [k for k in required if k not in obj]
    if missing:
        raise ValueError(f"AciResp missing keys: {missing}")
    if not isinstance(obj["success"], bool):
        raise TypeError("AciResp.success must be bool")
    if not isinstance(obj["message"], str):
        raise TypeError("AciResp.message must be str")
    if not isinstance(obj["data"], dict):
        raise TypeError("AciResp.data must be dict")
    if not isinstance(obj["logs"], list):
        raise TypeError("AciResp.logs must be list")
    if not isinstance(obj["metrics"], dict):
        raise TypeError("AciResp.metrics must be dict")
    # ts/elapsed_ms: basic format checks
    try:
        datetime.fromisoformat(obj["ts"].replace("Z", "+00:00"))
    except Exception as e:
        raise ValueError(f"AciResp.ts invalid ISO: {e}")
    if not isinstance(obj["elapsed_ms"], int):
        raise TypeError("AciResp.elapsed_ms must be int")


# ---------- Plan / Patch / Feedback ----------

@dataclass
class PlanTarget:
    path: str
    start: int
    end: int
    id: str
    confidence: float = 1.0
    why: str = ""


@dataclass
class Plan:
    targets: List[PlanTarget]
    budget: Dict[str, Any] = field(default_factory=dict)
    priority_tests: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "targets": [asdict(t) for t in self.targets],
            "budget": self.budget,
            "priority_tests": self.priority_tests,
        }


class PatchEdit(TypedDict):
    path: str
    start: int          # 1-based inclusive
    end: int            # 1-based inclusive
    new_text: str


class Patch(TypedDict, total=False):
    """
    对 CGM 输出最小兼容。ACI 仅负责落地与验证，不做生成。
    A minimal structure ACI can apply.
    """
    edits: List[PatchEdit]
    summary: str


class Feedback(TypedDict, total=False):
    """
    ACI 执行后的归纳反馈：lint/test/diff/回执
    """
    ok: bool
    test_report: Dict[str, Any]
    lint_report: Dict[str, Any]
    diff_summary: str
    changed_files: List[str]


# ---------- Step 4.0 新增：Collate & CGM 元信息 ----------

class CollateMeta(TypedDict, total=False):
    """
    由 Collater/线性化阶段统计出的上下文摘要，便于事件与预算控制。
    """
    chunks: int
    total_lines: int
    avg_lines: float
    max_lines: int
    tfile_chunk_ratio: float
    est_tokens: int
    reordered: bool            # 是否启用过轻量重排
    warnings: List[str]


class CGMReqMeta(TypedDict, total=False):
    """
    CGM 调用的请求侧元信息（用于事件/排障）。
    """
    endpoint: str
    model: str
    temperature: float
    max_tokens: int
    timeout_s: int


__all__ = [
    "AciResp",
    "validate_aci_resp",
    "Plan", "PlanTarget",
    "Patch", "PatchEdit",
    "Feedback",
    "CollateMeta",
    "CGMReqMeta",
]
