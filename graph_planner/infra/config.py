# -*- coding: utf-8 -*-
from __future__ import annotations
"""
infra/config.py

Step 4.0：配置定稿
- 固定上下文模式（wsd/full）
- 预算/配额/多样性
- 事件路径/开关
- Collater 配置（预算、片段上限、是否交错测试片段、轻量重排开关）
- CGM 配置（是否启用、endpoint/key/model/温度/超时/最大tokens）

优先级：环境变量 > .aci/config.json > 默认值
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional
import os
import json


PLANNER_MODEL_DIR = os.path.join("models", "qwen3-14b-instruct")
CGM_MODEL_DIR = os.path.join("models", "codefuse-cgm")


# ---------------- dataclasses ----------------

@dataclass
class LintCfg:
    enabled: bool = True
    # 预留：你可以在 tools 层读这个字段决定走 ruff/flake8/black 等
    framework: str = "auto"


@dataclass
class TelemetryCfg:
    events_path: str = os.path.join("logs", "events.jsonl")
    test_runs_path: str = os.path.join("logs", "test_runs.jsonl")


@dataclass
class CollateCfg:
    # 4.0 新增：Collater 的预算与行为
    mode: str = "wsd"                 # "wsd" | "full"（与全局 mode 对齐，允许覆盖）
    budget_tokens: int = 40000
    max_chunks: int = 64
    interleave_tests: bool = True     # 是否交错插入测试片段
    window_pad: int = 2               # 线性化窗口左右扩展（占位，4.1使用）
    enable_light_reorder: bool = False  # 轻量重排（默认关闭，只在超限时触发）
    per_file_max_chunks: int = 8      # 单文件片段上限（防止一个文件淹没上下文）


@dataclass
class CGMCfg:
    enabled: bool = False
    endpoint: Optional[str] = None       # e.g. https://cgm.internal/v1/generate
    api_key_env: str = "CGM_API_KEY"     # 从这个环境变量里取 key
    model: str = "cgm-default"
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 2048
    timeout_s: int = 60
    model_path: Optional[str] = CGM_MODEL_DIR      # 本地模型权重路径（Hugging Face 兼容）
    tokenizer_path: Optional[str] = None  # 如未指定则复用 model_path
    max_input_tokens: int = 8192
    device: Optional[str] = None


@dataclass
class PlannerModelCfg:
    enabled: bool = False
    endpoint: Optional[str] = None          # OpenAI-compatible chat endpoint
    api_key_env: str = "PLANNER_MODEL_API_KEY"
    model: str = "qwen2.5-coder-7b-instruct"
    temperature: float = 0.2
    max_tokens: int = 1024
    top_p: float = 0.95
    timeout_s: int = 60
    system_prompt: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    model_path: Optional[str] = PLANNER_MODEL_DIR          # 本地推理：HF checkpoint
    tokenizer_path: Optional[str] = None
    max_input_tokens: int = 4096
    device: Optional[str] = None


@dataclass
class Config:
    # ---- 核心运行模式 ----
    mode: str = "wsd"               # "wsd" | "full"
    enable_step3: bool = True       # 是否启用 Step 3（子图维护）

    # ---- 预算&配额（Step 3/4 用）----
    subgraph_total_cap: int = 800
    plan_k: int = 1                 # 预留（后续 Planner-LLM 用）
    max_nodes_per_anchor: int = 50
    candidate_total_limit: int = 200
    dir_diversity_k: int = 3

    # ---- 记忆/测试策略 ----
    prefer_test_files: bool = True
    max_tfile_fraction: float = 0.60
    memory_caps: Dict[str, int] = field(
        default_factory=lambda: {
            "nodes": 200,
            "edges": 1000,
            "frontier": 50,
            "planner_tokens": 2000,
            "cgm_tokens": 16000,
        }
    )

    # ---- 子系统配置 ----
    lint: LintCfg = field(default_factory=LintCfg)
    telemetry: TelemetryCfg = field(default_factory=TelemetryCfg)
    collate: CollateCfg = field(default_factory=CollateCfg)
    cgm: CGMCfg = field(default_factory=CGMCfg)
    planner_model: PlannerModelCfg = field(default_factory=PlannerModelCfg)

    # ---- 其他开关 ----
    memlog: Dict[str, Any] = field(default_factory=lambda: {"enabled": True})

    # ---- 序列化 ----
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


# ---------------- helpers ----------------

def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _load_json_file(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _apply_env_overrides(raw: Dict[str, Any]) -> Dict[str, Any]:
    # COLLATE
    if "COLLATE_MODE" in os.environ:
        raw.setdefault("collate", {})["mode"] = os.environ["COLLATE_MODE"]
    if "COLLATE_BUDGET_TOKENS" in os.environ:
        raw.setdefault("collate", {})["budget_tokens"] = int(os.environ["COLLATE_BUDGET_TOKENS"])
    if "COLLATE_MAX_CHUNKS" in os.environ:
        raw.setdefault("collate", {})["max_chunks"] = int(os.environ["COLLATE_MAX_CHUNKS"])
    if "COLLATE_INTERLEAVE_TESTS" in os.environ:
        raw.setdefault("collate", {})["interleave_tests"] = os.environ["COLLATE_INTERLEAVE_TESTS"].lower() in ("1", "true", "yes")
    if "COLLATE_ENABLE_LIGHT_REORDER" in os.environ:
        raw.setdefault("collate", {})["enable_light_reorder"] = os.environ["COLLATE_ENABLE_LIGHT_REORDER"].lower() in ("1", "true", "yes")

    # CGM
    if "CGM_ENABLED" in os.environ:
        raw.setdefault("cgm", {})["enabled"] = os.environ["CGM_ENABLED"].lower() in ("1", "true", "yes")
    if "CGM_ENDPOINT" in os.environ:
        raw.setdefault("cgm", {})["endpoint"] = os.environ["CGM_ENDPOINT"]
    if "CGM_MODEL" in os.environ:
        raw.setdefault("cgm", {})["model"] = os.environ["CGM_MODEL"]
    if "CGM_TEMPERATURE" in os.environ:
        raw.setdefault("cgm", {})["temperature"] = float(os.environ["CGM_TEMPERATURE"])
    if "CGM_TOP_P" in os.environ:
        raw.setdefault("cgm", {})["top_p"] = float(os.environ["CGM_TOP_P"])
    if "CGM_MAX_TOKENS" in os.environ:
        raw.setdefault("cgm", {})["max_tokens"] = int(os.environ["CGM_MAX_TOKENS"])
    if "CGM_TIMEOUT_S" in os.environ:
        raw.setdefault("cgm", {})["timeout_s"] = int(os.environ["CGM_TIMEOUT_S"])
    if "CGM_API_KEY_ENV" in os.environ:
        raw.setdefault("cgm", {})["api_key_env"] = os.environ["CGM_API_KEY_ENV"]
    if "CGM_MODEL_PATH" in os.environ:
        raw.setdefault("cgm", {})["model_path"] = os.environ["CGM_MODEL_PATH"]
    if "CGM_TOKENIZER_PATH" in os.environ:
        raw.setdefault("cgm", {})["tokenizer_path"] = os.environ["CGM_TOKENIZER_PATH"]
    if "CGM_MAX_INPUT_TOKENS" in os.environ:
        raw.setdefault("cgm", {})["max_input_tokens"] = int(os.environ["CGM_MAX_INPUT_TOKENS"])
    if "CGM_DEVICE" in os.environ:
        raw.setdefault("cgm", {})["device"] = os.environ["CGM_DEVICE"]

    # PLANNER MODEL
    if "PLANNER_MODEL_ENABLED" in os.environ:
        raw.setdefault("planner_model", {})["enabled"] = os.environ["PLANNER_MODEL_ENABLED"].lower() in ("1", "true", "yes")
    if "PLANNER_MODEL_ENDPOINT" in os.environ:
        raw.setdefault("planner_model", {})["endpoint"] = os.environ["PLANNER_MODEL_ENDPOINT"]
    if "PLANNER_MODEL_MODEL" in os.environ:
        raw.setdefault("planner_model", {})["model"] = os.environ["PLANNER_MODEL_MODEL"]
    if "PLANNER_MODEL_API_KEY_ENV" in os.environ:
        raw.setdefault("planner_model", {})["api_key_env"] = os.environ["PLANNER_MODEL_API_KEY_ENV"]
    if "PLANNER_MODEL_TEMPERATURE" in os.environ:
        raw.setdefault("planner_model", {})["temperature"] = float(os.environ["PLANNER_MODEL_TEMPERATURE"])
    if "PLANNER_MODEL_TOP_P" in os.environ:
        raw.setdefault("planner_model", {})["top_p"] = float(os.environ["PLANNER_MODEL_TOP_P"])
    if "PLANNER_MODEL_MAX_TOKENS" in os.environ:
        raw.setdefault("planner_model", {})["max_tokens"] = int(os.environ["PLANNER_MODEL_MAX_TOKENS"])
    if "PLANNER_MODEL_TIMEOUT_S" in os.environ:
        raw.setdefault("planner_model", {})["timeout_s"] = int(os.environ["PLANNER_MODEL_TIMEOUT_S"])
    if "PLANNER_MODEL_SYSTEM_PROMPT" in os.environ:
        raw.setdefault("planner_model", {})["system_prompt"] = os.environ["PLANNER_MODEL_SYSTEM_PROMPT"]
    if "PLANNER_MODEL_PATH" in os.environ:
        raw.setdefault("planner_model", {})["model_path"] = os.environ["PLANNER_MODEL_PATH"]
    if "PLANNER_MODEL_TOKENIZER_PATH" in os.environ:
        raw.setdefault("planner_model", {})["tokenizer_path"] = os.environ["PLANNER_MODEL_TOKENIZER_PATH"]
    if "PLANNER_MODEL_MAX_INPUT_TOKENS" in os.environ:
        raw.setdefault("planner_model", {})["max_input_tokens"] = int(os.environ["PLANNER_MODEL_MAX_INPUT_TOKENS"])
    if "PLANNER_MODEL_DEVICE" in os.environ:
        raw.setdefault("planner_model", {})["device"] = os.environ["PLANNER_MODEL_DEVICE"]

    # GLOBAL
    if "MODE" in os.environ:
        raw["mode"] = os.environ["MODE"]
    if "ENABLE_STEP3" in os.environ:
        raw["enable_step3"] = os.environ["ENABLE_STEP3"].lower() in ("1", "true", "yes")
    if "SUBGRAPH_TOTAL_CAP" in os.environ:
        raw["subgraph_total_cap"] = int(os.environ["SUBGRAPH_TOTAL_CAP"])
    if "DIR_DIVERSITY_K" in os.environ:
        raw["dir_diversity_k"] = int(os.environ["DIR_DIVERSITY_K"])
    if "PREFER_TEST_FILES" in os.environ:
        raw["prefer_test_files"] = os.environ["PREFER_TEST_FILES"].lower() in ("1", "true", "yes")
    if "MAX_TFILE_FRACTION" in os.environ:
        raw["max_tfile_fraction"] = float(os.environ["MAX_TFILE_FRACTION"])

    # LINT
    if "LINT_ENABLED" in os.environ:
        raw.setdefault("lint", {})["enabled"] = os.environ["LINT_ENABLED"].lower() in ("1", "true", "yes")
    if "LINT_FRAMEWORK" in os.environ:
        raw.setdefault("lint", {})["framework"] = os.environ["LINT_FRAMEWORK"]

    # TELEMETRY
    if "EVENTS_PATH" in os.environ:
        raw.setdefault("telemetry", {})["events_path"] = os.environ["EVENTS_PATH"]
    if "TEST_RUNS_PATH" in os.environ:
        raw.setdefault("telemetry", {})["test_runs_path"] = os.environ["TEST_RUNS_PATH"]

    return raw


def load() -> Config:
    """
    读取 .aci/config.json（若存在）并套用环境变量覆盖，返回 Config 对象。
    """
    cfg = Config()  # defaults
    # 1) 从文件合并
    file_path = os.environ.get("ACI_CONFIG", os.path.join(".aci", "config.json"))
    raw = _load_json_file(file_path)
    if raw:
        # dataclass -> dict
        merged = asdict(cfg)
        _deep_update(merged, raw)
        cfg = _dict_to_config(merged)

    # 2) 应用环境变量覆盖
    merged2 = asdict(cfg)
    merged2 = _apply_env_overrides(merged2)
    cfg = _dict_to_config(merged2)

    # 3) Collate 的 mode 默认继承全局 mode（若未指定）
    if not cfg.collate.mode:
        cfg.collate.mode = cfg.mode

    # 保底创建 logs 目录
    for path in (cfg.telemetry.events_path, cfg.telemetry.test_runs_path):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        except Exception:
            pass

    return cfg


def _dict_to_config(d: Dict[str, Any]) -> Config:
    # 子对象重建
    lint = LintCfg(**(d.get("lint") or {}))
    telemetry = TelemetryCfg(**(d.get("telemetry") or {}))
    collate = CollateCfg(**(d.get("collate") or {}))
    cgm = CGMCfg(**(d.get("cgm") or {}))
    planner_model = PlannerModelCfg(**(d.get("planner_model") or {}))

    return Config(
        mode=d.get("mode", "wsd"),
        enable_step3=bool(d.get("enable_step3", True)),

        subgraph_total_cap=int(d.get("subgraph_total_cap", 800)),
        plan_k=int(d.get("plan_k", 1)),
        max_nodes_per_anchor=int(d.get("max_nodes_per_anchor", 50)),
        candidate_total_limit=int(d.get("candidate_total_limit", 200)),
        dir_diversity_k=int(d.get("dir_diversity_k", 3)),

        prefer_test_files=bool(d.get("prefer_test_files", True)),
        max_tfile_fraction=float(d.get("max_tfile_fraction", 0.60)),

        lint=lint,
        telemetry=telemetry,
        collate=collate,
        cgm=cgm,
        planner_model=planner_model,

        memlog=d.get("memlog") or {"enabled": True},
    )
