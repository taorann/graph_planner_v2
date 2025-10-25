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

import argparse
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional
import os
import json
import sys
from copy import deepcopy

import yaml


def _detect_repo_root() -> Path:
    """Best-effort detection of the repository root for absolute paths."""

    env_root = os.environ.get("GRAPH_PLANNER_ROOT")
    if env_root:
        candidate = Path(env_root).expanduser()
        try:
            return candidate.resolve()
        except FileNotFoundError:
            pass

    for entry in list(sys.path):
        if not entry:
            continue
        candidate = Path(entry).expanduser()
        try:
            resolved = candidate.resolve()
        except FileNotFoundError:
            continue
        if (resolved / "graph_planner").exists():
            return resolved

    return Path(__file__).resolve().parents[2]


REPO_ROOT = _detect_repo_root()

PLANNER_MODEL_DIR = (REPO_ROOT / "models" / "Qwen3-14B").resolve()
CGM_MODEL_DIR = (REPO_ROOT / "models" / "CodeFuse-CGM").resolve()


# ---------------- dataclasses ----------------

@dataclass
class LintCfg:
    enabled: bool = True
    # 预留：你可以在 tools 层读这个字段决定走 ruff/flake8/black 等
    framework: str = "auto"


@dataclass
class TelemetryCfg:
    events_path: str = str(REPO_ROOT / "logs" / "events.jsonl")
    test_runs_path: str = str(REPO_ROOT / "logs" / "test_runs.jsonl")


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


# ---------------------------------------------------------------------------
# 训练/评估 YAML 配置解析（Pain Point #2 之后新增）
# ---------------------------------------------------------------------------


DEFAULT_TRAIN_DATASET = str((REPO_ROOT / "datasets" / "r2e_gym" / "train.jsonl").resolve())
DEFAULT_VAL_DATASET = str((REPO_ROOT / "datasets" / "r2e_gym" / "val.jsonl").resolve())


@dataclass
class ExperimentSection:
    name: str = "graph_planner_rl"
    seed: int = 42
    notes: Optional[str] = None


@dataclass
class PathsSection:
    dataset_train: str = DEFAULT_TRAIN_DATASET
    dataset_val: Optional[str] = DEFAULT_VAL_DATASET
    planner_model: str = str(PLANNER_MODEL_DIR)
    planner_tokenizer: Optional[str] = None
    cgm_model: str = str(CGM_MODEL_DIR)
    cgm_tokenizer: Optional[str] = None


@dataclass
class BackendsSection:
    planner_backend: str = "local"
    cgm_backend: str = "local"
    dtype: str = "fp32"
    device_map_planner: list[int] = field(default_factory=list)
    device_map_cgm: list[int] = field(default_factory=list)
    max_gpu_memory: Optional[str] = None


@dataclass
class SamplingSection:
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_new_tokens: Optional[int] = None
    repetition_penalty: Optional[float] = None
    do_sample: Optional[bool] = None
    stop_on_invalid_json: Optional[bool] = None
    stop: list[str] = field(default_factory=list)
    stop_ids: list[int] = field(default_factory=list)


@dataclass
class GenerationSection:
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_new_tokens: Optional[int] = None
    num_return_sequences: Optional[int] = None
    do_sample: Optional[bool] = None


@dataclass
class TrainingSection:
    total_epochs: int = 1
    train_batch_size: int = 4
    grad_accum_steps: int = 1
    precision: Optional[str] = None
    lr: Optional[float] = None
    weight_decay: Optional[float] = None
    warmup_steps: Optional[int] = None
    gradient_checkpointing: bool = False
    clip_grad_norm: Optional[float] = None
    kl_coef: Optional[float] = None
    entropy_coef: Optional[float] = None
    value_coef: Optional[float] = None
    clip_coef: Optional[float] = None
    target_kl: Optional[float] = None
    total_steps: Optional[int] = None
    resume_from: Optional[str] = None


@dataclass
class EnvSection:
    max_steps: int = 6
    reward_scale: float = 1.0
    failure_penalty: float = 0.0
    step_penalty: float = 0.0
    timeout_penalty: float = 0.0
    repo_op_limit: Optional[int] = None
    disable_cgm_synthesis: bool = False
    apply_patches: bool = True
    docker_manifest: Optional[str] = None
    prepull_containers: bool = False
    prepull_max_workers: Optional[int] = None
    prepull_retries: Optional[int] = None
    prepull_delay: Optional[int] = None
    prepull_timeout: Optional[int] = None


@dataclass
class ParallelSection:
    tensor_parallel_planner: int = 1
    tensor_parallel_cgm: int = 1
    replicas: int = 1
    parallel_agents: int = 1
    rollout_workers: int = 1
    workflow_parallel: int = 1


@dataclass
class ResourceSection:
    num_gpus: int = 1
    num_nodes: int = 1
    ray_num_gpus: int = 1
    ray_num_cpus: int = 4
    ray_memory: Optional[int] = None
    ray_object_store_memory: Optional[int] = None


@dataclass
class WandbWatchSection:
    enabled: bool = False
    log: str = "gradients"
    log_freq: int = 200


@dataclass
class WandbSection:
    enabled: bool = False
    offline: bool = False
    project: str = "graph-planner"
    entity: Optional[str] = None
    run_name: str = ""
    watch: WandbWatchSection = field(default_factory=WandbWatchSection)


@dataclass
class LoggingSection:
    wandb: WandbSection = field(default_factory=WandbSection)
    log_backend: str = "none"
    output_dir: str = PLANNER_MODEL_DIR
    save_interval: Optional[int] = None
    eval_interval: Optional[int] = None


@dataclass
class TelemetrySection:
    log_gpu: bool = True
    log_ray: bool = True
    log_patch_stats: bool = True
    log_planner_parse_errors: bool = True
    log_cgm_errors: bool = True


@dataclass
class TrainingRunConfig:
    experiment: ExperimentSection = field(default_factory=ExperimentSection)
    paths: PathsSection = field(default_factory=PathsSection)
    backends: BackendsSection = field(default_factory=BackendsSection)
    planner_sampling: SamplingSection = field(default_factory=SamplingSection)
    cgm_generation: GenerationSection = field(default_factory=GenerationSection)
    training: TrainingSection = field(default_factory=TrainingSection)
    env: EnvSection = field(default_factory=EnvSection)
    parallel: ParallelSection = field(default_factory=ParallelSection)
    resources: ResourceSection = field(default_factory=ResourceSection)
    logging: LoggingSection = field(default_factory=LoggingSection)
    telemetry: TelemetrySection = field(default_factory=TelemetrySection)
    verl_overrides: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data


def _deepcopy_dict(data: Mapping[str, Any]) -> Dict[str, Any]:
    return deepcopy(dict(data))


def _ensure_run_name(config: Dict[str, Any], agent: str) -> None:
    wandb_cfg = config.setdefault("logging", {}).setdefault("wandb", {})
    if not wandb_cfg.get("run_name"):
        wandb_cfg["run_name"] = f"{agent}-run"


def default_training_run_config(agent: str) -> Dict[str, Any]:
    base = TrainingRunConfig().to_dict()
    if agent == "planner":
        base["logging"]["output_dir"] = str(PLANNER_MODEL_DIR)
        base["paths"]["planner_model"] = str(PLANNER_MODEL_DIR)
    else:
        base["logging"]["output_dir"] = str(CGM_MODEL_DIR)
        base["paths"]["planner_model"] = str(PLANNER_MODEL_DIR)
        base["paths"]["cgm_model"] = str(CGM_MODEL_DIR)
    _ensure_run_name(base, agent)
    return base


def _load_yaml_dict(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    resolved = _resolve_pathlike(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Config file {resolved} does not exist")
    with open(resolved, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError("Top-level YAML config must be a mapping")
    return data


def load_run_config_file(path: Optional[Path]) -> Dict[str, Any]:
    """Public helper to read YAML config files for training/eval launches."""

    return _load_yaml_dict(path)


def _resolve_pathlike(value: Path | str) -> Path:
    """Expand user components and return an absolute path."""

    return Path(value).expanduser().resolve()


def _normalise_cli_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(_resolve_pathlike(value))
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        return [_normalise_cli_value(v) for v in value]
    return value


def _cli_override_entry(namespace: argparse.Namespace, attr: str) -> Any:  # type: ignore[name-defined]
    specified = getattr(namespace, "_specified_cli_args", None)
    if specified is not None and attr not in specified:
        return None
    if not hasattr(namespace, attr):  # pragma: no cover - defensive guard
        return None
    value = getattr(namespace, attr)
    if value is None:
        return None
    return _normalise_cli_value(value)


def build_cli_overrides(args: Any, *, mode: str) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}

    seed = _cli_override_entry(args, "seed")
    if seed is not None:
        overrides.setdefault("experiment", {})["seed"] = int(seed)

    dataset = _cli_override_entry(args, "dataset")
    if dataset is not None:
        overrides.setdefault("paths", {})["dataset_train"] = dataset

    val_dataset = _cli_override_entry(args, "val_dataset")
    if val_dataset is not None:
        overrides.setdefault("paths", {})["dataset_val"] = val_dataset

    model_path = _cli_override_entry(args, "model_path")
    if model_path is not None:
        key = "planner_model" if getattr(args, "agent", "planner") == "planner" else "cgm_model"
        overrides.setdefault("paths", {})[key] = model_path

    tokenizer_path = _cli_override_entry(args, "tokenizer_path")
    if tokenizer_path is not None:
        overrides.setdefault("paths", {})["planner_tokenizer"] = tokenizer_path

    cgm_model_path = _cli_override_entry(args, "cgm_model_path")
    if cgm_model_path is not None:
        overrides.setdefault("paths", {})["cgm_model"] = cgm_model_path

    cgm_tokenizer_path = _cli_override_entry(args, "cgm_tokenizer_path")
    if cgm_tokenizer_path is not None:
        overrides.setdefault("paths", {})["cgm_tokenizer"] = cgm_tokenizer_path

    precision = _cli_override_entry(args, "precision")
    if precision is not None:
        overrides.setdefault("training", {})["precision"] = precision

    for field in ["train_batch_size", "total_epochs", "grad_accum_steps", "lr", "weight_decay", "warmup_steps"]:
        value = _cli_override_entry(args, field)
        if value is not None:
            overrides.setdefault("training", {})[field if field != "total_epochs" else "total_epochs"] = value

    total_steps = _cli_override_entry(args, "total_steps")
    if total_steps is not None:
        overrides.setdefault("training", {})["total_steps"] = total_steps

    resume_from = _cli_override_entry(args, "resume")
    if resume_from is not None:
        overrides.setdefault("training", {})["resume_from"] = resume_from

    temperature = _cli_override_entry(args, "temperature")
    if temperature is not None:
        overrides.setdefault("planner_sampling", {})["temperature"] = float(temperature)

    top_p = _cli_override_entry(args, "top_p")
    if top_p is not None:
        overrides.setdefault("planner_sampling", {})["top_p"] = float(top_p)

    max_input_tokens = _cli_override_entry(args, "max_input_tokens")
    if max_input_tokens is not None:
        overrides.setdefault("planner_sampling", {})["max_input_tokens"] = int(max_input_tokens)

    max_output_tokens = _cli_override_entry(args, "max_output_tokens")
    if max_output_tokens is not None:
        overrides.setdefault("planner_sampling", {})["max_new_tokens"] = int(max_output_tokens)

    stop = _cli_override_entry(args, "stop")
    if stop is not None:
        overrides.setdefault("planner_sampling", {})["stop"] = stop

    stop_ids = _cli_override_entry(args, "stop_ids")
    if stop_ids is not None:
        overrides.setdefault("planner_sampling", {})["stop_ids"] = [int(s) for s in stop_ids]

    tensor_parallel = _cli_override_entry(args, "tensor_parallel")
    if tensor_parallel is not None:
        overrides.setdefault("parallel", {})["tensor_parallel_planner"] = int(tensor_parallel)
        overrides.setdefault("parallel", {})["tensor_parallel_cgm"] = int(tensor_parallel)
    replicas = _cli_override_entry(args, "rollout_replicas")
    if replicas is not None:
        overrides.setdefault("parallel", {})["replicas"] = int(replicas)

    parallel_agents = _cli_override_entry(args, "parallel_agents")
    if parallel_agents is not None:
        overrides.setdefault("parallel", {})["parallel_agents"] = int(parallel_agents)

    rollout_workers = _cli_override_entry(args, "rollout_workers")
    if rollout_workers is not None:
        overrides.setdefault("parallel", {})["rollout_workers"] = int(rollout_workers)

    workflow_parallel = _cli_override_entry(args, "workflow_parallel")
    if workflow_parallel is not None:
        overrides.setdefault("parallel", {})["workflow_parallel"] = int(workflow_parallel)

    num_gpus = _cli_override_entry(args, "num_gpus")
    if num_gpus is not None:
        overrides.setdefault("resources", {})["num_gpus"] = int(num_gpus)
    num_nodes = _cli_override_entry(args, "num_nodes")
    if num_nodes is not None:
        overrides.setdefault("resources", {})["num_nodes"] = int(num_nodes)

    ray_num_gpus = _cli_override_entry(args, "ray_num_gpus")
    if ray_num_gpus is not None:
        overrides.setdefault("resources", {})["ray_num_gpus"] = int(ray_num_gpus)

    ray_num_cpus = _cli_override_entry(args, "ray_num_cpus")
    if ray_num_cpus is not None:
        overrides.setdefault("resources", {})["ray_num_cpus"] = int(ray_num_cpus)

    ray_memory = _cli_override_entry(args, "ray_memory")
    if ray_memory is not None:
        overrides.setdefault("resources", {})["ray_memory"] = int(ray_memory)

    ray_object_store_memory = _cli_override_entry(args, "ray_object_store_memory")
    if ray_object_store_memory is not None:
        overrides.setdefault("resources", {})["ray_object_store_memory"] = int(ray_object_store_memory)

    log_backend = _cli_override_entry(args, "log_backend")
    if log_backend is not None:
        overrides.setdefault("logging", {})["log_backend"] = log_backend

    output_dir = _cli_override_entry(args, "output_dir")
    if output_dir is not None:
        overrides.setdefault("logging", {})["output_dir"] = output_dir

    save_interval = _cli_override_entry(args, "save_interval")
    if save_interval is not None:
        overrides.setdefault("logging", {})["save_interval"] = int(save_interval)

    eval_interval = _cli_override_entry(args, "eval_interval")
    if eval_interval is not None:
        overrides.setdefault("logging", {})["eval_interval"] = int(eval_interval)

    log_to_wandb = bool(getattr(args, "log_to_wandb", False))
    if log_to_wandb:
        overrides.setdefault("logging", {}).setdefault("wandb", {})["enabled"] = True

    if getattr(args, "wandb_offline", False):
        overrides.setdefault("logging", {}).setdefault("wandb", {})["offline"] = True

    project_name = _cli_override_entry(args, "project_name")
    if project_name is not None:
        overrides.setdefault("logging", {}).setdefault("wandb", {})["project"] = project_name

    experiment_name = _cli_override_entry(args, "experiment_name")
    if experiment_name is not None:
        overrides.setdefault("logging", {}).setdefault("wandb", {})["run_name"] = experiment_name

    env_fields = {
        "max_steps": int(getattr(args, "max_steps", 6) or 6),
        "reward_scale": _cli_override_entry(args, "reward_scale"),
        "failure_penalty": _cli_override_entry(args, "failure_penalty"),
        "step_penalty": _cli_override_entry(args, "step_penalty"),
        "timeout_penalty": _cli_override_entry(args, "timeout_penalty"),
        "repo_op_limit": _cli_override_entry(args, "repo_op_limit"),
        "docker_manifest": _cli_override_entry(args, "docker_manifest"),
        "prepull_max_workers": _cli_override_entry(args, "prepull_max_workers"),
        "prepull_retries": _cli_override_entry(args, "prepull_retries"),
        "prepull_delay": _cli_override_entry(args, "prepull_delay"),
        "prepull_timeout": _cli_override_entry(args, "prepull_timeout"),
    }
    env_overrides = {k: v for k, v in env_fields.items() if v is not None}
    for key in ("prepull_max_workers", "prepull_retries", "prepull_delay", "prepull_timeout", "repo_op_limit"):
        if key in env_overrides and env_overrides[key] is not None:
            env_overrides[key] = int(env_overrides[key])
    if env_overrides:
        overrides.setdefault("env", {}).update(env_overrides)

    if getattr(args, "disable_cgm_synthesis", False):
        overrides.setdefault("env", {})["disable_cgm_synthesis"] = True

    if getattr(args, "apply_patches", None) is not None:
        overrides.setdefault("env", {})["apply_patches"] = bool(args.apply_patches)

    if getattr(args, "prepull_containers", False):
        overrides.setdefault("env", {})["prepull_containers"] = True

    return overrides


def merge_run_config(
    defaults: Dict[str, Any],
    yaml_cfg: Mapping[str, Any],
    cli_overrides: Mapping[str, Any],
    *,
    yaml_only: bool,
    agent: Optional[str] = None,
) -> Dict[str, Any]:
    merged = _deepcopy_dict(defaults)
    yaml_copy = _deepcopy_dict(yaml_cfg or {})

    agent_section: Dict[str, Any] = {}
    if agent and agent in yaml_copy:
        section = yaml_copy.pop(agent)
        if section is None:
            agent_section = {}
        elif isinstance(section, Mapping):
            agent_section = _deepcopy_dict(section)
        else:
            raise TypeError(f"Agent-specific config for '{agent}' must be a mapping")

    _deep_update(merged, yaml_copy)
    if agent_section:
        _deep_update(merged, agent_section)

    if not yaml_only:
        _deep_update(merged, cli_overrides or {})
    return merged


def update_args_from_config(
    args: Any, config: Mapping[str, Any], *, respect_cli: bool = True
) -> None:
    specified: set[str] = (
        set(getattr(args, "_specified_cli_args", set())) if respect_cli else set()
    )

    def _can_set(field: str) -> bool:
        return hasattr(args, field) and field not in specified

    def _set_value(field: str, value: Any, *, transform: Optional[Any] = None) -> None:
        if value is None or not _can_set(field):
            return
        setattr(args, field, transform(value) if transform else value)

    def _set_path(field: str, value: Any) -> None:
        _set_value(field, value, transform=_resolve_pathlike)

    def _set_int(field: str, value: Any) -> None:
        _set_value(field, value, transform=int)

    def _set_float(field: str, value: Any) -> None:
        _set_value(field, value, transform=float)

    paths = config.get("paths", {})
    agent = getattr(args, "agent", "planner")
    model_key = "planner_model" if agent == "planner" else "cgm_model"
    model_path = paths.get(model_key)
    if model_path:
        _set_path("model_path", model_path)
    planner_tokenizer = paths.get("planner_tokenizer")
    if planner_tokenizer:
        _set_path("tokenizer_path", planner_tokenizer)
    cgm_model = paths.get("cgm_model")
    if cgm_model:
        _set_path("cgm_model_path", cgm_model)
    cgm_tokenizer = paths.get("cgm_tokenizer")
    if cgm_tokenizer:
        _set_path("cgm_tokenizer_path", cgm_tokenizer)

    dataset_train = paths.get("dataset_train")
    if dataset_train:
        _set_path("dataset", dataset_train)
    dataset_val = paths.get("dataset_val")
    if dataset_val:
        _set_path("val_dataset", dataset_val)

    experiment = config.get("experiment", {})
    if "seed" in experiment:
        _set_int("seed", experiment["seed"])

    training = config.get("training", {})
    if "train_batch_size" in training:
        train_bs = int(training["train_batch_size"])
        _set_value("train_batch_size", train_bs)
        if hasattr(args, "batch_size"):
            _set_value("batch_size", train_bs)
    if "total_epochs" in training and training["total_epochs"] is not None:
        _set_int("total_epochs", training["total_epochs"])
    if training.get("grad_accum_steps") is not None:
        _set_int("grad_accum_steps", training["grad_accum_steps"])
    if training.get("precision"):
        _set_value("precision", training["precision"])
    if training.get("lr") is not None:
        _set_float("lr", training["lr"])
    if training.get("weight_decay") is not None:
        _set_float("weight_decay", training["weight_decay"])
    if training.get("warmup_steps") is not None:
        _set_int("warmup_steps", training["warmup_steps"])
    if training.get("total_steps") is not None:
        _set_int("total_steps", training["total_steps"])
    if training.get("resume_from"):
        _set_path("resume", training["resume_from"])

    sampling = config.get("planner_sampling", {})
    if sampling.get("temperature") is not None:
        _set_float("temperature", sampling["temperature"])
    if sampling.get("top_p") is not None:
        _set_float("top_p", sampling["top_p"])
    if sampling.get("max_new_tokens") is not None:
        _set_int("max_output_tokens", sampling["max_new_tokens"])
    if sampling.get("max_input_tokens") is not None:
        _set_int("max_input_tokens", sampling["max_input_tokens"])
    if sampling.get("stop") and _can_set("stop"):
        setattr(args, "stop", list(sampling["stop"]))
    if sampling.get("stop_ids") and _can_set("stop_ids"):
        setattr(args, "stop_ids", [int(v) for v in sampling["stop_ids"]])

    parallel = config.get("parallel", {})
    tp_planner = int(parallel.get("tensor_parallel_planner", 1) or 1)
    tp_cgm = int(parallel.get("tensor_parallel_cgm", 1) or 1)
    target_tp = tp_planner if agent == "planner" else tp_cgm
    _set_int("tensor_parallel", target_tp)
    replicas = int(parallel.get("replicas", 1) or 1)
    _set_int("rollout_replicas", replicas)
    if parallel.get("parallel_agents") is not None:
        _set_int("parallel_agents", parallel["parallel_agents"])
    if parallel.get("rollout_workers") is not None:
        _set_int("rollout_workers", parallel["rollout_workers"])
    if parallel.get("workflow_parallel") is not None:
        _set_int("workflow_parallel", parallel["workflow_parallel"])

    resources = config.get("resources", {})
    if resources.get("num_gpus") is not None:
        _set_int("num_gpus", resources["num_gpus"])
    if resources.get("num_nodes") is not None:
        _set_int("num_nodes", resources["num_nodes"])
    if resources.get("ray_num_gpus") is not None:
        _set_int("ray_num_gpus", resources["ray_num_gpus"])
    if resources.get("ray_num_cpus") is not None:
        _set_int("ray_num_cpus", resources["ray_num_cpus"])
    if resources.get("ray_memory") is not None:
        _set_int("ray_memory", resources["ray_memory"])
    if resources.get("ray_object_store_memory") is not None:
        _set_int("ray_object_store_memory", resources["ray_object_store_memory"])

    env_cfg = config.get("env", {})
    if env_cfg.get("max_steps") is not None:
        _set_int("max_steps", env_cfg["max_steps"])
    if env_cfg.get("reward_scale") is not None:
        _set_float("reward_scale", env_cfg["reward_scale"])
    if env_cfg.get("failure_penalty") is not None:
        _set_float("failure_penalty", env_cfg["failure_penalty"])
    if env_cfg.get("step_penalty") is not None:
        _set_float("step_penalty", env_cfg["step_penalty"])
    if env_cfg.get("timeout_penalty") is not None:
        _set_float("timeout_penalty", env_cfg["timeout_penalty"])
    if env_cfg.get("repo_op_limit") is not None:
        _set_int("repo_op_limit", env_cfg["repo_op_limit"])
    if env_cfg.get("disable_cgm_synthesis") is not None:
        _set_value("disable_cgm_synthesis", bool(env_cfg["disable_cgm_synthesis"]))
    if env_cfg.get("apply_patches") is not None:
        _set_value("apply_patches", bool(env_cfg["apply_patches"]))
    if env_cfg.get("docker_manifest"):
        _set_path("docker_manifest", env_cfg["docker_manifest"])
    if env_cfg.get("prepull_containers") is not None:
        _set_value("prepull_containers", bool(env_cfg["prepull_containers"]))
    for field in ("prepull_max_workers", "prepull_retries", "prepull_delay", "prepull_timeout"):
        if env_cfg.get(field) is not None:
            _set_int(field, env_cfg[field])

    logging_cfg = config.get("logging", {})
    wandb_cfg = logging_cfg.get("wandb", {})
    if wandb_cfg.get("project"):
        _set_value("project_name", wandb_cfg["project"])
    if wandb_cfg.get("run_name"):
        _set_value("experiment_name", wandb_cfg["run_name"])
    if wandb_cfg.get("enabled") is not None:
        _set_value("log_to_wandb", bool(wandb_cfg["enabled"]))
    if wandb_cfg.get("offline") is not None:
        _set_value("wandb_offline", bool(wandb_cfg["offline"]))

    if logging_cfg.get("log_backend") is not None:
        _set_value("log_backend", logging_cfg["log_backend"])
    if logging_cfg.get("save_interval") is not None:
        _set_value("save_interval", logging_cfg["save_interval"])
    if logging_cfg.get("eval_interval") is not None:
        _set_value("eval_interval", logging_cfg["eval_interval"])
    if logging_cfg.get("output_dir") is not None:
        _set_path("output_dir", logging_cfg["output_dir"])



def serialise_resolved_config(config: Mapping[str, Any], path: Path) -> None:
    resolved_path = _resolve_pathlike(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=False, allow_unicode=True)



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
