# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import inspect
import itertools
import json
import json as _json
import logging
import os
import uuid
from collections import defaultdict
from collections.abc import Mapping as MappingABC, Sequence as SequenceABC
from functools import lru_cache
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Any, Mapping, Optional, Tuple

import numpy as np
import ray
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torch.utils.data._utils.collate import default_collate as _default_collate
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger

from rllm.verl.verl.workers.cgm_service import CGMService

WorkerType = type[Worker]


LOG = logging.getLogger(__name__)


_CONTAINER_ATTR_HINTS = (
    "children",
    "subgroups",
    "members",
    "groups",
    "worker_groups",
    "workers",
    "pools",
    "pool",
    "resource_pools",
    "resource_pool",
    "actors",
    "actor_groups",
    "groups_dict",
)

_CONTAINER_KEYWORD_HINTS = (
    "group",
    "worker",
    "pool",
    "actor",
    "member",
    "child",
    "rollout",
    "ref",
)

_CONTAINER_PREVIEW_LIMIT = 32


def _is_container(obj: Any) -> bool:
    if obj is None:
        return False
    if isinstance(obj, MappingABC):
        return True
    if isinstance(obj, set):
        return True
    if isinstance(obj, SequenceABC) and not isinstance(obj, (str, bytes, bytearray)):
        return True
    return False


def _yield_container_items(obj: Any):
    if isinstance(obj, MappingABC):
        for item in obj.values():
            if item is not None:
                yield item
    elif isinstance(obj, set):
        for item in obj:
            if item is not None:
                yield item
    elif isinstance(obj, SequenceABC) and not isinstance(obj, (str, bytes, bytearray)):
        for item in obj:
            if item is not None:
                yield item


def _container_len(obj: Any) -> Any:
    try:
        return len(obj)
    except Exception:
        return "?"


def _is_group_like(obj: Any) -> bool:
    if obj is None:
        return False
    for attr in ("spawn", "start", "create", "name", "prefix"):
        if hasattr(obj, attr):
            return True
    if _is_container(obj):
        return True
    for attr in _discover_container_attrs(obj.__class__):
        if hasattr(obj, attr):
            return True
    return False


@lru_cache(maxsize=256)
def _discover_container_attrs(cls: type) -> tuple[str, ...]:
    names: list[str] = []
    for attr in _CONTAINER_ATTR_HINTS:
        if hasattr(cls, attr):
            names.append(attr)
    annotations = getattr(cls, "__annotations__", {}) or {}
    for attr in annotations:
        if attr in names:
            continue
        low = attr.lower()
        if any(keyword in low for keyword in _CONTAINER_KEYWORD_HINTS):
            names.append(attr)
    for attr in dir(cls):
        if attr in names or attr.startswith("__"):
            continue
        low = attr.lower()
        if any(keyword in low for keyword in _CONTAINER_KEYWORD_HINTS):
            names.append(attr)
    return tuple(dict.fromkeys(names))


def _iter_group_children(group: Any):
    if group is None:
        return
    if _is_container(group):
        for child in _yield_container_items(group):
            yield child
        return

    candidate_attrs = list(_discover_container_attrs(group.__class__))
    if hasattr(group, "__dict__"):
        for key in list(group.__dict__.keys())[:128]:
            if not isinstance(key, str):
                continue
            if key in candidate_attrs:
                continue
            low = key.lower()
            if any(keyword in low for keyword in _CONTAINER_KEYWORD_HINTS):
                candidate_attrs.append(key)

    seen: set[str] = set()
    for attr in candidate_attrs:
        if not isinstance(attr, str):
            continue
        if attr in seen:
            continue
        seen.add(attr)
        if not hasattr(group, attr):
            continue
        try:
            value = getattr(group, attr)
        except Exception:
            continue
        if value is None:
            continue
        if _is_container(value):
            iterator = _yield_container_items(value)
            preview = list(itertools.islice(iterator, _CONTAINER_PREVIEW_LIMIT))
            if not preview:
                continue
            has_group_like = any(_is_group_like(item) for item in preview)
            if not has_group_like:
                for item in iterator:
                    preview.append(item)
                    if _is_group_like(item):
                        has_group_like = True
                        break
            if not has_group_like:
                continue
            for item in preview:
                if item is not None:
                    yield item
            for item in iterator:
                if item is not None:
                    yield item
            continue
        if _is_group_like(value):
            yield value


def _safe_spawn(wg, prefix_set=None, world_size=None, _visited=None, **kwargs):
    if wg is None:
        if prefix_set:
            return {p: [] for p in prefix_set if isinstance(p, str)}
        return []

    if _visited is None:
        _visited = set()

    obj_id = id(wg)
    if obj_id in _visited:
        return []
    _visited.add(obj_id)

    def _matches_prefix(group) -> bool:
        if not prefix_set:
            return True
        names = []
        for attr in ("name", "prefix"):
            value = getattr(group, attr, None)
            if isinstance(value, str):
                names.append(value)
        if not names:
            return False
        for prefix in prefix_set:
            if not isinstance(prefix, str):
                continue
            for value in names:
                if value == prefix or value.startswith(prefix):
                    return True
        return False

    methods = ("spawn", "start", "create")

    def _method_accepts_prefix(group) -> bool:
        if not prefix_set:
            return False
        for name in methods:
            method = getattr(group, name, None)
            if method is None:
                continue
            try:
                sig = inspect.signature(method)
            except (TypeError, ValueError):
                continue
            if "prefix_set" in sig.parameters:
                return True
        return False

    def _call_spawn(group):
        for name in methods:
            method = getattr(group, name, None)
            if method is None:
                continue
            call_kwargs = dict(kwargs)
            try:
                sig = inspect.signature(method)
            except (TypeError, ValueError):
                sig = None
            parameters = sig.parameters if sig is not None else {}
            if prefix_set is not None and "prefix_set" in parameters:
                call_kwargs["prefix_set"] = prefix_set
            elif prefix_set is not None and "prefix_set" not in parameters:
                call_kwargs.pop("prefix_set", None)
            if world_size is not None and "world_size" in parameters:
                call_kwargs["world_size"] = world_size
            elif "world_size" not in parameters:
                call_kwargs.pop("world_size", None)
            try:
                result = method(**call_kwargs)
            except TypeError:
                continue
            return result
        return None

    results_map: dict[str, list[Any]] = {}
    results_list: list[Any] = []

    def _collect(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, MappingABC):
            for key, item in value.items():
                if not isinstance(key, str):
                    continue
                if item is None:
                    continue
                bucket = results_map.setdefault(key, [])
                if isinstance(item, MappingABC):
                    for sub in item.values():
                        if sub is None:
                            continue
                        if _is_container(sub):
                            bucket.extend(_yield_container_items(sub))
                        else:
                            bucket.append(sub)
                elif _is_container(item):
                    bucket.extend(_yield_container_items(item))
                else:
                    bucket.append(item)
            return
        if _is_container(value):
            for item in _yield_container_items(value):
                _collect(item)
            return
        results_list.append(value)

    if not _is_container(wg) and (_matches_prefix(wg) or _method_accepts_prefix(wg)):
        spawn_result = _call_spawn(wg)
        _collect(spawn_result)

    for child in _iter_group_children(wg):
        child_result = _safe_spawn(
            child,
            prefix_set=prefix_set,
            world_size=world_size,
            _visited=_visited,
            **kwargs,
        )
        _collect(child_result)

    if results_map:
        return results_map

    if prefix_set:
        inferred: dict[str, list[Any]] = {}
        str_prefixes = [p for p in prefix_set if isinstance(p, str)]
        for item in results_list:
            names: list[str] = []
            for attr in ("name", "prefix", "sub_cls_name"):
                value = getattr(item, attr, None)
                if isinstance(value, str):
                    names.append(value)
            for prefix in list(str_prefixes):
                bucket = inferred.setdefault(prefix, [])
                for name in names:
                    if name == prefix or name.startswith(prefix):
                        bucket.append(item)
                        break
        if not inferred and len(str_prefixes) == 1 and results_list:
            inferred[str_prefixes[0]] = list(results_list)
        if inferred:
            return inferred

    return results_list


def _safe_spawn_map(wg, wanted_prefixes, world_size=None, **kwargs) -> dict[str, list[Any]]:
    """Return a prefix-keyed mapping of spawned worker groups."""

    wanted = [p for p in (wanted_prefixes or []) if isinstance(p, str)]
    prefix_set = set(wanted)
    spawn_result = _safe_spawn(
        wg,
        prefix_set=prefix_set if prefix_set else None,
        world_size=world_size,
        **kwargs,
    )

    mapping: dict[str, list[Any]] = {key: [] for key in wanted}

    if isinstance(spawn_result, MappingABC):
        for key, value in spawn_result.items():
            if key not in mapping or value is None:
                continue
            if isinstance(value, MappingABC):
                for sub in value.values():
                    if sub is None:
                        continue
                    if _is_container(sub):
                        mapping[key].extend(_yield_container_items(sub))
                    else:
                        mapping[key].append(sub)
            elif _is_container(value):
                mapping[key].extend(_yield_container_items(value))
            else:
                mapping[key].append(value)
        return mapping

    if isinstance(spawn_result, (list, tuple, set)) or spawn_result is None:
        return mapping

    if wanted and spawn_result is not None:
        # No prefix information available; return explicit empty buckets.
        return mapping

    return mapping


def _get_sp_from_cfg(cfg) -> int:
    actor_rollout_ref = _oc_get(cfg, "actor_rollout_ref", None)
    actor = _oc_get(actor_rollout_ref, "actor", None)
    if actor is None:
        return 1
    enable = bool(_oc_get(actor, "enable_sequence_parallel", False) or False)
    sp_raw = _oc_get(actor, "ulysses_sequence_parallel_size", 1) or 1
    try:
        sp = int(sp_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"actor.ulysses_sequence_parallel_size must be integer-compatible, got {sp_raw!r}"
        ) from exc
    if not enable or sp <= 1:
        if sp > 1 and not enable:
            LOG.info(
                "[trainer] SequenceParallel requested (sp=%s) but disabled by flag; forcing sp=1",
                sp,
            )
        return 1
    return max(1, sp)


@dataclass
class VLLMGroupConfig:
    tp: int = 1
    dp: int = 1
    gpu_memory_utilization: float | None = None
    max_model_len: int | None = None
    max_num_seqs: int | None = None
    kv_cache_dtype: str | None = None
    enforce_eager: bool | None = None
    model_path: str | None = None
    adapters: Any | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TopologyGroupConfig:
    name: str
    gpus: list[int]
    fsdp_size: int | None = None
    vllm: VLLMGroupConfig | None = None


class StaticVLLMEngine:
    """Metadata container that can optionally host a real inference runtime."""

    def __init__(
        self,
        *,
        group: str,
        tp: int,
        dp: int,
        gpus: list[int],
        max_model_len: int | None = None,
        gpu_memory_utilization: float | None = None,
        max_num_seqs: int | None = None,
        kv_cache_dtype: str | None = None,
        enforce_eager: bool | None = None,
        model_path: str | None = None,
        adapters: Any | None = None,
        extra_config: Mapping | None = None,
    ) -> None:
        self.group = group
        self.tp = tp
        self.dp = dp
        self.gpus = list(gpus)
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_seqs = max_num_seqs
        self.kv_cache_dtype = kv_cache_dtype
        self.enforce_eager = enforce_eager
        self.model_path = model_path
        self.adapters = adapters
        self.extra_config = dict(extra_config or {})
        self._latest_snapshot: int | None = None
        self._runtime: Any | None = None
        self._runtime_kind: str = "static"
        self._sampling_defaults: dict[str, Any] = dict(
            _oc_get(self.extra_config, "sampling_params", {}) or {}
        )

    def set_snapshot(self, version: int, **metadata: Any) -> None:
        self._latest_snapshot = version
        if metadata:
            self.extra_config.setdefault("snapshot_metadata", {}).update(metadata)

    # ------------------------------------------------------------------
    # Runtime helpers
    # ------------------------------------------------------------------
    def attach_runtime(
        self,
        runtime: Any | None,
        *,
        kind: str = "static",
        sampling_defaults: Mapping[str, Any] | None = None,
    ) -> None:
        self._runtime = runtime
        self._runtime_kind = kind
        if sampling_defaults is not None:
            self._sampling_defaults = dict(sampling_defaults)

    def has_runtime(self) -> bool:
        return self._runtime is not None

    def set_max_num_seqs(self, value: int) -> None:
        self.max_num_seqs = value
        runtime = self._runtime
        if runtime is None:
            return
        if hasattr(runtime, "set_max_num_seqs"):
            try:
                runtime.set_max_num_seqs(value)
            except Exception:
                return

    def generate(self, prompts, **kwargs):
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError(
                f"vLLM runtime not available for group={self.group}; check deployment"
            )
        if self._runtime_kind == "vllm":
            kwargs.setdefault("sampling_kwargs", self._sampling_defaults)
        return runtime.generate(prompts, **kwargs)


class PlannerToVLLMSyncer:
    """Book-keeping helper to track policy→inference synchronization."""

    def __init__(self, trainer_wg: RayWorkerGroup | None, engine: StaticVLLMEngine | None) -> None:
        self._trainer = trainer_wg
        self._engine = engine
        self._published_version: int | None = None

    def sync(self) -> None:
        """Synchronize the latest learner weights to rollout engines."""

        if self._trainer is not None:
            for attr in ("sync_parameters", "broadcast_parameters", "sync_model"):
                try:
                    getattr(self._trainer, attr)()
                    break
                except AttributeError:
                    continue

    def publish_snapshot(self, version: int, **metadata: Any) -> int:
        """Record the latest policy weights snapshot for rollout engines."""

        self._published_version = version
        if self._engine is not None:
            self._engine.set_snapshot(version, **metadata)
        return version

    @property
    def latest_version(self) -> int | None:
        return self._published_version


class _VLLMRuntimeWrapper:
    """Adapter that exposes a simplified ``generate`` API around vLLM."""

    def __init__(self, llm, *, sampling_defaults: Mapping[str, Any] | None = None) -> None:
        self._llm = llm
        self._sampling_defaults = dict(sampling_defaults or {})

    def set_max_num_seqs(self, value: int) -> None:
        try:
            engine = getattr(self._llm, "llm_engine", None)
            scheduler = getattr(engine, "scheduler_config", None)
            if scheduler is not None and hasattr(scheduler, "max_num_seqs"):
                scheduler.max_num_seqs = int(value)
        except Exception:
            pass

    def generate(self, prompts, *, sampling_kwargs: Mapping[str, Any] | None = None, **kwargs):
        try:
            from vllm import SamplingParams  # type: ignore
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError("vLLM SamplingParams unavailable; ensure vllm is installed") from exc

        params = dict(self._sampling_defaults)
        if sampling_kwargs:
            params.update(dict(sampling_kwargs))
        sampling_params = SamplingParams(**params)
        outputs = self._llm.generate(prompts, sampling_params, **kwargs)
        results: list[str] = []
        for out in outputs:
            text = None
            if hasattr(out, "outputs") and out.outputs:
                first = out.outputs[0]
                text = getattr(first, "text", None)
            if text is None:
                text = getattr(out, "text", None)
            if text is None:
                text = str(out)
            results.append(text)
        return results


class _CodeFuseRuntimeWrapper:
    """Adapter that calls the CodeFuse CGM HTTP service."""

    def __init__(self, client) -> None:
        try:
            from aci.schema import Plan
        except Exception as exc:  # pragma: no cover - informative
            raise RuntimeError("aci.schema.Plan is required for CodeFuse CGM runtime") from exc
        self._client = client
        self._plan_cls = Plan

    def _coerce_plan(self, payload: Any):
        if isinstance(payload, self._plan_cls):
            return payload
        if isinstance(payload, MappingABC):
            try:
                return self._plan_cls(**payload)
            except Exception:
                pass
        return self._plan_cls(targets=[], budget={}, priority_tests=[])

    def generate(self, prompts, **kwargs):
        responses: list[dict[str, Any]] = []
        for prompt in prompts:
            if isinstance(prompt, str):
                try:
                    data = _json.loads(prompt)
                except _json.JSONDecodeError:
                    data = {}
            elif isinstance(prompt, MappingABC):
                data = dict(prompt)
            else:
                data = {}

            collated = data.get("collated") or {}
            plan_payload = data.get("plan_struct") or data.get("plan") or {}
            plan = self._coerce_plan(plan_payload)
            patch = self._client.generate_patch(
                issue=data.get("issue"),
                plan=plan,
                plan_text=data.get("plan_text"),
                subgraph_linearized=collated.get("chunks"),
                snippets=collated.get("snippets"),
                metadata=data.get("constraints"),
            )
            responses.append(dict(patch))
        return responses


def build_vllm_engine(
    *,
    group: str,
    gpus: list[int],
    tp: int,
    dp: int = 1,
    max_model_len: int | None = None,
    gpu_memory_utilization: float | None = None,
    max_num_seqs: int | None = None,
    kv_cache_dtype: str | None = None,
    enforce_eager: bool | None = None,
    model_path: str | None = None,
    adapters: Any | None = None,
    extra_config: Mapping | None = None,
) -> StaticVLLMEngine:
    """Construct a topology-aware vLLM engine, attaching a runtime when possible."""

    extra = dict(extra_config or {})
    sampling_defaults = dict(extra.pop("sampling_params", {}) or {})
    backend = str(extra.get("backend", "") or extra.get("runtime", "")).lower()

    engine = StaticVLLMEngine(
        group=group,
        tp=tp,
        dp=dp,
        gpus=gpus,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
        kv_cache_dtype=kv_cache_dtype,
        enforce_eager=enforce_eager,
        model_path=model_path,
        adapters=adapters,
        extra_config=extra_config,
    )

    runtime = None
    runtime_kind = "static"

    try:
        if backend in {"codefuse", "codefuse-cgm"} or extra.get("endpoint"):
            from graph_planner.integrations.codefuse_cgm.client import CodeFuseCGMClient

            endpoint = extra.get("endpoint")
            if not endpoint:
                raise ValueError(
                    f"system.topology.groups.{group}.vllm.endpoint must be provided for CodeFuse runtime"
                )
            client_kwargs = {"endpoint": endpoint}
            for key in ("api_key", "model", "temperature", "max_tokens", "timeout_s"):
                if extra.get(key) is not None:
                    client_kwargs[key] = extra[key]
            runtime = _CodeFuseRuntimeWrapper(CodeFuseCGMClient(**client_kwargs))
            runtime_kind = "codefuse-cgm"
        elif model_path:
            from verl.third_party.vllm import LLM

            llm_kwargs: dict[str, Any] = {
                "model": model_path,
                "tensor_parallel_size": tp,
                "trust_remote_code": True,
            }
            if gpu_memory_utilization is not None:
                llm_kwargs["gpu_memory_utilization"] = gpu_memory_utilization
            if max_model_len is not None:
                llm_kwargs["max_model_len"] = max_model_len
            if kv_cache_dtype:
                llm_kwargs["kv_cache_dtype"] = kv_cache_dtype
            if enforce_eager is not None:
                llm_kwargs["enforce_eager"] = bool(enforce_eager)
            for key in ("dtype", "device", "max_num_batched_tokens", "max_num_seqs"):
                if extra.get(key) is not None:
                    llm_kwargs[key] = extra[key]
            runtime = _VLLMRuntimeWrapper(LLM(**llm_kwargs), sampling_defaults=sampling_defaults)
            runtime_kind = "vllm"
    except Exception as exc:  # pragma: no cover - defensive
        LOG.warning("Failed to create %s runtime for group=%s: %s", runtime_kind, group, exc)
        runtime = None
        runtime_kind = "static"

    engine.attach_runtime(runtime, kind=runtime_kind, sampling_defaults=sampling_defaults)
    if max_num_seqs is not None:
        try:
            engine.set_max_num_seqs(int(max_num_seqs))
        except Exception:
            pass
    return engine


def _oc_get(mapping, key, default=None) -> Any:
    # OmegaConf or dict safe get
    if hasattr(mapping, "get"):
        try:
            return mapping.get(key, default)
        except Exception:
            return default
    return getattr(mapping, key, default)


def _oc_has(mapping, key) -> bool:
    try:
        if hasattr(mapping, "__contains__"):
            return key in mapping
    except Exception:
        pass
    try:
        _ = _oc_get(mapping, key, None)
        return _ is not None
    except Exception:
        return False


def _oc_set(mapping, key, value):
    # OmegaConf or dict safe set with dotted-key support
    if mapping is None:
        return
    try:
        OmegaConf.update(mapping, key, value, merge=True)  # type: ignore[arg-type]
        return
    except Exception:
        pass

    if isinstance(key, str) and "." in key:
        parts = [p for p in key.split(".") if p]
        target = mapping
        for part in parts[:-1]:
            nxt = _oc_get(target, part, None)
            if nxt is None:
                nxt = {}
                try:
                    if hasattr(target, "__setitem__"):
                        target[part] = nxt  # type: ignore[index]
                    else:
                        setattr(target, part, nxt)
                except Exception:
                    return
            target = nxt
        key = parts[-1]

    try:
        if hasattr(mapping, "__setitem__"):
            mapping[key] = value  # type: ignore[index]
            return
    except Exception:
        pass
    try:
        setattr(mapping, key, value)
    except Exception:
        pass


def _oc_del(mapping, key):
    try:
        if hasattr(mapping, "__delitem__"):
            del mapping[key]
            return
    except Exception:
        pass
    try:
        delattr(mapping, key)
    except Exception:
        pass


def _normalize_actor_batch_keys(actor_cfg) -> None:
    """Emit compatibility warnings for legacy vs. modern micro-batch keys."""

    if actor_cfg is None:
        return

    modern_per_gpu = _oc_get(actor_cfg, "ppo_micro_batch_size_per_gpu", None)
    legacy_per_gpu_keys = [
        ("micro_batch_size_per_gpu", _oc_get(actor_cfg, "micro_batch_size_per_gpu", None)),
        ("micro_batch_size", _oc_get(actor_cfg, "micro_batch_size", None)),
        ("mbs", _oc_get(actor_cfg, "mbs", None)),
    ]

    if modern_per_gpu is not None:
        for key, value in legacy_per_gpu_keys:
            if value is not None:
                LOG.warning(
                    "actor.%s is ignored because actor.ppo_micro_batch_size_per_gpu is set; keeping the modern key",
                    key,
                )
                break

    modern_global = _oc_get(actor_cfg, "ppo_micro_batch_size", None)
    legacy_global = _oc_get(actor_cfg, "micro_batch_size", None)
    if modern_global is not None and legacy_global is not None:
        LOG.warning(
            "actor.micro_batch_size is ignored because actor.ppo_micro_batch_size is set; keeping the modern key"
        )


def _compute_dp_world(cfg, *, suppress_log: bool = False) -> int:
    sp = max(1, _get_sp_from_cfg(cfg))
    actor_rollout_ref = _oc_get(cfg, "actor_rollout_ref", None)
    actor_cfg = _oc_get(actor_rollout_ref, "actor", None)
    fsdp_cfg = _oc_get(actor_cfg, "fsdp_config", None)
    fsdp_size_raw = _oc_get(fsdp_cfg, "fsdp_size", 0)

    dp_world = 0
    try:
        fsdp_size = int(fsdp_size_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"actor.fsdp_config.fsdp_size must be integer-compatible, got {fsdp_size_raw!r}"
        ) from exc

    if fsdp_size > 0:
        dp_world = fsdp_size // sp
        if dp_world <= 0:
            if not suppress_log:
                LOG.warning(
                    "[trainer] fsdp_size=%s with sp=%s produced dp_world=%s; forcing dp_world=1",
                    fsdp_size,
                    sp,
                    dp_world,
                )
            dp_world = 1
    else:
        trainer_cfg = _oc_get(cfg, "trainer", None)
        n_gpus_raw = _oc_get(trainer_cfg, "n_gpus_per_node", 1)
        nnodes_raw = _oc_get(trainer_cfg, "nnodes", 1)
        try:
            n_gpus = int(n_gpus_raw)
            nnodes = int(nnodes_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"trainer.n_gpus_per_node ({n_gpus_raw!r}) and trainer.nnodes ({nnodes_raw!r}) must be integer-compatible"
            ) from exc
        dp_world = (n_gpus * nnodes) // sp
        if dp_world <= 0:
            if not suppress_log:
                LOG.warning(
                    "[trainer] Derived non-positive dp_world=%s from n_gpus_per_node=%s, nnodes=%s, sp=%s; forcing 1",
                    dp_world,
                    n_gpus,
                    nnodes,
                    sp,
                )
            dp_world = 1
        else:
            if not suppress_log:
                LOG.info(
                    "[trainer] Using trainer.n_gpus_per_node=%s and trainer.nnodes=%s to derive dp_world=%s with sp=%s",
                    n_gpus,
                    nnodes,
                    dp_world,
                    sp,
                )

    return max(1, dp_world)


def _resolve_micro_batch_sizes(cfg) -> Tuple[int | None, int | None]:
    """Return (global_micro, per_gpu_micro) using DP/SP-aware derivation."""

    actor_rollout_ref = _oc_get(cfg, "actor_rollout_ref", None)
    actor_cfg = _oc_get(actor_rollout_ref, "actor", None)
    if actor_cfg is None:
        return None, None

    per_gpu_candidates = [
        ("actor.ppo_micro_batch_size_per_gpu", _oc_get(actor_cfg, "ppo_micro_batch_size_per_gpu", None)),
        ("actor.micro_batch_size_per_gpu", _oc_get(actor_cfg, "micro_batch_size_per_gpu", None)),
        ("actor.micro_batch_size", _oc_get(actor_cfg, "micro_batch_size", None)),
        ("actor.mbs", _oc_get(actor_cfg, "mbs", None)),
    ]

    per_gpu = None
    per_gpu_source = None
    for key, value in per_gpu_candidates:
        if value is None:
            continue
        try:
            per_gpu = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be integer-compatible, got {value!r}") from exc
        per_gpu_source = key
        break

    if per_gpu is None:
        return None, None
    if per_gpu <= 0:
        raise ValueError(f"{per_gpu_source} resolved to non-positive value {per_gpu}")

    global_candidates = [
        ("actor.ppo_micro_batch_size", _oc_get(actor_cfg, "ppo_micro_batch_size", None)),
        ("ppo.micro_batch_size", _oc_get(_oc_get(cfg, "ppo", None), "micro_batch_size", None)),
    ]

    global_mb = None
    global_source = None
    for key, value in global_candidates:
        if value is None:
            continue
        try:
            global_mb = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be integer-compatible, got {value!r}") from exc
        global_source = key
        break

    dp_world = _compute_dp_world(cfg)

    if global_mb is None:
        derived = per_gpu * dp_world
        LOG.info(
            "[trainer] Deriving actor.ppo_micro_batch_size=%s from %s=%s with dp_world=%s",
            derived,
            per_gpu_source,
            per_gpu,
            dp_world,
        )
        global_mb = derived
    elif global_mb <= 0:
        raise ValueError(
            f"{global_source} resolved to non-positive value {global_mb}"
        )

    return global_mb, per_gpu


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}"
                    + "cannot be satisfied in this ray cluster"
                )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.reweight_method,
                config.pf_ppo.weight_pow,
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:


    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, and vLLM integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # Run identifiers used for naming detached Ray actors.
        self._run_id = getattr(self.config.trainer, "run_id", uuid.uuid4().hex[:8])

        # Topology-aware worker state (populated lazily in init_workers).
        self.topology_groups: dict[str, TopologyGroupConfig] = self._parse_topology_groups(self.config)
        self.worker_groups: dict[str, RayWorkerGroup | None] = {
            "planner": None,
            "rollout": None,
            "cgm": None,
        }
        self.planner_trainer: RayWorkerGroup | None = None
        self.planner_engine: StaticVLLMEngine | None = None
        self.planner_sync: PlannerToVLLMSyncer | None = None
        self.cgm_engine: StaticVLLMEngine | None = None
        self.cgm_actor = None
        self.async_pipeline_mode: bool = False
        self.pipeline_max_policy_version_lag: int = 0
        self.pipeline_rollout_prefetch: int = 1
        self.cur_snapshot_id: int = 0
        self._last_val_metrics: dict | None = None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.OPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.GPG,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _parse_topology_groups(self, config) -> dict[str, TopologyGroupConfig]:
        """Parse the YAML-driven topology section into structured configs."""

        result: dict[str, TopologyGroupConfig] = {}
        topology_cfg = _oc_get(_oc_get(config, "system", {}), "topology", {})
        group_cfg = _oc_get(topology_cfg, "groups", {}) or {}

        def _normalize_gpu_list(raw) -> list[int]:
            if raw is None:
                return []
            if isinstance(raw, str):
                raw = raw.strip()
                if not raw:
                    return []
                return [int(x) for x in raw.split(",")]
            if isinstance(raw, Mapping):
                # Support {start: 0, stop: 4}
                if {"start", "stop"}.issubset(raw.keys()):
                    start = int(raw["start"])
                    stop = int(raw["stop"])
                    return list(range(start, stop))
            try:
                return [int(x) for x in raw]
            except Exception:
                return []

        def _as_int(value):
            if value in (None, ""):
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        def _as_bool(value):
            if value is None:
                return None
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"true", "1", "yes", "on"}:
                    return True
                if lowered in {"false", "0", "no", "off"}:
                    return False
            return None

        for name in ("planner", "cgm"):
            cfg = _oc_get(group_cfg, name, {})
            if cfg is None:
                continue
            gpus = _normalize_gpu_list(_oc_get(cfg, "gpus", []))
            fsdp_size = _oc_get(cfg, "fsdp_size", None)
            vllm_cfg = _oc_get(cfg, "vllm", None)
            vllm = None
            if vllm_cfg is not None:
                if isinstance(vllm_cfg, DictConfig):
                    container = OmegaConf.to_container(vllm_cfg, resolve=True)
                elif isinstance(vllm_cfg, MappingABC):
                    container = dict(vllm_cfg)
                else:
                    container = None

                if isinstance(container, MappingABC):
                    cfg_dict = dict(container)

                    def _pop(key: str, default=None):
                        return cfg_dict.pop(key, default)

                    vllm = VLLMGroupConfig(
                        tp=int(_pop("tp", 1) or 1),
                        dp=int(_pop("dp", 1) or 1),
                        gpu_memory_utilization=_pop("gpu_memory_utilization", None),
                        max_model_len=_pop("max_model_len", None),
                        max_num_seqs=_as_int(_pop("max_num_seqs", None)),
                        kv_cache_dtype=_pop("kv_cache_dtype", None),
                        enforce_eager=_as_bool(_pop("enforce_eager", None)),
                        model_path=_pop("model_path", None),
                        adapters=_pop("adapters", None),
                        extra={str(k): v for k, v in cfg_dict.items()},
                    )
            result[name] = TopologyGroupConfig(name=name, gpus=gpus, fsdp_size=fsdp_size, vllm=vllm)
        return result

    def _validate_config(self):
        """
        Make RayPPOTrainer robust to config aliasing and missing keys.
        - Prefer modern keys (e.g., *_per_gpu) while tolerating legacy ones.
        - Inject safe defaults for commonly-missing fields.
        - Warn instead of raising whenever we can recover automatically.
        """
        import torch
        from omegaconf import OmegaConf

        cfg = self.config  # OmegaConf DictConfig expected

        # ---------- small helpers ----------
        def _has(path: str) -> bool:
            return OmegaConf.select(cfg, path) is not None

        def _get(path: str, default=None):
            val = OmegaConf.select(cfg, path)
            return default if val is None else val

        def _ensure(path: str, value):
            """setdefault for nested OmegaConf paths"""
            parts = path.split(".")
            node = cfg
            for p in parts[:-1]:
                if node.get(p, None) is None:
                    node[p] = OmegaConf.create({})
                node = node[p]
            if node.get(parts[-1], None) is None:
                node[parts[-1]] = value

        def _delete(path: str):
            try:
                parts = path.split(".")
                node = cfg
                for p in parts[:-1]:
                    node = node[p]
                if parts[-1] in node:
                    del node[parts[-1]]
            except Exception:
                pass

        # ---------- 1) trainer defaults ----------
        _ensure("trainer.device", "cuda" if torch.cuda.is_available() else "cpu")
        _ensure("trainer.default_local_dir", f"checkpoints/{_get('trainer.project_name','proj')}/{_get('trainer.experiment_name','exp')}")
        _ensure("trainer.resume_mode", "auto")
        _ensure("trainer.log_val_generations", 0)
        _ensure("trainer.total_training_steps", None)
        _ensure("trainer.profile_steps", None)
        _ensure("trainer.balance_batch", True)
        _ensure("trainer.del_local_ckpt_after_load", False)
        _ensure("trainer.esi_redundant_time", 0)

        # ---------- 2) data defaults ----------
        _ensure("data.dataloader_num_workers", 8)
        _ensure("data.sampler", None)                    # critical: ensure key exists
        _ensure("data.train_batch_size", 8)              # safe fallback
        _ensure("data.val_batch_size", 512)
        _ensure("data.max_prompt_length", 4096)
        _ensure("data.max_response_length", 32768)
        _ensure("data.filter_overlong_prompts", False)   # YAML may override; this is a fallback
        _ensure("data.filter_overlong_prompts_workers", 8)

        # optional but helpful fallbacks
        _ensure("data.dataset_size", None)
        _ensure("data.shuffle", True)
        _ensure("data.drop_last", False)

        # ---------- 3) algorithm defaults ----------
        _ensure("algorithm.gamma", 1.0)
        _ensure("algorithm.lam", 1.0)
        _ensure("algorithm.kl_penalty", "kl")  # common default

        # ---------- 4) rollout/log-prob alias & defaults ----------
        # Prefer *_per_gpu; map legacy -> modern; remove conflict
        legacy_path = "actor_rollout_ref.rollout.log_prob_micro_batch_size"
        modern_path = "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu"
        legacy_v = _get(legacy_path, None)
        modern_v = _get(modern_path, None)

        if legacy_v is not None and modern_v is not None:
            _delete(legacy_path)
        elif legacy_v is not None and modern_v is None:
            # migrate legacy -> modern
            cfg.actor_rollout_ref.rollout["log_prob_micro_batch_size_per_gpu"] = legacy_v
            _delete(legacy_path)
        elif legacy_v is None and modern_v is None:
            # conservative default
            _ensure(modern_path, 1)

        _ensure("actor_rollout_ref.rollout.log_prob_use_dynamic_bsz", True)
        _ensure("actor_rollout_ref.rollout.log_prob_max_batch_size", 1)

        # Optional: align max token len for logprob with actor ppo_max_token_len_per_gpu if present
        if not _has("actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu"):
            ref_len = _get("actor_rollout_ref.actor.ppo_max_token_len_per_gpu", None)
            if ref_len is not None:
                _ensure("actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu", ref_len)

        # Make sure nested blocks exist with safe defaults
        _ensure("actor_rollout_ref.rollout.agent.num_workers", 8)
        _ensure("actor_rollout_ref.rollout.multi_turn.enable", False)

        actor_cfg = _get("actor_rollout_ref.actor", None)
        if actor_cfg is not None:
            _normalize_actor_batch_keys(actor_cfg)

            global_micro, per_gpu_micro = _resolve_micro_batch_sizes(cfg)
            if per_gpu_micro is not None:
                dp_world = _compute_dp_world(cfg, suppress_log=True)
                existing_global = _oc_get(actor_cfg, "ppo_micro_batch_size", None)
                if existing_global is None and global_micro is not None:
                    try:
                        with open_dict(actor_cfg):
                            actor_cfg["ppo_micro_batch_size"] = global_micro
                    except Exception:
                        LOG.debug("[trainer] actor.ppo_micro_batch_size not updated due to struct constraints")

                mini_raw = _get("actor_rollout_ref.actor.ppo_mini_batch_size", None)
                if mini_raw is not None:
                    try:
                        mini_batch = int(mini_raw)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"actor.ppo_mini_batch_size must be integer-compatible, got {mini_raw!r}"
                        ) from exc

                    rollout_n_raw = _get("actor_rollout_ref.rollout.n", 1) or 1
                    try:
                        rollout_n = int(rollout_n_raw)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"actor_rollout_ref.rollout.n must be integer-compatible, got {rollout_n_raw!r}"
                        ) from exc

                    numerator = mini_batch * rollout_n
                    quotient_est = numerator / (dp_world * per_gpu_micro)

                    if numerator <= 0:
                        raise ValueError(
                            "actor.ppo_mini_batch_size must be positive after normalization: "
                            f"mini={mini_batch}, per_gpu={per_gpu_micro}, dp_world={dp_world}, "
                            f"rollout_n={rollout_n}, quotient={quotient_est}"
                        )

                    if numerator % dp_world != 0:
                        raise ValueError(
                            "actor.ppo_mini_batch_size * rollout.n must be divisible by DP world size: "
                            f"mini={mini_batch}, per_gpu={per_gpu_micro}, dp_world={dp_world}, "
                            f"rollout_n={rollout_n}, quotient={quotient_est}"
                        )

                    normalized = numerator // dp_world
                    if normalized % per_gpu_micro != 0:
                        raise ValueError(
                            "Normalized actor mini batch must be divisible by per-GPU micro batch: "
                            f"mini={mini_batch}, per_gpu={per_gpu_micro}, dp_world={dp_world}, "
                            f"rollout_n={rollout_n}, quotient={quotient_est}"
                        )

                    final_quot = normalized // per_gpu_micro
                    if final_quot <= 0:
                        raise ValueError(
                            "Normalized actor mini batch quotient must be positive: "
                            f"mini={mini_batch}, per_gpu={per_gpu_micro}, dp_world={dp_world}, "
                            f"rollout_n={rollout_n}, quotient={final_quot}"
                        )

        # val kwargs: default do_sample -> False; if temperature > 0, you may flip it later in your pipeline
        if not _has("actor_rollout_ref.rollout.val_kwargs.do_sample"):
            temp = float(_get("actor_rollout_ref.rollout.val_kwargs.temperature", 0.0) or 0.0)
            _ensure("actor_rollout_ref.rollout.val_kwargs.do_sample", bool(temp > 0.0))

        # ---------- 5) reward model defaults (even if disabled elsewhere) ----------
        _ensure("reward_model.enable", False)
        _ensure("reward_model.launch_reward_fn_async", False)

        # ---------- 6) sanity checks we cannot recover from ----------
        assert _has("actor_rollout_ref.model.path"), "actor_rollout_ref.model.path is required"
        assert _has("actor_rollout_ref.model.tokenizer_path"), "actor_rollout_ref.model.tokenizer_path is required"
        assert _has("trainer.device"), "trainer.device must exist (should have been injected)"

        # ---------- 7) final debug print ----------
        mb = _get("actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu")
        print(f"[CFG] rollout.log_prob_micro_batch_size_per_gpu = {mb}")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        base_collate = collate_fn or _default_collate

        def _with_extra_info(_examples):
            metas = []
            sanitized = []
            for ex in _examples:
                record: Any = ex
                if isinstance(ex, Mapping):
                    record = dict(ex)
                    metas.append(record.pop("extra_info", None))
                else:
                    meta = None
                    if hasattr(ex, "get"):
                        try:
                            meta = ex.get("extra_info")  # type: ignore[call-arg]
                        except Exception:
                            meta = None
                    if meta is None and hasattr(ex, "extra_info"):
                        meta = getattr(ex, "extra_info")
                    metas.append(meta)
                    if hasattr(ex, "copy"):
                        try:
                            record = ex.copy()
                            if hasattr(record, "pop"):
                                record.pop("extra_info", None)
                        except Exception:
                            record = ex
                sanitized.append(record)

            batch = base_collate(sanitized)

            if metas:
                existing_meta = batch.get("meta") if isinstance(batch, Mapping) else None
                if existing_meta is None:
                    existing_meta = np.array([{} for _ in range(len(metas))], dtype=object)
                elif isinstance(existing_meta, list):
                    existing_meta = np.array(existing_meta, dtype=object)
                elif isinstance(existing_meta, np.ndarray):
                    if existing_meta.dtype != object:
                        existing_meta = existing_meta.astype(object)
                else:
                    existing_meta = np.array([existing_meta] * len(metas), dtype=object)

                merged_meta: list[dict] = []
                for idx, extra in enumerate(metas):
                    base_meta = {}
                    if idx < len(existing_meta):
                        base_meta = existing_meta[idx]
                    if hasattr(base_meta, "item"):
                        base_meta = base_meta.item()
                    if isinstance(base_meta, Mapping):
                        base_meta = dict(base_meta)
                    elif isinstance(base_meta, dict):
                        base_meta = dict(base_meta)
                    elif base_meta is None:
                        base_meta = {}
                    else:
                        base_meta = {"value": base_meta}
                    if extra is not None:
                        base_meta["extra_info"] = extra
                    merged_meta.append(base_meta)

                batch = dict(batch)
                batch["meta"] = np.array(merged_meta, dtype=object)
            return batch

        collate_fn = _with_extra_info

        num_workers = self.config.data["dataloader_num_workers"]

        drop_last = bool(self.config.data.get("drop_last_train", False))

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=drop_last,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )
        # Prevent empty train dataloader edge case
        try:
            if hasattr(self.train_dataset, "__len__"):
                assert len(self.train_dataset) >= int(self.config.data.train_batch_size), (
                    "len(train_dataset) < data.train_batch_size — reduce batch or expand dataset"
                )
        except Exception:
            pass

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        try:
            _probe = next(iter(self.train_dataloader))
            if isinstance(_probe, Mapping):
                keys = list(_probe.keys())
            else:
                keys = list(getattr(_probe, "keys", lambda: [])())
            print("[DEBUG] train batch keys:", keys)
            if isinstance(_probe, Mapping) and "meta" in _probe:
                sample_meta = _probe["meta"][0] if _probe.get("meta") else None
                if isinstance(sample_meta, str):
                    try:
                        sample_meta = _json.loads(sample_meta)
                    except Exception:
                        pass
                print("[DEBUG] meta sample:", sample_meta)
        except Exception as _e:
            print("[DEBUG] skip batch probe:", repr(_e))

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        sample_turns = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "interaction_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("interaction_kwargs")
            if "agent_name" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("agent_name")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = self.actor_rollout_wg.world_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
                    print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        if self.topology_groups.get("planner"):
            self._init_worker_groups_from_topology()

            # create async rollout manager and request scheduler once worker groups are available
            self.async_rollout_manager = None
            if (
                self.config.actor_rollout_ref.rollout.mode == "async"
                and getattr(self, "actor_rollout_wg", None) is not None
            ):
                from verl.experimental.agent_loop import AgentLoopManager

                self.async_rollout_manager = AgentLoopManager(
                    config=self.config,
                    worker_group=self.actor_rollout_wg,
                )
            return

        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        profile_option = OmegaConf.select(self.config, "trainer.npu_profile.options", default=None)

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
                profile_option=profile_option,
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
                profile_option=profile_option,
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.trainer, "profile_steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.trainer, "profile_steps")
            assert OmegaConf.select(self.config.trainer, "worker_nsight_options") is not None, (
                "worker_nsight_options must be set when profile_steps is set"
            )
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                OmegaConf.select(self.config.trainer, "worker_nsight_options")
            )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        topo_groups = OmegaConf.select(self.config, "system.topology.groups", default=None)
        has_dedicated_rollout = bool(topo_groups is not None and "rollout" in topo_groups)

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        if not has_dedicated_rollout and "actor_rollout" in all_wg:
            self.actor_rollout_wg = all_wg["actor_rollout"]
            self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler once worker groups are available
        self.async_rollout_manager = None
        if (
            self.config.actor_rollout_ref.rollout.mode == "async"
            and getattr(self, "actor_rollout_wg", None) is not None
        ):
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )

    def _init_worker_groups_from_topology(self) -> None:
        """Create worker groups and engines as described by ``system.topology``."""

        topology_cfg = OmegaConf.select(self.config, "system.topology.groups", default={}) or {}
        if isinstance(topology_cfg, DictConfig):
            topology_cfg = OmegaConf.to_container(topology_cfg, resolve=True)
        if not isinstance(topology_cfg, MappingABC):
            topology_cfg = {}

        planner_group = self.topology_groups.get("planner")
        if planner_group is None or not planner_group.gpus:
            raise ValueError("Planner topology must specify at least one GPU")

        planner_cfg = topology_cfg.get("planner") or {}
        if isinstance(planner_cfg, DictConfig):
            planner_cfg = OmegaConf.to_container(planner_cfg, resolve=True)

        rollout_cfg = topology_cfg.get("rollout")
        if isinstance(rollout_cfg, DictConfig):
            rollout_cfg = OmegaConf.to_container(rollout_cfg, resolve=True)

        cgm_cfg_raw = topology_cfg.get("cgm") or {}
        if isinstance(cgm_cfg_raw, DictConfig):
            cgm_cfg_raw = OmegaConf.to_container(cgm_cfg_raw, resolve=True)

        env_vars: dict[str, str] = {}
        global_env = OmegaConf.select(self.config, "system.runtime_env.env_vars", default=None)
        if isinstance(global_env, DictConfig):
            global_env = OmegaConf.to_container(global_env, resolve=True)
        if isinstance(global_env, MappingABC):
            env_vars.update({str(k): str(v) for k, v in global_env.items() if v is not None})

        group_env = None
        if isinstance(planner_cfg, MappingABC):
            group_env = planner_cfg.get("env_vars")
        elif isinstance(planner_cfg, DictConfig):
            group_env = planner_cfg.get("env_vars")
        if isinstance(group_env, DictConfig):
            group_env = OmegaConf.to_container(group_env, resolve=True)
        if isinstance(group_env, MappingABC):
            env_vars.update({str(k): str(v) for k, v in group_env.items() if v is not None})

        if self._run_id:
            env_vars["GRAPH_PLANNER_RUN_ID"] = str(self._run_id)
        fsdp_world = int(planner_group.fsdp_size or len(planner_group.gpus))
        planner_gpus = list(planner_group.gpus or [])
        planner_worker_env_overrides = None
        if planner_gpus:
            if fsdp_world > len(planner_gpus):
                raise ValueError(
                    "Planner FSDP world size exceeds configured GPU list; cannot assign unique GPUs"
                )
            planner_worker_env_overrides = []
            for rank, gpu_id in enumerate(planner_gpus[:fsdp_world]):
                actor_env = {
                    "CUDA_VISIBLE_DEVICES": str(gpu_id),    
                    "LOCAL_RANK": "0",
                    "RANK": str(rank),
                    "WORLD_SIZE": str(fsdp_world),
                }
                planner_worker_env_overrides.append(actor_env)




        _oc_set(self.config, "actor_rollout_ref.actor.fsdp_config.fsdp_size", fsdp_world)
        print(f"[planner] fsdp_size set to {fsdp_world} prior to worker init")
        print(f"FSDP planner world_size={fsdp_world} on GPUs {planner_group.gpus}")

        # extra guard: planner 的 vllm tp 不要比当前分配的 GPU 数大
        if getattr(planner_group, "vllm", None) is not None:
            current_tp = getattr(planner_group.vllm, "tp", None)
            if current_tp is not None and current_tp > len(planner_group.gpus):
                planner_group.vllm.tp = 1

        rollout_tp = _oc_get(_oc_get(self.config, "actor_rollout_ref", {}), "rollout", {})
        tp_value = _oc_get(rollout_tp, "tensor_model_parallel_size", None) if isinstance(rollout_tp, MappingABC) else None
        if tp_value in (None, 0):
            planner_tp = None
            if getattr(planner_group, "vllm", None) is not None:
                planner_tp = getattr(planner_group.vllm, "tp", None)
            if planner_tp in (None, 0):
                planner_tp = len(planner_group.gpus)
            _oc_set(
                self.config,
                "actor_rollout_ref.rollout.tensor_model_parallel_size",
                int(planner_tp),
            )

        planner_pool = RayResourcePool(
            process_on_nodes=[fsdp_world],
            use_gpu=True,
            name_prefix=f"planner-{self._run_id}",
            max_colocate_count=1,
        )

        profile_option = OmegaConf.select(self.config, "trainer.npu_profile.options", default=None)

        use_dedicated_rollout = isinstance(rollout_cfg, MappingABC) and bool(rollout_cfg)
        has_separate_actor_cfg = _oc_get(self.config, "actor_rollout_ref.actor", None) is not None
        
        if use_dedicated_rollout and has_separate_actor_cfg:
            # 即便 role_worker_mapping 里没挂 Role.Actor，也强行用 actor 的这段配置
            actor_cls = self.role_worker_mapping.get(getattr(Role, "Actor", None), None)
            if actor_cls is None:
                # 没有专门的类，就还是用 rollout 的类，但走 actor 的配置
                actor_cls = self.role_worker_mapping[Role.ActorRollout]
            planner_cls = RayClassWithInitArgs(
                cls=actor_cls,
                config=self.config.actor_rollout_ref.actor,
                role="actor",
                profile_option=profile_option,
            )
        else:
            planner_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
                profile_option=profile_option,
            )

        if env_vars:
            planner_cls.update_options({"runtime_env": {"env_vars": env_vars}})
        planner_cls.update_options({"num_gpus": 0})
            
        planner_wg_root = self.ray_worker_group_cls(
            resource_pool=planner_pool,
            ray_cls_with_init=planner_cls,
            device_name=self.device_name,
            worker_env_overrides=planner_worker_env_overrides,
        )
        wanted_prefixes = {
            "actor_rollout",
            "actor_rollout_ref",
            "actor",
            "rollout",
            "ref",
        }
        spawned = _safe_spawn_map(
            planner_wg_root,
            wanted_prefixes=wanted_prefixes,
            world_size=fsdp_world,
        )

        planner_wg = None
        for key in ("actor_rollout", "actor_rollout_ref", "actor", "rollout"):
            candidates = spawned.get(key) or []
            if candidates:
                planner_wg = candidates[0]
                break

        if planner_wg is None:
            def _dump_tree(root, depth=0, lines=None, visited=None, max_nodes=512):
                lines = [] if lines is None else lines
                visited = set() if visited is None else visited
                if root is None:
                    lines.append(f"{'  ' * depth}- <None>")
                    return lines

                if len(lines) >= max_nodes:
                    return lines

                obj_id = id(root)
                if obj_id in visited:
                    lines.append(f"{'  ' * depth}- <cycle>")
                    return lines
                visited.add(obj_id)

                indent = '  ' * depth
                if _is_container(root):
                    size = _container_len(root)
                    lines.append(f"{indent}- <{root.__class__.__name__}> size={size}")
                    child_iter = _yield_container_items(root)
                else:
                    display_name = None
                    for attr in ("name", "prefix", "_name", "sub_cls_name"):
                        value = getattr(root, attr, None)
                        if isinstance(value, str):
                            display_name = value
                            break
                    if display_name is None:
                        display_name = root.__class__.__name__
                    lines.append(f"{indent}- {display_name} <{root.__class__.__name__}>")
                    child_iter = _iter_group_children(root)

                for idx, child in enumerate(child_iter):
                    if len(lines) >= max_nodes:
                        break
                    if idx >= 64:
                        lines.append(f"{'  ' * (depth + 1)}... (truncated)")
                        break
                    _dump_tree(child, depth + 1, lines, visited, max_nodes)
                return lines

            counts = {key: len(value) for key, value in spawned.items()}
            discovered_attrs = sorted(_discover_container_attrs(planner_wg_root.__class__))
            tree_dump = "\n".join(_dump_tree(planner_wg_root))
            raise RuntimeError(
                "Planner worker group not found. "
                f"Expected one of {sorted(wanted_prefixes)}, got counts={counts}.\n"
                f"Discovered container attrs on root: {discovered_attrs}\n"
                f"Discovered tree:\n{tree_dump}"
            )

        planner_wg.init_model()

        # Optional: keep n_gpus metrics correct when bypassing resource_pool_manager defaults
        try:
            self.resource_pool_manager.override_n_gpus(len(planner_group.gpus))
        except Exception:
            pass

        # Ensure FSDP world size shows up in config for init_model() and logs
        _oc_set(self.config, "actor_rollout_ref.actor.fsdp_config.fsdp_size", fsdp_world)

        self.worker_groups["planner"] = planner_wg
        self.planner_trainer = planner_wg
        
        # 只有没有拓扑版 rollout 时，才把 planner 当成 rollout 用
        if not (isinstance(rollout_cfg, MappingABC) and rollout_cfg):
            self.actor_rollout_wg = planner_wg

        rollout_wg: RayWorkerGroup | None = None
        if isinstance(rollout_cfg, MappingABC) and rollout_cfg:
            raw_num_gpus = rollout_cfg.get("num_gpus", 0)
            try:
                rollout_need_gpu = int(raw_num_gpus) > 0
            except (TypeError, ValueError):
                rollout_need_gpu = bool(raw_num_gpus)

            rollout_pool = RayResourcePool(
                process_on_nodes=[1],
                use_gpu=rollout_need_gpu,
                name_prefix=f"rollout-{self._run_id}",
                max_colocate_count=1,
            )

            # === 关键改动开始：构造一份“只有 rollout 的配置” ===
            # 原来是直接用 self.config.actor_rollout_ref
            full_ar_cfg = self.config.actor_rollout_ref
            # 转成普通 dict，方便删字段
            if isinstance(full_ar_cfg, DictConfig):
                full_ar_cfg = OmegaConf.to_container(full_ar_cfg, resolve=True)

            # 拷一份，别动原 config
            rollout_only_cfg = deepcopy(full_ar_cfg)
            # 这几段是训练 actor 用的，我们不想在 rollout worker 上建 FSDP，就删掉
            for key in ("actor", "ref", "model"):
                rollout_only_cfg.pop(key, None)
            # 有的版本里面还有这个字段，就顺手干掉
            rollout_only_cfg.pop("hybrid_engine", None)

            # 我们仍然要告诉 rollout 它的 rollout 段来自原来的配置
            # （大部分字段你 YAML 里都写了）
            # rollout_only_cfg 现在应该至少有：
            #   rollout: { name: vllm, mode: async, ... }

            rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=rollout_only_cfg,
                role="actor_rollout",
                profile_option=profile_option,
            )

                
            rollout_env_vars = rollout_cfg.get("env_vars") if isinstance(rollout_cfg, MappingABC) else None
            if isinstance(rollout_env_vars, DictConfig):
                rollout_env_vars = OmegaConf.to_container(rollout_env_vars, resolve=True)

            runtime_env = {}
            merged_env = dict(env_vars)
            if isinstance(rollout_env_vars, MappingABC):
                merged_env.update({str(k): str(v) for k, v in rollout_env_vars.items() if v is not None})
            if merged_env:
                runtime_env["env_vars"] = merged_env
            if runtime_env:
                rollout_cls.update_options({"runtime_env": runtime_env})

            # 在决定不用 GPU 的情况下，告诉 Ray 明确是 0
            if not rollout_need_gpu:
                rollout_cls.update_options({"num_gpus": 0})
            else:
                rollout_cls.update_options({"num_gpus": 1})
                
            
            rollout_wg = self.ray_worker_group_cls(
                resource_pool=rollout_pool,
                ray_cls_with_init=rollout_cls,
                device_name=self.device_name,
            )
            rollout_wg.init_model()
            self.worker_groups["rollout"] = rollout_wg
            self.actor_rollout_wg = rollout_wg

        if planner_group.vllm is not None:
            self.planner_engine = build_vllm_engine(
                group="planner",
                gpus=planner_group.gpus,
                tp=planner_group.vllm.tp,
                dp=planner_group.vllm.dp,
                max_model_len=planner_group.vllm.max_model_len,
                gpu_memory_utilization=planner_group.vllm.gpu_memory_utilization,
                max_num_seqs=planner_group.vllm.max_num_seqs,
                kv_cache_dtype=planner_group.vllm.kv_cache_dtype,
                enforce_eager=planner_group.vllm.enforce_eager,
                model_path=planner_group.vllm.model_path,
                adapters=planner_group.vllm.adapters,
                extra_config=planner_group.vllm.extra,
            )
            print(
                "vLLM planner TP=%s on GPUs %s"
                % (
                    planner_group.vllm.tp,
                    planner_group.gpus,
                )
            )
        else:
            self.planner_engine = None

        self.planner_sync = PlannerToVLLMSyncer(self.planner_trainer, self.planner_engine)

        cgm_group = self.topology_groups.get("cgm")
        if cgm_group is not None and cgm_group.gpus:
            self.cgm_engine = build_vllm_engine(
                group="cgm",
                gpus=cgm_group.gpus,
                tp=cgm_group.vllm.tp if cgm_group.vllm else 1,
                dp=cgm_group.vllm.dp if cgm_group.vllm else 1,
                max_model_len=cgm_group.vllm.max_model_len if cgm_group.vllm else None,
                gpu_memory_utilization=(
                    cgm_group.vllm.gpu_memory_utilization if cgm_group.vllm else None
                ),
                max_num_seqs=cgm_group.vllm.max_num_seqs if cgm_group.vllm else None,
                kv_cache_dtype=cgm_group.vllm.kv_cache_dtype if cgm_group.vllm else None,
                enforce_eager=cgm_group.vllm.enforce_eager if cgm_group.vllm else None,
                model_path=cgm_group.vllm.model_path if cgm_group.vllm else None,
                adapters=cgm_group.vllm.adapters if cgm_group.vllm else None,
                extra_config=cgm_group.vllm.extra if cgm_group.vllm else None,
            )
            print(
                "vLLM CGM TP=%s on GPUs %s"
                % (
                    cgm_group.vllm.tp if cgm_group.vllm else 1,
                    cgm_group.gpus,
                )
            )
            actor_name = f"CGMService::{self._run_id}"
            if ray.is_initialized():
                cgm_env: dict[str, str] = {}
                global_env = OmegaConf.select(self.config, "system.runtime_env.env_vars", default=None)
                if isinstance(global_env, DictConfig):
                    global_env = OmegaConf.to_container(global_env, resolve=True)
                if isinstance(global_env, MappingABC):
                    cgm_env.update({str(k): str(v) for k, v in global_env.items() if v is not None})

                group_env = None
                if isinstance(cgm_cfg_raw, MappingABC):
                    group_env = cgm_cfg_raw.get("env_vars")
                elif isinstance(cgm_cfg_raw, DictConfig):
                    group_env = cgm_cfg_raw.get("env_vars")
                if isinstance(group_env, DictConfig):
                    group_env = OmegaConf.to_container(group_env, resolve=True)
                if isinstance(group_env, MappingABC):
                    cgm_env.update({str(k): str(v) for k, v in group_env.items() if v is not None})

                preview = list(cgm_env.keys())[:6]
                print(
                    f"[topology] group=cgm cuda_visible={cgm_env['CUDA_VISIBLE_DEVICES']} env_keys={preview}..."
                )
                system_cfg = _oc_get(self.config, "system", {})
                topology_cfg = _oc_get(system_cfg, "topology", {})
                groups_cfg = _oc_get(topology_cfg, "groups", {})
                cgm_group_cfg = _oc_get(groups_cfg, "cgm", {})
                vllm_cfg = _oc_get(cgm_group_cfg, "vllm", {}) or {}
                if isinstance(vllm_cfg, DictConfig):
                    vllm_cfg = OmegaConf.to_container(vllm_cfg, resolve=True)
                self.cgm_actor = CGMService.options(
                    name=actor_name,
                    lifetime="detached",
                    num_gpus=len(cgm_group.gpus),
                    runtime_env={"env_vars": cgm_env},
                ).remote(self.cgm_engine, vllm_cfg=vllm_cfg)
                print(f"[Topology] CGM actor '{actor_name}' spawned")
            else:
                self.cgm_actor = None
        else:
            self.cgm_engine = None
            self.cgm_actor = None
            print("[Topology] CGM vLLM disabled (no GPUs)")
        self.worker_groups["cgm"] = None

        # NOTE: keep self.async_rollout_manager intact so async rollouts stay available.

        self.async_rollout_manager = None
        if (
            _oc_get(self.config, "actor_rollout_ref.rollout.mode", None) == "async"
            and getattr(self, "actor_rollout_wg", None) is not None
        ):
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )


        self.critic_wg = None
        self.ref_policy_wg = None
        self.rm_wg = None

        pipeline_cfg = _oc_get(_oc_get(self.config, "system", {}), "pipeline", {})
        self.async_pipeline_mode = bool(_oc_get(pipeline_cfg, "async_mode", False))
        self.pipeline_max_policy_version_lag = int(_oc_get(pipeline_cfg, "max_policy_version_lag", 0) or 0)
        self.pipeline_rollout_prefetch = int(_oc_get(pipeline_cfg, "rollout_prefetch", 1) or 1)

    def publish_snapshot(self) -> int:
        """Push the current learner weights to rollout engines and bump the snapshot id."""

        if self.planner_sync is not None:
            self.planner_sync.sync()
        self.cur_snapshot_id = getattr(self, "cur_snapshot_id", 0) + 1
        if self.planner_sync is not None:
            self.planner_sync.publish_snapshot(self.cur_snapshot_id, global_step=self.global_steps)
        print(f"Snapshot published: snapshot_id={self.cur_snapshot_id}")
        return self.cur_snapshot_id

    def _collect_rollout_batch(
        self,
        *,
        batch_dict: Mapping,
        snapshot_id: int,
        epoch: int,
        metrics: dict,
        timing_raw: dict,
    ) -> DataProto:
        """Prepare a rollout batch and run planner inference for a training step."""

        metas = batch_dict.pop("meta", None)
        batch: DataProto = DataProto.from_single_dict(batch_dict)

        def _normalize_meta_entries(meta_obj):
            if meta_obj is None:
                return []
            if isinstance(meta_obj, np.ndarray):
                entries = meta_obj.tolist()
            elif isinstance(meta_obj, list | tuple):
                entries = list(meta_obj)
            else:
                entries = [meta_obj]
            normalized: list[dict] = []
            for entry in entries:
                if hasattr(entry, "item"):
                    entry = entry.item()
                if isinstance(entry, Mapping):
                    normalized.append(dict(entry))
                elif isinstance(entry, dict):
                    normalized.append(dict(entry))
                elif entry is None:
                    normalized.append({})
                else:
                    normalized.append({"extra_info": entry})
            return normalized

        meta_entries = _normalize_meta_entries(metas)
        if meta_entries:
            meta_array = np.array(meta_entries, dtype=object)
            batch.non_tensor_batch["meta"] = meta_array
            meta_keys = ("tools_kwargs", "interaction_kwargs", "extra_info", "data_source", "raw_prompt_ids")
            for key in meta_keys:
                values = [meta.get(key, None) for meta in meta_entries]
                if any(val is not None for val in values):
                    batch.non_tensor_batch[key] = np.array(values, dtype=object)

        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        if "multi_modal_data" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        if "raw_prompt" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "tools_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")
        if "interaction_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("interaction_kwargs")
        if "index" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("index")
        if "agent_name" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("agent_name")

        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )

        step_index = self.global_steps + 1
        print(f"Rollout started with snapshot_id={snapshot_id}")
        gen_batch.meta_info["global_steps"] = step_index
        gen_batch.meta_info["policy_version"] = snapshot_id
        gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

        with marked_timer("gen", timing_raw, color="red"):
            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
            timing_raw.update(gen_batch_output.meta_info.get("timing", {}))
            gen_batch_output.meta_info.pop("timing", None)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
            with marked_timer("gen_max", timing_raw, color="purple"):
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["do_sample"] = False
                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                batch = batch.union(gen_baseline_output)
                reward_baseline_tensor = self.reward_fn(batch)
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                batch.batch["reward_baselines"] = reward_baseline_tensor

                del gen_baseline_batch, gen_baseline_output

        batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
        )
        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
        batch = batch.union(gen_batch_output)

        batch.meta_info["epoch"] = epoch
        batch.meta_info["policy_version"] = snapshot_id

        return batch

    def _learn_from_rollout(
        self,
        *,
        batch: DataProto,
        metrics: dict,
        timing_raw: dict,
        logger,
        progress_bar,
    ) -> bool:
        """Consume a rollout batch, run learner updates, and record metrics."""

        step_index = self.global_steps + 1
        epoch = int(batch.meta_info.get("epoch", 0))
        reward_extra_infos_dict: dict[str, list] = {}

        if "response_mask" not in batch.batch.keys():
            batch.batch["response_mask"] = compute_response_mask(batch)

        if self.config.trainer.balance_batch:
            self._balance_batch(batch, metrics=metrics)

        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

        with marked_timer("reward", timing_raw, color="yellow"):
            if self.use_rm:
                reward_tensor = self.rm_wg.compute_rm_score(batch)
                batch = batch.union(reward_tensor)

            if self.config.reward_model.launch_reward_fn_async:
                future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
            else:
                reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

        with marked_timer("old_log_prob", timing_raw, color="blue"):
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            entropys = old_log_prob.batch["entropys"]
            response_masks = batch.batch["response_mask"]
            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
            entropy_agg = agg_loss(
                loss_mat=entropys,
                loss_mask=response_masks,
                loss_agg_mode=loss_agg_mode,
            )
            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
            metrics.update(old_log_prob_metrics)
            old_log_prob.batch.pop("entropys")
            batch = batch.union(old_log_prob)

            if "rollout_log_probs" in batch.batch.keys():
                rollout_old_log_probs = batch.batch["rollout_log_probs"]
                actor_old_log_probs = batch.batch["old_log_probs"]
                attention_mask = batch.batch["attention_mask"]
                responses = batch.batch["responses"]
                response_length = responses.size(1)
                response_mask = attention_mask[:, -response_length:]

                rollout_probs = torch.exp(rollout_old_log_probs)
                actor_probs = torch.exp(actor_old_log_probs)
                rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                rollout_probs_diff_max = torch.max(rollout_probs_diff)
                rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                rollout_probs_diff_std = torch.std(rollout_probs_diff)
                metrics.update(
                    {
                        "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                        "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                        "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                    }
                )

        if self.use_reference_policy:
            with marked_timer("ref", timing_raw, color="olive"):
                if not self.ref_in_actor:
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                else:
                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)

        if self.use_critic:
            with marked_timer("values", timing_raw, color="cyan"):
                values = self.critic_wg.compute_values(batch)
                batch = batch.union(values)

        with marked_timer("adv", timing_raw, color="brown"):
            if self.config.reward_model.launch_reward_fn_async:
                reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
            batch.batch["token_level_scores"] = reward_tensor

            if reward_extra_infos_dict:
                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

            if self.config.algorithm.use_kl_in_reward:
                batch, kl_metrics = apply_kl_penalty(
                    batch,
                    kl_ctrl=self.kl_ctrl_in_reward,
                    kl_penalty=self.config.algorithm.kl_penalty,
                )
                metrics.update(kl_metrics)
            else:
                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

            norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)

            batch = compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=self.config.actor_rollout_ref.rollout.n,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                config=self.config.algorithm,
            )

        if self.use_critic:
            with marked_timer("update_critic", timing_raw, color="pink"):
                critic_output = self.critic_wg.update_critic(batch)
            critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
            metrics.update(critic_output_metrics)

        if self.config.trainer.critic_warmup <= step_index:
            with marked_timer("update_actor", timing_raw, color="red"):
                batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                actor_output = self.actor_rollout_wg.update_actor(batch)
            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
            metrics.update(actor_output_metrics)

        rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
        if rollout_data_dir:
            with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                if "request_id" in batch.non_tensor_batch:
                    reward_extra_infos_dict.setdefault(
                        "request_id",
                        batch.non_tensor_batch["request_id"].tolist(),
                    )
                self._dump_generations(
                    inputs=inputs,
                    outputs=outputs,
                    scores=scores,
                    reward_extra_infos_dict=reward_extra_infos_dict,
                    dump_path=rollout_data_dir,
                )

        is_last_step = step_index >= self.total_training_steps
        if (
            self.val_reward_fn is not None
            and self.config.trainer.test_freq > 0
            and (is_last_step or step_index % self.config.trainer.test_freq == 0)
        ):
            with marked_timer("testing", timing_raw, color="green"):
                val_metrics: dict = self._validate()
                if is_last_step:
                    self._last_val_metrics = val_metrics
            metrics.update(val_metrics)

        esi_close_to_expiration = should_save_ckpt_esi(
            max_steps_duration=self.max_steps_duration,
            redundant_time=self.config.trainer.esi_redundant_time,
        )

        if self.config.trainer.save_freq > 0 and (
            is_last_step
            or step_index % self.config.trainer.save_freq == 0
            or esi_close_to_expiration
        ):
            if esi_close_to_expiration:
                print("Force saving checkpoint: ESI instance expiration approaching.")
            with marked_timer("save_checkpoint", timing_raw, color="green"):
                self._save_checkpoint()

        metrics.update(
            {
                "training/global_step": step_index,
                "training/epoch": epoch,
            }
        )
        metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
        metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
        n_gpus = self.resource_pool_manager.get_n_gpus()
        metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

        if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
            self.train_dataloader.sampler.update(batch=batch)

        logger.log(data=metrics, step=step_index)

        progress_bar.update(1)
        self.global_steps = step_index

        if "step" in timing_raw:
            self.max_steps_duration = max(self.max_steps_duration, timing_raw["step"])

        if is_last_step:
            if self._last_val_metrics is not None:
                pprint(f"Final validation metrics: {self._last_val_metrics}")
            progress_bar.close()

        if hasattr(self.train_dataset, "on_batch_end"):
            self.train_dataset.on_batch_end(batch=batch)

        return is_last_step

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile()
            if self.use_critic:
                self.critic_wg.start_profile()
            if self.use_rm:
                self.rm_wg.start_profile()

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        # Auto-init guard: ensure worker groups exist if entrypoint forgot to call init_workers()
        if not hasattr(self, "actor_rollout_wg"):
            print("[AgentPPOTrainer] init_workers() was not called by the entrypoint; auto-initializing now.")
            self.init_workers()

        profile_steps = OmegaConf.select(self.config, "trainer.profile_steps", None)
        enable_npu_profile = bool(OmegaConf.select(self.config, "trainer.npu_profile.enable", default=False))
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.cur_snapshot_id = 0
        self.max_steps_duration = 0
        self._last_val_metrics = None

        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        total_epochs = max(int(self.config.trainer.total_epochs), 1)
        epoch = 0
        train_iter = iter(self.train_dataloader)

        def _next_batch() -> tuple[int, Mapping] | None:
            nonlocal epoch, train_iter
            if self.global_steps >= self.total_training_steps:
                return None
            while True:
                try:
                    batch_item = next(train_iter)
                    return epoch, batch_item
                except StopIteration:
                    epoch += 1
                    if epoch >= total_epochs:
                        return None
                    train_iter = iter(self.train_dataloader)

        while self.global_steps < self.total_training_steps:
            next_payload = _next_batch()
            if next_payload is None:
                break
            epoch_idx, batch_dict = next_payload
            metrics: dict = {}
            timing_raw: dict = {}
            do_profile = bool(
                enable_npu_profile
                and profile_steps
                and (self.global_steps + 1) in profile_steps
            )

            with marked_timer("start_profile", timing_raw):
                self._start_profiling(do_profile)

            snapshot_id = self.publish_snapshot()

            with marked_timer("step", timing_raw):
                batch = self._collect_rollout_batch(
                    batch_dict=batch_dict,
                    snapshot_id=snapshot_id,
                    epoch=epoch_idx,
                    metrics=metrics,
                    timing_raw=timing_raw,
                )
                finished = self._learn_from_rollout(
                    batch=batch,
                    metrics=metrics,
                    timing_raw=timing_raw,
                    logger=logger,
                    progress_bar=progress_bar,
                )

            with marked_timer("stop_profile", timing_raw):
                self._stop_profiling(do_profile)

            if finished:
                break


def _demo_resolve_mb():
    cfg = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "fsdp_config": {"fsdp_size": 8},
                    "enable_sequence_parallel": False,
                    "ulysses_sequence_parallel_size": 2,
                    "ppo_micro_batch_size_per_gpu": 2,
                }
            },
            "trainer": {"n_gpus_per_node": 4, "nnodes": 1},
        }
    )
    global_mb, per_gpu = _resolve_micro_batch_sizes(cfg)
    assert (global_mb, per_gpu) == (16, 2), (global_mb, per_gpu)
    print("[demo] resolved micro batch sizes:", global_mb, per_gpu)


if __name__ == "__main__":
    _demo_resolve_mb()
