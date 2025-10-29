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

import json
import json as _json
import os
import uuid
from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Any, Optional, Tuple

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
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


@dataclass
class VLLMGroupConfig:
    tp: int = 1
    dp: int = 1
    gpu_memory_utilization: float | None = None
    max_model_len: int | None = None
    model_path: str | None = None
    adapters: Any | None = None


@dataclass
class TopologyGroupConfig:
    name: str
    gpus: list[int]
    fsdp_size: int | None = None
    vllm: VLLMGroupConfig | None = None


class StaticVLLMEngine:
    """Lightweight stand-in for a vLLM engine.

    The actual project wires a true vLLM engine inside Ray workers.  During
    refactors we still keep a Python-side handle so that other components can
    share metadata (TP degree, GPU placement, etc.) without eagerly
    constructing heavyweight resources in the driver process.
    """

    def __init__(
        self,
        *,
        group: str,
        tp: int,
        dp: int,
        gpus: list[int],
        max_model_len: int | None = None,
        gpu_memory_utilization: float | None = None,
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
        self.model_path = model_path
        self.adapters = adapters
        self.extra_config = dict(extra_config or {})
        self._latest_snapshot: int | None = None

    def set_snapshot(self, version: int, **metadata: Any) -> None:
        self._latest_snapshot = version
        if metadata:
            self.extra_config.setdefault("snapshot_metadata", {}).update(metadata)


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


def build_vllm_engine(
    *,
    group: str,
    gpus: list[int],
    tp: int,
    dp: int = 1,
    max_model_len: int | None = None,
    gpu_memory_utilization: float | None = None,
    model_path: str | None = None,
    adapters: Any | None = None,
    extra_config: Mapping | None = None,
) -> StaticVLLMEngine:
    """Construct a topology-aware vLLM engine placeholder."""

    engine = StaticVLLMEngine(
        group=group,
        tp=tp,
        dp=dp,
        gpus=gpus,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        model_path=model_path,
        adapters=adapters,
        extra_config=extra_config,
    )
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
    # OmegaConf or dict safe set
    try:
        mapping[key] = value
    except Exception:
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
    """
    Normalize synonyms and drop legacy keys to avoid mutual-exclusion checks:
      - Prefer 'ppo_micro_batch_size_per_gpu' over 'micro_batch_size_per_gpu'
      - Prefer 'ppo_micro_batch_size' over 'micro_batch_size'
      - If per-gpu is set, drop any global key to avoid conflicts (new VERL style)
    """
    # Normalize per-gpu
    ppo_pg = _oc_get(actor_cfg, "ppo_micro_batch_size_per_gpu", None)
    gen_pg = _oc_get(actor_cfg, "micro_batch_size_per_gpu", None)
    if ppo_pg is None and gen_pg is not None:
        _oc_set(actor_cfg, "ppo_micro_batch_size_per_gpu", gen_pg)

    # Normalize global (legacy)
    ppo_g = _oc_get(actor_cfg, "ppo_micro_batch_size", None)
    gen_g = _oc_get(actor_cfg, "micro_batch_size", None)
    if ppo_g is None and gen_g is not None:
        _oc_set(actor_cfg, "ppo_micro_batch_size", gen_g)

    # If per-gpu exists, drop any global to honor new rule and avoid mutual exclusion
    if _oc_has(actor_cfg, "ppo_micro_batch_size_per_gpu"):
        if _oc_has(actor_cfg, "ppo_micro_batch_size"):
            _oc_del(actor_cfg, "ppo_micro_batch_size")
        if _oc_has(actor_cfg, "micro_batch_size"):
            _oc_del(actor_cfg, "micro_batch_size")

    # Drop generic legacy keys unconditionally after normalization
    if _oc_has(actor_cfg, "micro_batch_size_per_gpu"):
        _oc_del(actor_cfg, "micro_batch_size_per_gpu")
    if _oc_has(actor_cfg, "micro_batch_size"):
        _oc_del(actor_cfg, "micro_batch_size")


def _resolve_micro_batch_sizes(cfg) -> Tuple[int | None, int | None]:
    """
    Return (global_micro, per_gpu_micro) with graceful fallback.
    If global is missing, derive it as per_gpu * dp_world (not multiplied by grad_accum).
    """
    actor = cfg.actor_rollout_ref.actor
    per_gpu = _oc_get(actor, "ppo_micro_batch_size_per_gpu", None)
    global_mb = _oc_get(actor, "ppo_micro_batch_size", None)

    # Derive a global value only if missing and per-gpu present
    if (global_mb is None) and (per_gpu is not None):
        try:
            dp_world = int(cfg.trainer.n_gpus_per_node) * int(cfg.trainer.nnodes)
            if dp_world <= 0:
                dp_world = 1
        except Exception:
            dp_world = 1
        try:
            global_mb = int(per_gpu) * dp_world
        except Exception:
            global_mb = None

    try:
        global_mb = int(global_mb) if global_mb is not None else None
    except Exception:
        global_mb = None
    try:
        per_gpu = int(per_gpu) if per_gpu is not None else None
    except Exception:
        per_gpu = None

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
        self.worker_groups: dict[str, RayWorkerGroup | None] = {"planner": None, "cgm": None}
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

        for name in ("planner", "cgm"):
            cfg = _oc_get(group_cfg, name, {})
            if cfg is None:
                continue
            gpus = _normalize_gpu_list(_oc_get(cfg, "gpus", []))
            fsdp_size = _oc_get(cfg, "fsdp_size", None)
            vllm_cfg = _oc_get(cfg, "vllm", None)
            vllm = None
            if vllm_cfg is not None:
                vllm = VLLMGroupConfig(
                    tp=int(_oc_get(vllm_cfg, "tp", 1) or 1),
                    dp=int(_oc_get(vllm_cfg, "dp", 1) or 1),
                    gpu_memory_utilization=_oc_get(vllm_cfg, "gpu_memory_utilization", None),
                    max_model_len=_oc_get(vllm_cfg, "max_model_len", None),
                    model_path=_oc_get(vllm_cfg, "model_path", None),
                    adapters=_oc_get(vllm_cfg, "adapters", None),
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

            try:
                batch["meta"] = metas
            except Exception:
                try:
                    batch = dict(batch)
                    batch["meta"] = metas
                except Exception:
                    batch = {"data": batch, "meta": metas}
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
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

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
            return

        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
                profile_option=self.config.trainer.npu_profile.options,
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
                profile_option=self.config.trainer.npu_profile.options,
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

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )

    def _init_worker_groups_from_topology(self) -> None:
        """Create worker groups and engines as described by ``system.topology``."""

        planner_cfg = self.topology_groups.get("planner")
        if planner_cfg is None or not planner_cfg.gpus:
            raise ValueError("Planner topology must specify at least one GPU")

        env_vars: dict[str, str] = {}
        if planner_cfg.gpus:
            env_vars["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in planner_cfg.gpus)
        if self._run_id:
            env_vars["GRAPH_PLANNER_RUN_ID"] = str(self._run_id)

        fsdp_world = int(planner_cfg.fsdp_size or len(planner_cfg.gpus))
        print(f"FSDP planner world_size={fsdp_world} on GPUs {planner_cfg.gpus}")

        planner_pool = RayResourcePool(
            process_on_nodes=[len(planner_cfg.gpus)],
            use_gpu=True,
            name_prefix=f"planner-{self._run_id}",
            max_colocate_count=1,
        )

        planner_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.ActorRollout],
            config=self.config.actor_rollout_ref,
            role="actor_rollout",
            profile_option=self.config.trainer.npu_profile.options,
        )
        if env_vars:
            planner_cls.update_options({"runtime_env": {"env_vars": env_vars}})

        planner_wg_root = self.ray_worker_group_cls(
            resource_pool=planner_pool,
            ray_cls_with_init=planner_cls,
            device_name=self.device_name,
        )
        spawned = planner_wg_root.spawn(prefix_set={"actor_rollout"})
        planner_wg = spawned["actor_rollout"]
        planner_wg.init_model()

        self.worker_groups["planner"] = planner_wg
        self.planner_trainer = planner_wg
        self.actor_rollout_wg = planner_wg

        if planner_cfg.vllm is not None:
            self.planner_engine = build_vllm_engine(
                group="planner",
                gpus=planner_cfg.gpus,
                tp=planner_cfg.vllm.tp,
                dp=planner_cfg.vllm.dp,
                max_model_len=planner_cfg.vllm.max_model_len,
                gpu_memory_utilization=planner_cfg.vllm.gpu_memory_utilization,
                model_path=planner_cfg.vllm.model_path,
                adapters=planner_cfg.vllm.adapters,
            )
            print(
                "vLLM planner TP=%s on GPUs %s"
                % (
                    planner_cfg.vllm.tp,
                    planner_cfg.gpus,
                )
            )
        else:
            self.planner_engine = None

        self.planner_sync = PlannerToVLLMSyncer(self.planner_trainer, self.planner_engine)

        cgm_cfg = self.topology_groups.get("cgm")
        if cgm_cfg is not None and cgm_cfg.gpus:
            self.cgm_engine = build_vllm_engine(
                group="cgm",
                gpus=cgm_cfg.gpus,
                tp=cgm_cfg.vllm.tp if cgm_cfg.vllm else 1,
                dp=cgm_cfg.vllm.dp if cgm_cfg.vllm else 1,
                max_model_len=cgm_cfg.vllm.max_model_len if cgm_cfg.vllm else None,
                gpu_memory_utilization=(
                    cgm_cfg.vllm.gpu_memory_utilization if cgm_cfg.vllm else None
                ),
                model_path=cgm_cfg.vllm.model_path if cgm_cfg.vllm else None,
                adapters=cgm_cfg.vllm.adapters if cgm_cfg.vllm else None,
            )
            print(
                "vLLM CGM TP=%s on GPUs %s"
                % (
                    cgm_cfg.vllm.tp if cgm_cfg.vllm else 1,
                    cgm_cfg.gpus,
                )
            )
            actor_name = f"CGMService::{self._run_id}"
            if ray.is_initialized():
                self.cgm_actor = CGMService.options(name=actor_name, lifetime="detached").remote(self.cgm_engine)
                print(f"[Topology] CGM actor '{actor_name}' spawned")
            else:
                self.cgm_actor = None
        else:
            self.cgm_engine = None
            self.cgm_actor = None
            print("[Topology] CGM vLLM disabled (no GPUs)")
        self.worker_groups["cgm"] = None

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

        batch: DataProto = DataProto.from_single_dict(batch_dict)

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
            if not self.async_rollout_mode:
                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
            else:
                gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
            timing_raw.update(gen_batch_output.meta_info.get("timing", {}))
            gen_batch_output.meta_info.pop("timing", None)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
            with marked_timer("gen_max", timing_raw, color="purple"):
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["do_sample"] = False
                if not self.async_rollout_mode:
                    gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                else:
                    gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
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

        if not self.async_pipeline_mode:
            while self.global_steps < self.total_training_steps:
                next_payload = _next_batch()
                if next_payload is None:
                    break
                epoch_idx, batch_dict = next_payload
                metrics: dict = {}
                timing_raw: dict = {}
                do_profile = bool(
                    self.config.trainer.profile_steps
                    and (self.global_steps + 1) in self.config.trainer.profile_steps
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
        else:
            rollout_fut = None
            learn_fut = None
            pending_payload = None
            rollout_exhausted = False

            def rollout_async(snapshot_id: int):
                payload = _next_batch()
                if payload is None:
                    return None
                epoch_idx, batch_dict = payload
                metrics: dict = {}
                timing_raw: dict = {}
                batch = self._collect_rollout_batch(
                    batch_dict=batch_dict,
                    snapshot_id=snapshot_id,
                    epoch=epoch_idx,
                    metrics=metrics,
                    timing_raw=timing_raw,
                )
                container = {
                    "batch": batch,
                    "metrics": metrics,
                    "timing_raw": timing_raw,
                }
                return ray.put(container)

            def learner_step_async(container_ref):
                return container_ref

            snapshot_id = self.publish_snapshot()
            rollout_fut = rollout_async(snapshot_id)

            while self.global_steps < self.total_training_steps:
                wait_list = [f for f in [rollout_fut, learn_fut] if f is not None]
                if not wait_list:
                    break
                ready, _ = ray.wait(wait_list, num_returns=1)
                if rollout_fut and rollout_fut in ready:
                    pending_payload = ray.get(rollout_fut)
                    rollout_fut = None
                    if pending_payload is not None and learn_fut is None:
                        learn_fut = learner_step_async(ray.put(pending_payload))
                    if pending_payload is None:
                        rollout_exhausted = True
                    if not rollout_exhausted:
                        next_snapshot = self.cur_snapshot_id
                        rollout_fut = rollout_async(next_snapshot)
                if learn_fut and learn_fut in ready:
                    payload = ray.get(learn_fut)
                    learn_fut = None
                    if payload is None:
                        break
                    timing_raw = payload["timing_raw"]
                    with marked_timer("step", timing_raw):
                        finished = self._learn_from_rollout(
                            batch=payload["batch"],
                            metrics=payload["metrics"],
                            timing_raw=timing_raw,
                            logger=logger,
                            progress_bar=progress_bar,
                        )
                    self.publish_snapshot()
                    if finished:
                        break
                if rollout_exhausted and learn_fut is None and rollout_fut is None:
                    break
