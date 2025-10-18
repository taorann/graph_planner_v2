# coding=utf-8
# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen2.py
# Copyright 2024 The Codefuse team.
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""Inference-only Qwen2 model compatible with HuggingFace weights."""
from typing import Iterable, List, Optional, Tuple

import time
import torch
from torch import nn
from transformers import Qwen2Config

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplerOutput
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.image import get_dummy_image_data

from vllm.attention.backends.xformers import XFormersBackend


class Attention(nn.Module):
    """Attention layer.

    This class takes query, key, and value tensors as input. The input tensors
    can either contain prompt tokens or generation tokens.
    The class does the following:

    1. Store the input key and value tensors in the KV cache.
    2. Perform (multi-head/multi-query/grouped-query) attention.
    3. Return the output tensor.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        blocksparse_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
            block_size = cache_config.block_size
            sliding_window = cache_config.sliding_window
        else:
            kv_cache_dtype = "auto"
            block_size = 16
            sliding_window = None
        if num_kv_heads is None:
            num_kv_heads = num_heads

        # The default kv_scale is set to 1.0. This is ignored
        # when kv-cache is not fp8, and should be used with
        # kv-cache in fp8_e5m2. For kv-cache in fp8_e4m3, we
        # expect the pre-quantized kv_scale to be loaded along
        # with the model weights.
        self.kv_cache_dtype = kv_cache_dtype
        self._kv_scale = 1.0
        quant_method = quant_config.get_quant_method(
            self) if quant_config else None
        if quant_method is not None:
            if self.kv_cache_dtype == "fp8_e5m2":
                raise ValueError("fp8_e5m2 kv-cache is not supported with "
                                 "fp8 checkpoints.")
            # When FP8 quantization is enabled, we make a parameter
            # "kv_scale" so that it can be loaded from FP8 checkpoint.
            # The kv_scale will then be converted back
            # to self._kv_scale in a native float32 value after weight loading.
            self.quant_method = quant_method
            self.quant_method.create_weights(self)

        # During model initialization, the default dtype is set as the model
        # weight and activation dtype.
        dtype = torch.get_default_dtype()
        attn_backend = XFormersBackend
        # attn_backend = get_attn_backend(num_heads, head_size, num_kv_heads,
        #                                 sliding_window, dtype, kv_cache_dtype,
        #                                 block_size, blocksparse_params
        #                                 is not None)
        impl_cls = attn_backend.get_impl_cls()
        self.impl = impl_cls(num_heads, head_size, scale, num_kv_heads,
                             alibi_slopes, sliding_window, kv_cache_dtype,
                             blocksparse_params)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        return self.impl.forward(query, key, value, kv_cache, attn_metadata,
                                 self._kv_scale)

    def extra_repr(self) -> str:
        s = f"head_size={self.impl.head_size}"  # type: ignore
        s += f", num_heads={self.impl.num_heads}"  # type: ignore
        s += f", num_kv_heads={self.impl.num_kv_heads}"  # type: ignore
        s += f", scale={self.impl.scale}"  # type: ignore
        s += f", backend={self.impl.__class__.__name__}"
        return s



class Qwen2MLP(nn.Module):

    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            hidden_act: str,
            quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config)
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           quant_config=quant_config)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Qwen2Attention(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 max_position: int = 4096 * 32,
                 rope_theta: float = 10000,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 rope_scaling: Optional[Tuple] = None) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.rope_theta = rope_theta

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=self.rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config)

    def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen2DecoderLayer(nn.Module):

    def __init__(
            self,
            config: Qwen2Config,
            cache_config: Optional[CacheConfig] = None,
            quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        self.self_attn = Qwen2Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            cache_config=cache_config,
            quant_config=quant_config,
            rope_scaling=rope_scaling)
        self.mlp = Qwen2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: AttentionMetadata,
            residual: Optional[torch.Tensor],
            attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class CGMQwen2Model(nn.Module):

    def __init__(
            self,
            config: Qwen2Config,
            cache_config: Optional[CacheConfig] = None,
            quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList([
            Qwen2DecoderLayer(config, cache_config, quant_config)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[torch.Tensor],
            attn_metadata: AttentionMetadata,
            inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i],
                attn_metadata,
                residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class PerceiverResamplerLayer(nn.Module):
    def __init__(self, args, layer_idx):
        super(PerceiverResamplerLayer, self).__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=args['lm_hidden_size'],
            num_heads=args['num_heads'],
            kdim=args['graph_embedding_dim'],
            vdim=args['graph_embedding_dim'],
            batch_first=True
        )
        self.layer_idx = layer_idx

    def forward(self, graph_embedding, queries):
        features = graph_embedding.to(queries.dtype).unsqueeze(0)
        hidden = self.attn(queries, features, features, need_weights=False)[0]
        return hidden


class PerceiverResampler(nn.Module):
    def __init__(self, args):
        super(PerceiverResampler, self).__init__()
        self.num_layers = args.adapter_num_layers
        self.layers = nn.ModuleList([PerceiverResamplerLayer(args, i) for i in range(self.num_layers)])

    def forward(self, graph_embedding, queries):
        for layer in self.layers:
            queries = layer(graph_embedding, queries)
        return queries


class Adapter(nn.Module):
    """Define the Adapter part of Code Graph Model (CGM).
    """
    def __init__(self, args):
        super(Adapter, self).__init__()
        self.args = args

        self.q = nn.Parameter(torch.randn(args['graph_token_num'], args['lm_hidden_size']))
        self.attn = nn.MultiheadAttention(
            embed_dim=args['lm_hidden_size'],
            num_heads=args['num_heads'],
            kdim=args['graph_embedding_dim'],
            vdim=None,
            batch_first=True
        )

    def forward(self, features):
        """Forward.
        Args:
            features: torch.Tensor
        """
        # print(f'LBC - Adapter features type: {features.dtype}, shape: {features.shape}')
        features_2d = features.to(self.q.dtype)
        queries = self.q
        embeddings = self.attn(queries, features_2d, features_2d, need_weights=False)[0]
        return embeddings


class Adapter_v2(nn.Module):
    def __init__(self, args):
        super(Adapter_v2, self).__init__()
        # self.fc1 = nn.Linear(args.embedding_dim, args.adapter_hidden_dim)
        self.fc1 = RowParallelLinear(args.embedding_dim, args.adapter_hidden_dim)
        self.gelu = nn.GELU()
        # self.fc2 = nn.Linear(args.adapter_hidden_dim, args.lm_hidden_dim)
        self.fc2 = RowParallelLinear(args.adapter_hidden_dim, args.lm_hidden_dim)

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))

@MULTIMODAL_REGISTRY.register_image_feature_input()
@MULTIMODAL_REGISTRY.register_image_pixel_input()
@MULTIMODAL_REGISTRY.register_dummy_data(get_dummy_image_data)
class CGMQwen2ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ]
    embedding_modules = {}
    embedding_padding_modules = []

    def __init__(
            self,
            config: Qwen2Config,
            cache_config: Optional[CacheConfig] = None,
            quant_config: Optional[QuantizationConfig] = None,
            lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        # print(f'LBC - config in CGMQwen2: {vars(config)}')
        del lora_config
        # TODO (@robertgshaw2): see if this can be moved out
        if (cache_config.sliding_window is not None
                and hasattr(config, "max_window_layers")):
            raise ValueError("Sliding window for some but all layers is not "
                             "supported. This model uses sliding window "
                             "but `max_window_layers` = %s is less than "
                             "`num_hidden_layers` = %s. Please open an issue "
                             "to discuss this feature." % (
                                 config.max_window_layers,
                                 config.num_hidden_layers,
                             ))

        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = CGMQwen2Model(config, cache_config, quant_config)

        if config.tie_word_embeddings:
            self.lm_head_weight = self.model.embed_tokens.weight
        else:
            self.lm_head = ParallelLMHead(config.vocab_size,
                                          config.hidden_size)
            self.lm_head_weight = self.lm_head.weight

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()

        adapter_args = None
        if 'adapter_config' in vars(config):
            adapter_args = vars(config)['adapter_config']
            adapter_args['lm_hidden_size'] = config.hidden_size
            # print(f'LBC - adatper args: {adapter_args}')
            start_time = time.time()
            self.adapter = Adapter_v2(adapter_args)
            self.adapter.load_state_dict(torch.load(f"{config._name_or_path}/adapter.pth"))
            end_time = time.time()
            # print(f'Loading adapter model takes {end_time - start_time} seconds')
        else:
            self.adapter = None

        self.adapter_args = adapter_args

    def _parse_and_validate_graph_input(self, **kwargs: object) -> torch.Tensor:
        image_features = kwargs.pop("image_features", None)

        return image_features

    # def forward(
    #         self,
    #         input_ids: torch.Tensor,
    #         positions: torch.Tensor,
    #         kv_caches: List[torch.Tensor],
    #         attn_metadata: AttentionMetadata,
    #         node_repre: Optional[torch.Tensor] = None,
    #         **kwargs: object,
    # ) -> torch.Tensor:
    #     node_repre = self._parse_and_validate_graph_input(**kwargs)
    # 
    #     # if self.args.peft == "LoRA":
    #     #     inputs_embeds = self.lm.model.model.embed_tokens(x)
    #     # else:
    #     #     inputs_embeds = self.lm.model.embed_tokens(x)
    #     # embeddings = self.adapter(node_repre, inputs_embeds)
    #     # lm_embeds = torch.cat((embeddings, inputs_embeds), dim=1)
    #     #
    #     # outputs = self.lm(inputs_embeds=lm_embeds,
    #     #                   return_dict=True)
    # 
    #     # Merge Adapter.forward with LLM forward
    #     if self.adapter and node_repre is not None:
    #         inputs_embeds = self.model.embed_tokens(input_ids)
    #         # print(f'LBC - node_repre: {node_repre[0][:100]}')
    #         embeddings = self.adapter(node_repre[0], inputs_embeds)
    #         inputs_embeds = torch.cat((embeddings, inputs_embeds), dim=1)
    # 
    #         input_ids = None
    # 
    #     else:
    #         inputs_embeds = None
    # 
    #     hidden_states = self.model(
    #         input_ids=input_ids,
    #         positions=positions,
    #         kv_caches=kv_caches,
    #         attn_metadata=attn_metadata,
    #         inputs_embeds=inputs_embeds
    #     )
    # 
    #     return hidden_states

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[torch.Tensor],
            attn_metadata: AttentionMetadata,
            node_repre: Optional[torch.Tensor] = None,
            adj_matrix: Optional[torch.Tensor] = None,
            **kwargs: object,
    ) -> torch.Tensor:
        node_repre = self._parse_and_validate_graph_input(**kwargs)
        
        # Merge Adapter.forward with LLM forward
        if self.adapter and node_repre is not None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            # print(f'LBC - node_repre: {node_repre[0][:100]}')
            embeddings = self.adapter(node_repre[0])
            inputs_embeds = torch.cat((embeddings, inputs_embeds), dim=-2)

            if adj_matrix is not None:
                batch_size, seq_len_x, _ = adj_matrix.shape
                seq_len_q = input_ids.size(1)
            
                qa_matrix = torch.ones(batch_size, seq_len_q, seq_len_q, device=inputs_embeds.device)
                matrix_xq = torch.ones(batch_size, seq_len_x, seq_len_q, device=inputs_embeds.device)
                matrix_qx = torch.ones(batch_size, seq_len_q, seq_len_x, device=inputs_embeds.device)
            
                attention_mask = torch.cat([
                    torch.cat([adj_matrix, matrix_xq], dim=2),
                    torch.cat([matrix_qx, qa_matrix], dim=2)
                ], dim=1).unsqueeze(1)
            else:
                attention_mask = None

            input_ids = None

        else:
            inputs_embeds = None

        if attention_mask:
            attn_metadata.attn_bias = attention_mask

        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds
        )

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head_weight, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
            self,
            logits: torch.Tensor,
            sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # Only LLM but adpter is loaded now.
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
