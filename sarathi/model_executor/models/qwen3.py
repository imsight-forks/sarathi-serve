# coding=utf-8
"""Inference-only Qwen3 model compatible with HuggingFace weights.

The input of the model is flattened to a 1D tensor of tokens.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch import nn
from transformers import Qwen3Config

from sarathi.metrics.constants import OperationMetrics
from sarathi.metrics.cuda_timer import CudaTimer
from sarathi.model_executor.attention.base_attention_wrapper import BaseAttentionWrapper
from sarathi.model_executor.layers.activation import SiluAndMul
from sarathi.model_executor.layers.layernorm import RMSNorm
from sarathi.model_executor.layers.rotary_embedding import get_rope
from sarathi.model_executor.parallel_utils.parallel_state import (
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
)
from sarathi.model_executor.parallel_utils.pipeline_parallel.mappings import recv, send
from sarathi.model_executor.parallel_utils.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from sarathi.model_executor.weight_utils import (
    hf_model_weights_iterator,
    load_padded_tensor_parallel_vocab,
    load_tensor_parallel_weights,
)


class Qwen3MLP(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        layer_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = ColumnParallelLinear(
            hidden_size,
            2 * intermediate_size,
            bias=False,
            gather_output=False,
            perform_initialization=False,
            linear_metric_name=OperationMetrics.MLP_UP_PROJ,
            communication_metric_name=OperationMetrics.MLP_UP_PROJ_ALL_GATHER,
            layer_id=layer_id,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            perform_initialization=False,
            linear_metric_name=OperationMetrics.MLP_DOWN_PROJ,
            communication_metric_name=OperationMetrics.MLP_DOWN_PROJ_ALL_REDUCE,
            layer_id=layer_id,
        )

        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()
        self._mlp_activation_timer = CudaTimer(OperationMetrics.MLP_ACTIVATION, layer_id=layer_id)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        with self._mlp_activation_timer:
            x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


def _apply_head_rms_norm(x: torch.Tensor, *, norm: RMSNorm, num_heads: int, head_dim: int) -> torch.Tensor:
    # x: [num_tokens, num_heads * head_dim] -> apply RMSNorm over head_dim per head.
    x_3d = x.reshape(-1, num_heads, head_dim)
    x_2d = x_3d.reshape(-1, head_dim)
    x_2d = norm(x_2d)
    return x_2d.reshape(-1, num_heads * head_dim)


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        rms_norm_eps: float = 1e-6,
        layer_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.layer_id = layer_id

        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = int(num_heads)
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        self.total_num_kv_heads = int(num_kv_heads)
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size

        self.head_dim = int(head_dim)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = ColumnParallelLinear(
            self.hidden_size,
            (self.total_num_heads + 2 * self.total_num_kv_heads) * self.head_dim,
            bias=False,
            gather_output=False,
            perform_initialization=False,
            linear_metric_name=OperationMetrics.ATTN_PRE_PROJ,
            communication_metric_name=OperationMetrics.ATTN_PRE_PROJ_ALL_GATHER,
            layer_id=layer_id,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            perform_initialization=False,
            linear_metric_name=OperationMetrics.ATTN_POST_PROJ,
            communication_metric_name=OperationMetrics.ATTN_POST_PROJ_ALL_REDUCE,
            layer_id=layer_id,
        )

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            is_neox_style=True,
            rope_scaling=rope_scaling,
        )
        self._attn_rope_timer = CudaTimer(OperationMetrics.ATTN_ROPE, layer_id=layer_id)

        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        layer_cache_idx: int,
        attention_backend_wrapper: BaseAttentionWrapper,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        with self._attn_rope_timer:
            q, k = self.rotary_emb(positions, q, k)

        q = _apply_head_rms_norm(q, norm=self.q_norm, num_heads=self.num_heads, head_dim=self.head_dim)
        k = _apply_head_rms_norm(k, norm=self.k_norm, num_heads=self.num_kv_heads, head_dim=self.head_dim)

        attn_output = attention_backend_wrapper.forward(
            q,
            k,
            v,
            layer_cache_idx,
            self.scaling,
            self.layer_id,
        )
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        layer_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = int(config.hidden_size)

        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        head_dim = getattr(config, "head_dim", self.hidden_size // int(config.num_attention_heads))

        self.self_attn = Qwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=int(config.num_attention_heads),
            num_kv_heads=int(config.num_key_value_heads),
            head_dim=int(head_dim),
            rope_theta=float(rope_theta),
            rope_scaling=rope_scaling,
            max_position_embeddings=int(max_position_embeddings),
            rms_norm_eps=float(config.rms_norm_eps),
            layer_id=layer_id,
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=int(config.intermediate_size),
            hidden_act=str(config.hidden_act),
            layer_id=layer_id,
        )

        self.input_layernorm = RMSNorm(
            self.hidden_size,
            eps=float(config.rms_norm_eps),
            norm_name=OperationMetrics.INPUT_LAYERNORM,
            layer_id=layer_id,
        )
        self.post_attention_layernorm = RMSNorm(
            self.hidden_size,
            eps=float(config.rms_norm_eps),
            norm_name=OperationMetrics.POST_ATTENTION_LAYERNORM,
            layer_id=layer_id,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        layer_cache_idx: int,
        attention_backend_wrapper: BaseAttentionWrapper,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            layer_cache_idx=layer_cache_idx,
            attention_backend_wrapper=attention_backend_wrapper,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = int(config.vocab_size)

        self.embed_tokens = None
        if is_pipeline_first_stage():
            vocab_size = ((int(config.vocab_size) + 63) // 64) * 64
            self.embed_tokens = VocabParallelEmbedding(
                vocab_size,
                int(config.hidden_size),
                perform_initialization=False,
                linear_metric_name=OperationMetrics.EMBED_LINEAR,
                communication_metric_name=OperationMetrics.EMBED_ALL_REDUCE,
            )

        num_layers = int(config.num_hidden_layers) // get_pipeline_model_parallel_world_size()
        layer_offset = get_pipeline_model_parallel_rank() * num_layers

        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(config, layer_id=layer_id + layer_offset)
                for layer_id in range(num_layers)
            ]
        )

        self.norm = None
        if is_pipeline_last_stage():
            self.norm = RMSNorm(int(config.hidden_size), eps=float(config.rms_norm_eps))

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attention_backend_wrapper: BaseAttentionWrapper,
    ) -> torch.Tensor:
        if self.embed_tokens:
            hidden_states = self.embed_tokens(hidden_states)

        for i, layer in enumerate(self.layers):
            hidden_states = layer(positions, hidden_states, i, attention_backend_wrapper)

        if self.norm:
            hidden_states = self.norm(hidden_states)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)

        vocab_size = ((int(config.vocab_size) + 63) // 64) * 64

        self.is_pipeline_first_stage = is_pipeline_first_stage()
        self.is_pipeline_last_stage = is_pipeline_last_stage()

        self.lm_head = None
        if self.is_pipeline_last_stage:
            self.lm_head = ColumnParallelLinear(
                int(config.hidden_size),
                vocab_size,
                bias=False,
                gather_output=False,
                perform_initialization=False,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attention_backend_wrapper: BaseAttentionWrapper,
    ) -> torch.Tensor:
        if not self.is_pipeline_first_stage:
            hidden_states = torch.empty(
                (positions.shape[0], int(self.config.hidden_size)),
                dtype=self.config.dtype,
                device=hidden_states.device,
            )
            hidden_states = recv(hidden_states)

        hidden_states = self.model(hidden_states, positions, attention_backend_wrapper)

        if not self.is_pipeline_last_stage:
            send(hidden_states)

        return hidden_states

    _column_parallel_layers: List[str] = []
    _row_parallel_layers: List[str] = ["o_proj", "down_proj"]

    def load_weights(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        load_format: str = "auto",
        revision: Optional[str] = None,
    ) -> None:
        weight_suffixes = ["weight"]

        column_parallel_weights: List[str] = []
        for layer in self._column_parallel_layers:
            for suffix in weight_suffixes:
                column_parallel_weights.append(f"{layer}.{suffix}")

        row_parallel_weights: List[str] = []
        for layer in self._row_parallel_layers:
            for suffix in weight_suffixes:
                row_parallel_weights.append(f"{layer}.{suffix}")

        tp_size = get_tensor_model_parallel_world_size()
        pp_size = get_pipeline_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        pp_rank = get_pipeline_model_parallel_rank()

        assert int(self.config.num_hidden_layers) % pp_size == 0
        layers_per_stage = int(self.config.num_hidden_layers) // pp_size
        first_layer_id = layers_per_stage * pp_rank
        last_layer_id = layers_per_stage * (pp_rank + 1) - 1

        head_dim = getattr(self.config, "head_dim", int(self.config.hidden_size) // int(self.config.num_attention_heads))
        q_proj_total = int(self.config.num_attention_heads) * int(head_dim)
        kv_proj_total = int(self.config.num_key_value_heads) * int(head_dim)
        assert q_proj_total % tp_size == 0
        assert kv_proj_total % tp_size == 0

        q_proj_shard_size = q_proj_total // tp_size
        kv_proj_shard_size = kv_proj_total // tp_size
        attention_weight_specs = [
            ("q_proj", q_proj_shard_size, 0),
            ("k_proj", kv_proj_shard_size, q_proj_shard_size),
            ("v_proj", kv_proj_shard_size, q_proj_shard_size + kv_proj_shard_size),
        ]

        state_dict = self.state_dict()

        for name, loaded_weight in hf_model_weights_iterator(
            model_name_or_path, cache_dir, load_format, revision
        ):
            if "rotary_emb.inv_freq" in name:
                continue

            if pp_rank != 0 and "embed_tokens" in name:
                continue

            if pp_rank != pp_size - 1 and ("lm_head" in name or name == "model.norm.weight"):
                continue

            if "model.layers" in name:
                layer_id = int(name.split(".")[2])
                if layer_id < first_layer_id or layer_id > last_layer_id:
                    continue

                new_layer_id = layer_id - first_layer_id
                name = name.replace(str(layer_id), str(new_layer_id))

            is_attention_weight = False
            for weight_name, shard_size, offset in attention_weight_specs:
                if weight_name not in name:
                    continue
                param = state_dict[name.replace(weight_name, "qkv_proj")]

                loaded_weight = loaded_weight[shard_size * tp_rank : shard_size * (tp_rank + 1)]
                param_slice = param.data[offset : offset + shard_size]
                assert param_slice.shape == loaded_weight.shape
                param_slice.copy_(loaded_weight)
                is_attention_weight = True
                break
            if is_attention_weight:
                continue

            is_gate_up_weight = False
            for stride_id, weight_name in enumerate(["gate_proj", "up_proj"]):
                if weight_name not in name:
                    continue
                param = state_dict[name.replace(weight_name, "gate_up_proj")]

                shard_size = param.shape[0] // 2
                loaded_weight = loaded_weight[shard_size * tp_rank : shard_size * (tp_rank + 1)]
                param_slice = param.data[shard_size * stride_id : shard_size * (stride_id + 1)]
                assert param_slice.shape == loaded_weight.shape
                param_slice.copy_(loaded_weight)
                is_gate_up_weight = True
                break
            if is_gate_up_weight:
                continue

            param = state_dict[name]

            if "embed_tokens" in name or "lm_head" in name:
                load_padded_tensor_parallel_vocab(param, loaded_weight, tp_rank)
                continue

            load_tensor_parallel_weights(
                param,
                loaded_weight,
                name,
                column_parallel_weights,
                row_parallel_weights,
                tp_rank,
            )

