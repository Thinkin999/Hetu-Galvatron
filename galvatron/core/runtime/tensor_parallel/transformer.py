# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Transformer."""
import math
import os
from contextlib import nullcontext
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from megatron import core
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.utils import deprecate_inference_params
from megatron.legacy.model.enums import AttnMaskType, LayerType, AttnType
from megatron.legacy.model.fused_softmax import FusedScaleMaskSoftmax
from megatron.legacy.model.fused_bias_gelu import bias_gelu_impl
from megatron.core.models.common.embeddings import apply_rotary_pos_emb
from megatron.core.jit import jit_fuser
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.parallel_state import (
    get_expert_tensor_and_model_parallel_group,
    get_tensor_model_parallel_group,
)
from megatron.core.tensor_parallel import (
    gather_from_sequence_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    get_cuda_rng_tracker,
    get_data_parallel_rng_tracker_name,
)
from megatron.legacy.model.enums import AttnMaskType, AttnType, LayerType
from megatron.legacy.model.fused_bias_gelu import bias_gelu_impl
from megatron.legacy.model.fused_softmax import FusedScaleMaskSoftmax
from megatron.legacy.model.utils import (
    attention_mask_func,
    erf_gelu,
    get_norm,
    openai_gelu,
)
from megatron.training import get_args, get_timers

from megatron.legacy.model.module import MegatronModule

try:
    from einops import rearrange
except ImportError:
    rearrange = None

try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_func
except ImportError:
    try:
        from flash_attn.flash_attn_interface import (
            flash_attn_varlen_func as flash_attn_unpadded_func,
        )
    except ImportError:
        flash_attn_unpadded_func = None

""" We use the following notation throughout this file:
     h: hidden size
     n: number of attention heads
     p: number of model parallel partitions
     np: n/p
     hp: h/p
     hn: h/n
     b: batch size
     s: sequence length
     l: number of layers
    Transformer takes input of size [s, b, h] and returns a
    tensor of the same size. We use the following arguments:
        hyperparameters: transformer hyperparameters
"""

class ParallelMLP(MegatronModule):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, config, is_expert=False, tp_group=None):
        super(ParallelMLP, self).__init__()
        args = get_args()

        self.add_bias = config.add_bias_linear

        ffn_hidden_size = config.ffn_hidden_size
        if config.gated_linear_unit:
            ffn_hidden_size *= 2

        # Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        self.dense_h_to_4h = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            ffn_hidden_size,
            config=config,
            init_method=config.init_method,
            bias=self.add_bias,
            gather_output=False,
            skip_bias_add=True,
            is_expert=is_expert,
            tp_group=tp_group,
        )

        self.bias_gelu_fusion = False
        self.activation_func = None
        self.swiglu = args.swiglu

        if args.openai_gelu:
            self.activation_func = openai_gelu
        elif args.onnx_safe:
            self.activation_func = erf_gelu
        elif args.swiglu:
            def swiglu(x):
                x = torch.chunk(x, 2, dim=-1)
                return F.silu(x[0]) * x[1]
            self.activation_func = swiglu
        elif args.squared_relu:
            def squared_relu(x):
                return torch.pow(F.relu(x), 2)
            self.activation_func = squared_relu
        else:
            self.bias_gelu_fusion = args.bias_gelu_fusion
            self.activation_func = F.gelu

        # Project back to h.
        self.dense_4h_to_h = tensor_parallel.RowParallelLinear(
            config.ffn_hidden_size,
            config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=self.add_bias,
            skip_bias_add=True,
            input_is_parallel=True,
            is_expert=is_expert,
            tp_group=tp_group,
        )

    def forward(self, hidden_states):

        # [s, b, 4hp]
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)

        if self.bias_gelu_fusion:
            assert self.add_bias is True
            assert self.activation_func == F.gelu
            intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
        else:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            intermediate_parallel = self.activation_func(intermediate_parallel)

        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)
        return output, output_bias


class CoreAttention(MegatronModule):

    def __init__(self, layer_number, config, attn_mask_type=AttnMaskType.padding, tp_group=None, sp_group=None):
        super(CoreAttention, self).__init__()
        self.fp16 = config.fp16
        self.bf16 = config.bf16

        self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.sequence_parallel = config.sequence_parallel

        projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        world_size = mpu.get_tensor_model_parallel_world_size(tp_group)
        if sp_group is None:
            sp_world_size = 1
        else:
            sp_world_size = torch.distributed.get_world_size(sp_group)
        world_size = max(world_size, sp_world_size)
        self.hidden_size_per_partition = core.utils.divide(projection_size, world_size)
        self.hidden_size_per_attention_head = core.utils.divide(projection_size, config.num_attention_heads)
        self.num_attention_heads_per_partition = core.utils.divide(config.num_attention_heads, world_size)

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            self.fp16,
            self.bf16,
            self.attn_mask_type,
            config.masked_softmax_fusion,
            attention_mask_func,
            self.attention_softmax_in_fp32,
            coeff,
        )

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(config.attention_dropout)

    def forward(self, query_layer, key_layer, value_layer, attention_mask):

        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (query_layer.size(1), query_layer.size(2), query_layer.size(0), key_layer.size(0))

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.reshape(output_size[2], output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

        # preallocting input tensor: [b * np, sq, sk]
        matmul_input_buffer = mpu.get_global_memory_buffer().get_tensor(
            (output_size[0] * output_size[1], output_size[2], output_size[3]), query_layer.dtype, "mpu"
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query_layer.transpose(0, 1),  # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        if not self.sequence_parallel:
            with tensor_parallel.get_cuda_rng_tracker().fork():
                attention_probs = self.attention_dropout(attention_probs)
        else:
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.size(1), value_layer.size(2), query_layer.size(0), value_layer.size(3))

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(value_layer.size(0), output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class FlashSelfOrCrossAttention(torch.nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0, device=None, dtype=None):
        super().__init__()
        assert flash_attn_unpadded_func is not None, (
            "Please install FlashAttention first, " "e.g., with pip install flash-attn"
        )
        assert rearrange is not None, "Please install einops first, e.g., with pip install einops"
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, q, k, v):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q, k, v: The tensor containing the query, key, and value. (B, S, H, D)
        """

        assert all((i.dtype in [torch.float16, torch.bfloat16] for i in (q, k, v)))
        assert all((i.is_cuda for i in (q, k, v)))

        batch_size, seqlen_q = q.shape[0], q.shape[1]
        seqlen_k = k.shape[1]

        q, k, v = [rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v]]
        cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q.device)

        is_causal = self.causal
        if seqlen_k == seqlen_q:
            cu_seqlens_k = cu_seqlens_q
        else:
            cu_seqlens_k = torch.arange(
                0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32, device=k.device
            )
        if self.training:
            dropout_p = self.dropout_p
        else:
            dropout_p = 0
        # if self.training:
        #     # during training q,k,v always have same seqlen
        #     assert seqlen_k == seqlen_q

        #     is_causal = self.causal
        #     cu_seqlens_k = cu_seqlens_q
        #     dropout_p = self.dropout_p
        # else:
        #     # turn off FA causal mask after first inference autoregressive iteration
        #     # only on first autoregressive step q,k,v have same seqlen
        #     is_causal = seqlen_q == seqlen_k
        #     cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32,
        #                 device=q.device)
        #     dropout_p = 0

        output = flash_attn_unpadded_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            seqlen_q,
            seqlen_k,
            dropout_p,
            softmax_scale=self.softmax_scale,
            causal=is_causal,
        )

        output = rearrange(output, "(b s) ... -> b s ...", b=batch_size)
        return output


class ParallelAttention(MegatronModule):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        config,
        layer_number,
        attention_type=AttnType.self_attn,
        attn_mask_type=AttnMaskType.padding,
        tp_group=None,
        sp_group=None,
        use_ulysses=False,
    ):
        super(ParallelAttention, self).__init__()
        args = get_args()
        self.use_ulysses = use_ulysses

        self.layer_number = max(1, layer_number)
        self.attention_type = attention_type
        self.attn_mask_type = attn_mask_type
        self.params_dtype = config.params_dtype
        self.sequence_parallel = config.sequence_parallel
        self.config = config
        self.group_query_attention = args.group_query_attention
        self.num_query_groups = args.num_query_groups

        query_projection_size = config.kv_channels * config.num_attention_heads
        if self.group_query_attention:
            kv_projection_size = args.kv_channels * args.num_query_groups
        else:
            kv_projection_size = args.kv_channels * args.num_attention_heads

        self.use_flash_attn = args.use_flash_attn  #  \
        # and attention_type == AttnType.self_attn \
        # and self.attn_mask_type == AttnMaskType.causal
        if self.use_flash_attn:
            if flash_attn_unpadded_func is None:
                raise ImportError("FlashAttention is not installed, please install with " "pip install flash-attn")
            # assert attention_type == AttnType.self_attn, ('FlashAttention code path only supports '
            #                                               'self-attention for now')
            # assert self.attn_mask_type == AttnMaskType.causal, ('FlashAttention code path only '
            #                                                     'supports causal mask for now')
            if rearrange is None:
                raise ImportError("einops is not installed, please install with pip install einops")

        # Per attention head and per partition values.
        world_size = mpu.get_tensor_model_parallel_world_size(tp_group)
        if sp_group is None:
            sp_world_size = 1
        else:
            sp_world_size = torch.distributed.get_world_size(sp_group)
        self.hidden_size_per_attention_head = core.utils.divide(query_projection_size, config.num_attention_heads)
        self.num_attention_heads_per_partition = core.utils.divide(config.num_attention_heads, world_size)

        if self.group_query_attention:
            if args.num_query_groups % world_size != 0:
                raise NotImplementedError(
                    "Currently the num_query_groups should be " "a multiple of the tensor parallel size"
                )
            self.num_query_groups_per_partition = core.utils.divide(args.num_query_groups, world_size)
        else:
            self.num_query_groups_per_partition = self.num_attention_heads_per_partition

        # Strided linear layer.
        if attention_type == AttnType.self_attn:
            self.query_key_value = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                query_projection_size + 2 * kv_projection_size,
                config=config,
                init_method=config.init_method,
                bias=args.add_bias_linear or args.add_qkv_bias,
                gather_output=False,
                tp_group=tp_group,
            )
        else:
            assert attention_type == AttnType.cross_attn

            if self.group_query_attention:
                raise NotImplementedError("Grouped query attention not implemented for cross-attention.")
            assert query_projection_size == kv_projection_size

            self.query = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                query_projection_size,
                config=config,
                init_method=config.init_method,
                bias=config.add_bias_linear,
                gather_output=False,
                tp_group=tp_group,
            )

            self.key_value = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                2 * kv_projection_size,
                config=config,
                init_method=config.init_method,
                bias=config.add_bias_linear,
                gather_output=False,
                tp_group=tp_group,
            )

        self.core_attention = CoreAttention(
            self.layer_number, config, self.attn_mask_type, tp_group=tp_group, sp_group=sp_group
        )
        self.checkpoint_core_attention = config.recompute_granularity == "selective"

        if self.use_flash_attn:
            self.core_attention_flash = FlashSelfOrCrossAttention(
                causal=(attn_mask_type == AttnMaskType.causal), attention_dropout=config.attention_dropout
            )
        if self.use_ulysses:
            assert args.num_attention_heads % sp_world_size == 0
            self.dist_attn = DistributedAttention(
                self.core_attention_flash if self.use_flash_attn else self.core_attention,
                sp_group,
                gather_idx=1 if self.use_flash_attn else 0,
            )
            # flash attn [B,S,H,D] gather_idx = 1, normal attn [S,B,H,D] gather_idx = 0
        # Output.
        self.dense = tensor_parallel.RowParallelLinear(
            query_projection_size,
            config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=args.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            tp_group=tp_group,
        )

    def forward(self, hidden_states, attention_mask, encoder_output=None, rotary_pos_emb=None):
        # hidden_states: [sq, b, h]
        # =====================
        # Query, Key, and Value
        # =====================
        if self.attention_type == AttnType.self_attn:

            # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
            mixed_x_layer, _ = self.query_key_value(hidden_states)

            # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
            new_tensor_shape = mixed_x_layer.size()[:-1] + (
                self.num_query_groups_per_partition,
                (
                    (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                    * self.hidden_size_per_attention_head
                ),
            )
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query_layer, key_layer, value_layer) = torch.split(
                mixed_x_layer,
                [
                    (
                        self.num_attention_heads_per_partition
                        // self.num_query_groups_per_partition
                        * self.hidden_size_per_attention_head
                    ),
                    self.hidden_size_per_attention_head,
                    self.hidden_size_per_attention_head,
                ],
                dim=3,
            )

            # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn] -
            # TODO: check if it is necessary to reshape
            if self.group_query_attention:
                query_layer = query_layer.reshape(
                    query_layer.size(0), query_layer.size(1), -1, self.hidden_size_per_attention_head
                )
            else:
                query_layer = query_layer.view(
                    query_layer.size(0), query_layer.size(1), -1, self.hidden_size_per_attention_head
                )
        else:
            # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
            mixed_kv_layer, _ = self.key_value(encoder_output)

            # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
            new_tensor_shape = mixed_kv_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                2 * self.hidden_size_per_attention_head,
            )
            mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)

            # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
            (key_layer, value_layer) = tensor_parallel.split_tensor_along_last_dim(mixed_kv_layer, 2)

            # Attention head [sq, b, h] --> [sq, b, hp]
            query_layer, _ = self.query(hidden_states)
            # [sq, b, hp] --> [sq, b, np, hn]
            new_tensor_shape = query_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
            )
            query_layer = query_layer.view(*new_tensor_shape)

        # ==================================
        # Adjust key and value for inference
        # ==================================

        # duplicate the pos_emb for self attention
        if rotary_pos_emb is not None:
            if isinstance(rotary_pos_emb, tuple):
                rotary_pos_emb = rotary_pos_emb
            else:
                rotary_pos_emb = (rotary_pos_emb,) * 2

        # ==================================
        # core attention computation
        # ==================================

        # expand the key_layer and value_layer [sk, b, ng, hn] -> [sk, b, np, hn]
        if self.num_attention_heads_per_partition // self.num_query_groups_per_partition > 1:
            key_layer = key_layer.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition, dim=2
            )
            value_layer = value_layer.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition, dim=2
            )

        # apply relative positional encoding (rotary embedding)
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            query_layer = apply_rotary_pos_emb(query_layer, q_pos_emb, self.config)
            key_layer = apply_rotary_pos_emb(key_layer, k_pos_emb, self.config)
            # TODO, can apply positional embedding to value_layer so it has
            # absolute positional embedding.
            # otherwise, only relative positional embedding takes effect
            # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)

        if self.use_ulysses:
            if self.use_flash_attn:
                batch_dim_idx = 0
                query_layer, key_layer, value_layer = [
                    rearrange(x, "s b ... -> b s ...").contiguous() for x in (query_layer, key_layer, value_layer)
                ]

                context_layer = self.dist_attn(query_layer, key_layer, value_layer, batch_dim_idx)
                context_layer = rearrange(context_layer, "b s h d -> s b (h d)").contiguous()
            else:
                batch_dim_idx = 1  # [S,B,H,D]
                context_layer = self.dist_attn(query_layer, key_layer, value_layer, batch_dim_idx, attention_mask)
                context_layer = rearrange(context_layer, "... h d -> ... (h d)").contiguous()
        else:
            if not self.use_flash_attn:
                if self.checkpoint_core_attention:
                    context_layer = self._checkpointed_attention_forward(
                        query_layer, key_layer, value_layer, attention_mask
                    )
                else:
                    context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask)
            else:
                q, k, v = [
                    rearrange(x, "s b ... -> b s ...").contiguous() for x in (query_layer, key_layer, value_layer)
                ]
                if not self.sequence_parallel:
                    with tensor_parallel.get_cuda_rng_tracker().fork():
                        context_layer = self.core_attention_flash(q, k, v)
                else:
                    context_layer = self.core_attention_flash(q, k, v)
                context_layer = rearrange(context_layer, "b s h d -> s b (h d)").contiguous()

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.dense(context_layer)

        return output, bias


# --------- ulysses --------------

from typing import Any, Tuple

import torch.distributed as dist
from torch import Tensor
from torch.nn import Module


def post_all2all(scatter_idx, batch_dim_idx, seq_world_size, bs, seq_len, num_head, head_dim):

    def post_func(input):
        if batch_dim_idx == 0:
            # b, s, n, h
            if scatter_idx < 2:
                output = input.permute(1, 2, 0, 3, 4).contiguous()
                output = output.reshape(bs, seq_len // seq_world_size, seq_world_size * num_head, head_dim).contiguous()
            else:
                output = input.permute(1, 0, 2, 3, 4).contiguous()
                output = output.reshape(bs, seq_world_size * seq_len, num_head // seq_world_size, head_dim).contiguous()
        else:
            # s, b, n, h
            if scatter_idx < 2:
                output = input.transpose(0, 1).transpose(1, 2).contiguous()
                # output = input.permute(1, 2, 0, 3, 4).contiguous()
                output = output.reshape(seq_len // seq_world_size, bs, seq_world_size * num_head, head_dim).contiguous()
            else:
                output = input.reshape(seq_len * seq_world_size, bs, num_head // seq_world_size, head_dim).contiguous()
        return output

    return post_func


def single_all_to_all(input, scatter_idx, gather_idx, batch_dim_idx, group, async_op=False, handle=None, type=None):
    seq_world_size = dist.get_world_size(group)
    if batch_dim_idx == 0:
        # b, s, n, h
        if scatter_idx < 2:
            bs, global_seq_len, num_local_head, head_dim = input.shape
            input_t = input.reshape(
                [bs, seq_world_size, global_seq_len // seq_world_size, num_local_head, head_dim]
            ).contiguous()
            input_t = input_t.permute(1, 0, 2, 3, 4).contiguous()
        else:
            bs, local_seq_len, num_total_head, head_dim = input.shape
            assert (
                num_total_head % seq_world_size == 0
            ), f"Number of heads ({num_total_head}) must be divisible by the sequence parallel size ({seq_world_size})!"
            input_t = input.reshape(
                [bs, local_seq_len, seq_world_size, num_total_head // seq_world_size, head_dim]
            ).contiguous()
            input_t = input_t.permute(2, 0, 1, 3, 4).contiguous()
    else:
        # s, b, n, h
        if scatter_idx < 2:
            global_seq_len, bs, num_local_head, head_dim = input.shape
            input_t = input.reshape(
                [seq_world_size, global_seq_len // seq_world_size, bs, num_local_head, head_dim]
            ).contiguous()
        else:
            local_seq_len, bs, num_total_head, head_dim = input.shape
            assert (
                num_total_head % seq_world_size == 0
            ), f"Number of heads ({num_total_head}) must be divisible by the sequence parallel size ({seq_world_size})!"
            input_t = input.reshape(
                [local_seq_len * bs, seq_world_size, num_total_head // seq_world_size, head_dim]
            ).contiguous()
            input_t = input_t.transpose(0, 1).contiguous()
            # input_t = input.reshape([local_seq_len, bs, seq_world_size, num_total_head // seq_world_size,
            #                          head_dim]).contiguous()
            # input_t = input_t.permute(2, 0, 1, 3, 4).contiguous()

    if scatter_idx < 2:
        post_all2all_fun = post_all2all(
            scatter_idx, batch_dim_idx, seq_world_size, bs, global_seq_len, num_local_head, head_dim
        )
    else:
        post_all2all_fun = post_all2all(
            scatter_idx, batch_dim_idx, seq_world_size, bs, local_seq_len, num_total_head, head_dim
        )

    output = torch.empty_like(input_t)
    work = dist.all_to_all_single(output, input_t, group=group, async_op=async_op)

    if async_op:
        if type in ("dq", "dk"):
            handle[type + "_work"] = work
            handle[type + "_grad"] = output
            handle[type + "_post_all2all_func"] = post_all2all_fun
            return output

    res = post_all2all_fun(output)
    return res


class _SeqAllToAll(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        input: Tensor,
        scatter_idx: int,
        gather_idx: int,
        batch_dim_idx: int,
        stream=None,
        handle=None,
        type=None,
        is_fwd=True,
    ) -> Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        ctx.stream = stream
        ctx.handle = handle
        ctx.type = type
        ctx.batch_dim_idx = batch_dim_idx
        if ctx.handle is None:
            res = single_all_to_all(input, scatter_idx, gather_idx, batch_dim_idx, group, False)

        else:
            # overlap communication path
            if not is_fwd and type == "o":
                assert ctx.stream != None
                res = single_all_to_all(input, scatter_idx, gather_idx, batch_dim_idx, group, False)
                get_accelerator().current_stream().wait_stream(ctx.stream)
                del ctx.stream.activation_buffer_list
                # The computation of d o_weight can overlap with the communication of d o_input

            elif not is_fwd and type in ("q", "k"):
                # Achieve communication overlap by pipelining the matrix computation and communication of dq, dk, and dv
                type = "d" + type
                res = single_all_to_all(input, scatter_idx, gather_idx, batch_dim_idx, group, True, handle, type)

            elif is_fwd and type in ("q", "k"):
                # Achieve communication overlap by pipelining the matrix computation and communication of q, k, and v
                type = "fwd_" + type
                res = single_all_to_all(input, scatter_idx, gather_idx, batch_dim_idx, group, False, handle, type)

            else:
                res = single_all_to_all(input, scatter_idx, gather_idx, batch_dim_idx, group, False)

        return res

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None, None]:

        return (
            None,
            _SeqAllToAll.apply(
                ctx.group,
                *grad_output,
                ctx.gather_idx,
                ctx.scatter_idx,
                ctx.batch_dim_idx,
                ctx.stream,
                ctx.handle,
                ctx.type,
                False,
            ),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class DistributedAttention(torch.nn.Module):
    """Initialization.

    Arguments:
        local_attention (Module): local attention with q,k,v
        sequence_process_group (ProcessGroup): sequence parallel process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
    """

    def __init__(
        self,
        local_attention: Module,
        sequence_process_group: dist.ProcessGroup,
        scatter_idx: int = 2,
        gather_idx: int = 0,
        sp_stream=None,
    ) -> None:

        super(DistributedAttention, self).__init__()
        self.local_attn = local_attention
        self.spg = sequence_process_group
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        self.sp_overlap_comm = False
        self.overlap_handles = None
        self.sp_stream = sp_stream
        if sp_stream is not None:
            self.overlap_handles = {}
            self.sp_overlap_comm = True
            self.dafult_stream = get_accelerator().default_stream()

    def layer_sync(self, layer):
        if self.sp_overlap_comm and hasattr(layer, "done_event"):
            self.dafult_stream.wait_event(layer.done_event)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, batch_dim_idx: int, *args: Any, **kwargs) -> Tensor:
        """forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            batch_dim_idx (int): indicating which dim is batch
            args: other args

        Returns:
            * output (Tensor): context output
        """

        # TODO Merge three alltoall calls into one
        # TODO (Reza): change the api on the megatron-deepspeed side so that we only receive all data (q,k, and v) together!
        # in shape : e.g.,  [s/p:h:]

        def bwd_hook(layer_type):

            def pre_hook_fun(grad):
                type = "d" + layer_type
                self.overlap_handles[type + "_work"].wait()
                self.sp_stream.wait_stream(self.dafult_stream)
                all2all_output = self.overlap_handles[type + "_grad"]
                grad = list(grad)
                grad[0] = self.overlap_handles[type + "_post_all2all_func"](all2all_output)
                grad = tuple(grad)

            return pre_hook_fun

        if torch.distributed.get_world_size(self.spg) > 1:
            self.layer_sync(query)
            query_layer = _SeqAllToAll.apply(
                self.spg, query, self.scatter_idx, self.gather_idx, batch_dim_idx, None, self.overlap_handles, "q"
            )
            self.layer_sync(key)
            key_layer = _SeqAllToAll.apply(
                self.spg, key, self.scatter_idx, self.gather_idx, batch_dim_idx, None, self.overlap_handles, "k"
            )
            if self.sp_overlap_comm:
                self.dafult_stream.wait_stream(self.sp_stream)
            value_layer = _SeqAllToAll.apply(
                self.spg, value, self.scatter_idx, self.gather_idx, batch_dim_idx, None, self.overlap_handles, "v"
            )
            if self.sp_overlap_comm:
                # Register a hook to synchronize dq and dk after the all-to-all
                # operation when the gradient data is used.
                # Place this logic after the q, k, v all-to-all operation to
                # improve interpreter speed to
                # call and launch of the forward all-to-all communication.
                grad_fn_q = query.grad_fn.next_functions[0][0]
                grad_fn_q.register_prehook(bwd_hook(layer_type="q"))
                grad_fn_k = key.grad_fn.next_functions[0][0]
                grad_fn_k.register_prehook(bwd_hook(layer_type="k"))
        else:
            query_layer, key_layer, value_layer = query, key, value

        # out shape : e.g., [s:h/p:]
        head_dim = query_layer.shape[-1]
        context_layer = self.local_attn(query_layer, key_layer, value_layer, *args, **kwargs)
        context_layer = context_layer.view(context_layer.shape[0], context_layer.shape[1], -1, head_dim)
        if torch.distributed.get_world_size(self.spg) > 1:
            output = _SeqAllToAll.apply(
                self.spg,
                context_layer,
                self.gather_idx,
                self.scatter_idx,
                batch_dim_idx,
                self.sp_stream,
                self.overlap_handles,
                "o",
            )
        else:
            output = context_layer
        # out e.g., [s/p::h]
        return output
