import torch
import torch.nn as nn
from typing import Tuple
from torch import Tensor
# from transformers.models.llama.modeling_llama import LlamaRMSNorm
# from megatron.legacy.model.rms_norm import RMSNorm as LlamaRMSNorm
from flash_attn.ops.rms_norm import RMSNorm as LlamaRMSNorm
from megatron.core import mpu
from megatron.core import tensor_parallel
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.tensor_parallel.utils import VocabUtility
from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy

from galvatron.core import get_args
from galvatron.core.runtime import ModelInfo, mixed_precision_dtype
from galvatron.core.runtime.pipeline import PipeSequential
from galvatron.core.runtime.tensor_parallel import colummn_row_reset_parameters

def get_zigzag_local_tokens_and_cu_seqlens(
    packed_tokens: Tensor, 
    cu_seqlens: Tensor, 
    cp_group: torch.distributed.ProcessGroup, 
    cp_size: int
) -> Tuple[Tensor, Tensor]:
    
    cp_rank = torch.distributed.get_rank(cp_group)
    num_seqs = len(cu_seqlens) - 1 
    device = packed_tokens.device

    local_tokens_list = []
    new_seq_lens = []

    for seq_idx in range(num_seqs):
        seq_start = cu_seqlens[seq_idx].item()
        seq_end = cu_seqlens[seq_idx + 1].item()
        original_seq_tokens = packed_tokens[seq_start:seq_end]
        original_seq_len = original_seq_tokens.shape[0]
        chunk_size = original_seq_len // (2 * cp_size)
        assert original_seq_len == chunk_size * 2 * cp_size, \
            f"original_seq_len {original_seq_len} must be divisible by 2*cp_size ({2*cp_size})"

        first_chunk_start = cp_rank * chunk_size
        first_chunk = original_seq_tokens[first_chunk_start : first_chunk_start + chunk_size]
        second_chunk_idx = 2 * cp_size - 1 - cp_rank
        second_chunk_start = second_chunk_idx * chunk_size
        second_chunk = original_seq_tokens[second_chunk_start : second_chunk_start + chunk_size]  # [chunk_size, ...]

        local_seq_tokens = torch.cat([first_chunk, second_chunk], dim=0)  # [2*chunk_size, ...]
        local_tokens_list.append(local_seq_tokens)
        new_seq_lens.append(local_seq_tokens.shape[0])

    local_tokens = torch.cat(local_tokens_list, dim=0)
    new_cu_seqlens = torch.zeros(num_seqs + 1, dtype=cu_seqlens.dtype, device=device)
    new_cu_seqlens[1:] = torch.tensor(new_seq_lens, dtype=cu_seqlens.dtype, device=device).cumsum(dim=0)

    return local_tokens, new_cu_seqlens  

class LlamaEmbeddings_(nn.Module):
    def __init__(self, model):
        super().__init__()
        model = model.model
        self.embed_tokens = model.embed_tokens
        args = get_args()
        #TODO:需要在这里加上args group的相关操作
        self.sequence_parallel = args.sequence_parallel
        self.clone_scatter_output_in_embedding = args.clone_scatter_output_in_embedding
        self.tp_group = self.embed_tokens.tp_group
        self.sp_group = self.embed_tokens.sp_group
        self.cp_group = self.embed_tokens.cp_group
        self.cp_size = torch.distributed.get_world_size(self.cp_group) if self.cp_group is not None else 1
        self.sp_size = torch.distributed.get_world_size(self.sp_group) if self.sp_group is not None else 1
        self.vocab_sp = args.vocab_sp
        # if self.vocab_sp:
        #     seq_ulysses = int(args.seq_length / self.cp_size)
        #     self.seq_start_index, self.seq_end_index = VocabUtility.vocab_range_from_global_vocab_size(
        #         seq_ulysses,
        #         torch.distributed.get_rank(self.sp_group),
        #         torch.distributed.get_world_size(self.sp_group),
        #     )

    def forward(self, inputs_ids, cu_seqlens = None):
        tokens = inputs_ids.clone()
        args = get_args()
        local_tokens = tokens
        new_cu_seqlens = cu_seqlens
        
        # Step 1: CP split (zigzag ring attention data distribution)
        if self.cp_size > 1:
            local_tokens, new_cu_seqlens = get_zigzag_local_tokens_and_cu_seqlens(
                local_tokens, new_cu_seqlens, self.cp_group, self.cp_size
            )
            local_tokens = local_tokens.contiguous()
            new_cu_seqlens = new_cu_seqlens.contiguous()
        
        # Step 2: SP split (Ulysses sequence parallel data distribution)
        if self.sp_size > 1:
            total_local_seq = new_cu_seqlens[-1] if isinstance(new_cu_seqlens[-1], int) else new_cu_seqlens[-1].item()
            self.seq_start_index, self.seq_end_index = VocabUtility.vocab_range_from_global_vocab_size(
                total_local_seq,
                torch.distributed.get_rank(self.sp_group),
                torch.distributed.get_world_size(self.sp_group),
            )
            local_tokens = local_tokens[self.seq_start_index: self.seq_end_index]
        
            labels = local_tokens.clone()
        hidden_states = self.embed_tokens(local_tokens)
        if args.use_packing:
            hidden_states = hidden_states.unsqueeze(1)
        else:
            hidden_states = hidden_states.transpose(0, 1).contiguous()
        return hidden_states, labels, new_cu_seqlens


class LlamaLayers_(nn.Module):
    def __init__(self, model, layer_idx):
        super().__init__()
        model = model.model
        self.layer = model.layers[layer_idx]
        self.layer_idx = layer_idx
    def forward(self, hidden_states, labels=None, cu_seqlens = None):
        # attention_mask = get_ltor_masks_and_position_ids(input_ids)
        max_seqlen = torch.max(cu_seqlens[1:] - cu_seqlens[:-1]).item() if cu_seqlens is not None else None
        hidden_states = self.layer(hidden_states, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)  # , position_ids = position_ids)
        return hidden_states, labels, cu_seqlens


class LlamaPreNorm_(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, labels=None, cu_seqlens = None):
        hidden_states = self.norm(hidden_states)
        return hidden_states, labels, cu_seqlens


class LlamaLoss_(nn.Module):
    def __init__(self, weight, sequence_parallel, tp_group):
        super().__init__()
        self.weight = nn.Parameter(weight.clone())
        self.sequence_parallel = sequence_parallel
        self.tp_group = tp_group
        world_size = mpu.get_tensor_model_parallel_world_size(tp_group)
        if self.sequence_parallel and world_size <= 1:
            self.sequence_parallel = False
            # disable sp to avoid global buffer

    def forward(self, hidden_states):
        logits_parallel = tensor_parallel.linear_with_grad_accumulation_and_async_allreduce(
            input=hidden_states,
            weight=self.weight,
            bias=None,
            gradient_accumulation_fusion=False,
            allreduce_dgrad=False,
            sequence_parallel=self.sequence_parallel,
            tp_group=self.tp_group,
        )
        return logits_parallel

class LlamaCls_(nn.Module):
    def __init__(self, model, parallel_loss=True, half_entropy=True):
        super().__init__()
        self.sequence_parallel = get_args().sequence_parallel
        self.tp_group = model.lm_head.tp_group
        self.sp_group = model.lm_head.sp_group
        self.cp_group = model.lm_head.cp_group
        self.cp_size = torch.distributed.get_world_size(self.cp_group) if self.cp_group is not None else 1
        self.lm_head = LlamaLoss_(model.lm_head.weight, self.sequence_parallel, self.tp_group)
        self.clone_scatter_output_in_embedding = get_args().clone_scatter_output_in_embedding
        self.parallel_loss = parallel_loss
        self.half_entropy = half_entropy
        args = get_args()
        if args.entropy_in_fp32:
            self.half_entropy = False
        self.seq_length = args.seq_length
        self.vocab_sp = args.vocab_sp
        if self.vocab_sp:
            self.seq_start_index, self.seq_end_index = VocabUtility.vocab_range_from_global_vocab_size(
                self.seq_length // self.cp_size,
                torch.distributed.get_rank(self.sp_group),
                torch.distributed.get_world_size(self.sp_group),
            )

    def forward(self, hidden_states, labels=None, cu_seqlens = None):
        if not self.sequence_parallel:
            hidden_states = copy_to_tensor_model_parallel_region(hidden_states, self.tp_group)

        logits_parallel = self.lm_head(hidden_states)
        
        # For packing (cu_seqlens is not None), use simple cross entropy
        # fused_vocab_parallel_cross_entropy expects 2D tensors but packing produces 1D
        if cu_seqlens is not None or not self.parallel_loss:
            # Simple loss calculation for packing scenario
            if labels is not None:
                # Shift logits and labels for causal LM
                shift_logits = logits_parallel.contiguous()
                shift_labels = labels.contiguous()
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction='mean'
                )
            else:
                loss = torch.tensor(0.0, device=hidden_states.device)
            return loss
        else:
            #TODO: the current implementation is not correct, need to be fixed
            loss = fused_vocab_parallel_cross_entropy(logits_parallel, labels, self.half_entropy, tp_group=self.tp_group)
            loss = loss.transpose(0, 1).contiguous()
            loss = loss.mean()
            return loss


def construct_sequential_model(model, config):
    model_ = PipeSequential()
    model_.add_module("embeddings", LlamaEmbeddings_(model))
    for i in range(config.num_hidden_layers):
        enc = LlamaLayers_(model, i)
        model_.add_module("layer_%d" % i, enc)
    model_.add_module("prenorm", LlamaPreNorm_(model, config))
    model_.add_module("cls", LlamaCls_(model))
    LlamaLoss_.reset_parameters = colummn_row_reset_parameters
    return model_


class LlamaModelInfo(ModelInfo):
    def __init__(self, config, args):
        super(LlamaModelInfo, self).__init__()
        layernum_list = [config.num_hidden_layers]
        seq_len, hidden_size = config.max_position_embeddings, config.hidden_size
        mixed_precision = mixed_precision_dtype(args.mixed_precision)
        if args.shape_order == "SBH":
            layer_shapes_list = [[[seq_len, -1, hidden_size]]]
        else:
            layer_shapes_list = [[[-1, seq_len, hidden_size]]]
        layer_dtypes_list = [[mixed_precision]]
        module_types = ["embed"] + ["gpt_dec"] * config.num_hidden_layers + ["norm", "cls"]
        self.set_layernums(layernum_list)
        self.set_shapes(layer_shapes_list)
        self.set_dtypes(layer_dtypes_list)
        self.set_module_types(module_types)
