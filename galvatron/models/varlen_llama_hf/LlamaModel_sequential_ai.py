import torch
import torch.nn as nn
import torch.distributed as dist
from dataclasses import dataclass
from typing import Optional

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


@dataclass
class AdaCPSPConfig:
    """Configuration for current micro-batch's AdaCPSP strategy."""
    use_adacpsp: bool = False
    use_ulysses: bool = False
    use_zigzag_cp: bool = False
    ulysses_size: int = 1
    cp_size: int = 1
    sp_group: Optional[dist.ProcessGroup] = None
    cp_group: Optional[dist.ProcessGroup] = None


def extract_local_zigzag(value, cu_seqlens, rank, world_size):
    """Extract local data for zigzag CP.
    
    For each short sequence in the packed sequence, perform zigzag split internally,
    then concatenate the results.
    
    Args:
        value: Tensor of shape (total_seq_len,) or (total_seq_len, ...)
        cu_seqlens: Cumulative sequence lengths tensor
        rank: Current rank in CP group
        world_size: CP group size
    
    Returns:
        Local tensor after zigzag split
    """
    local_values = []
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
        seq_data = value[start:end]
        # Split into 2*world_size chunks for zigzag
        chunks = seq_data.chunk(2 * world_size, dim=0)
        # Zigzag pattern: take chunk[rank] and chunk[2*world_size - 1 - rank]
        local_values.extend([
            chunks[rank],
            chunks[2 * world_size - 1 - rank],
        ])
    return torch.cat(local_values, dim=0).contiguous()


def get_local_cu_seqlens_zigzag(cu_seqlens, world_size):
    """Compute local cu_seqlens after zigzag split.
    
    After zigzag split, each sequence is divided by world_size.
    """
    local_cu_seqlens = [0]
    for i in range(len(cu_seqlens) - 1):
        seq_len = (cu_seqlens[i + 1] - cu_seqlens[i]).item()
        local_seq_len = seq_len // world_size  # Each rank gets 1/world_size
        local_cu_seqlens.append(local_cu_seqlens[-1] + local_seq_len)
    return torch.tensor(local_cu_seqlens, dtype=cu_seqlens.dtype, device=cu_seqlens.device)


def get_local_cu_seqlens_ulysses(cu_seqlens, world_size):
    """Compute local cu_seqlens after Ulysses split.
    
    Ulysses splits the entire packed sequence by length.
    """
    return cu_seqlens // world_size

class LlamaEmbeddings_(nn.Module):
    def __init__(self, model):
        super().__init__()
        model = model.model
        self.embed_tokens = model.embed_tokens
        args = get_args()
        self.sequence_parallel = args.sequence_parallel
        self.clone_scatter_output_in_embedding = args.clone_scatter_output_in_embedding
        self.tp_group = self.embed_tokens.tp_group
        self.sp_group = self.embed_tokens.sp_group
        self.cp_group = self.embed_tokens.cp_group
        self.cp_size = torch.distributed.get_world_size(self.cp_group) if self.cp_group is not None else 1
        self.sp_size = torch.distributed.get_world_size(self.sp_group) if self.sp_group is not None else 1
        self.vocab_sp = args.vocab_sp
        self.use_adacpsp = getattr(args, 'use_adacpsp', False)
        if self.vocab_sp:
            seq_ulysses = int(args.seq_length / self.cp_size)
            self.seq_start_index, self.seq_end_index = VocabUtility.vocab_range_from_global_vocab_size(
                seq_ulysses,
                torch.distributed.get_rank(self.sp_group),
                torch.distributed.get_world_size(self.sp_group),
            )

    def forward(self, tokens, position_ids=None, attention_mask=None, labels=None, 
                cu_seqlens=None, max_seqlen=None, adacpsp_config=None):
        """
        Forward pass for embeddings with AdaCPSP support.
        
        Args:
            tokens: Input token ids. Shape: (batch, seq) or (total_seq,) for varlen
            labels: Labels for loss computation
            cu_seqlens: Cumulative sequence lengths for varlen mode
            max_seqlen: Maximum sequence length for varlen mode
            adacpsp_config: AdaCPSP configuration for dynamic strategy
        
        Returns:
            For non-varlen: hidden_states
            For varlen: (hidden_states, labels, cu_seqlens, max_seqlen, adacpsp_config)
        """
        args = get_args()
        labels = tokens.clone() if labels is None else labels.clone()
        # Get AdaCPSP config from args if not provided
        if self.use_adacpsp and adacpsp_config is None:
            adacpsp_config = getattr(args, 'current_adacpsp_config', None)
        
        # Varlen (packing) mode
        if cu_seqlens is not None:
            cu_seqlens = cu_seqlens.cuda() if not cu_seqlens.is_cuda else cu_seqlens
            
            # Determine strategy
            if adacpsp_config is not None and adacpsp_config.use_adacpsp:
                use_ulysses = adacpsp_config.use_ulysses
                use_zigzag_cp = adacpsp_config.use_zigzag_cp
                sp_group = adacpsp_config.sp_group
                cp_group = adacpsp_config.cp_group
                ulysses_size = adacpsp_config.ulysses_size
                cp_size = adacpsp_config.cp_size
            else:
                use_ulysses = self.sp_size > 1
                use_zigzag_cp = self.cp_size > 1
                sp_group = self.sp_group
                cp_group = self.cp_group
                ulysses_size = self.sp_size
                cp_size = self.cp_size
            
            # Data splitting based on strategy
            if use_zigzag_cp and cp_size > 1:
                # Zigzag CP: split each short sequence internally with zigzag pattern
                cp_rank = torch.distributed.get_rank(cp_group)
                tokens = extract_local_zigzag(tokens, cu_seqlens, cp_rank, cp_size)
                if labels is not None:
                    labels = extract_local_zigzag(labels, cu_seqlens, cp_rank, cp_size)
                cu_seqlens = get_local_cu_seqlens_zigzag(cu_seqlens, cp_size)
                max_seqlen = max_seqlen // cp_size
                
            elif use_ulysses and ulysses_size > 1:
                # Ulysses SP: split by total length
                sp_rank = torch.distributed.get_rank(sp_group)
                total_len = tokens.shape[0]
                local_len = total_len // ulysses_size
                start_idx = sp_rank * local_len
                end_idx = start_idx + local_len
                tokens = tokens[start_idx:end_idx].contiguous()
                if labels is not None:
                    labels = labels[start_idx:end_idx].contiguous()
                cu_seqlens = get_local_cu_seqlens_ulysses(cu_seqlens, ulysses_size)
                max_seqlen = max_seqlen // ulysses_size
            
            # Embedding for varlen: tokens shape is (total_seq,)
            hidden_states = self.embed_tokens(tokens)
            # Output shape: (total_seq, hidden_size) -> (total_seq, 1, hidden_size) for compatibility
            hidden_states = hidden_states.unsqueeze(1)
            
            return hidden_states, labels, cu_seqlens, max_seqlen, adacpsp_config
        
        # Non-varlen mode (original logic)
        if self.vocab_sp:
            tokens = tokens[:, self.seq_start_index : self.seq_end_index].contiguous()
        
        # [b, s] -> [s /cp / tp, b, h]
        hidden_states = self.embed_tokens(tokens)
        return hidden_states


class LlamaLayers_(nn.Module):
    def __init__(self, model, layer_idx):
        super().__init__()
        model = model.model
        self.layer = model.layers[layer_idx]
        self.layer_idx = layer_idx

    def forward(self, hidden_states, labels=None, cu_seqlens=None, max_seqlen=None, 
                adacpsp_config=None, attention_mask=None, rotary_embedding=None):
        """
        Forward pass for transformer layers with AdaCPSP support.
        
        For varlen mode, receives tuple from embeddings and passes through.
        For non-varlen mode, uses original signature.
        """
        # Check if this is varlen mode (tuple input from embeddings)
        if isinstance(hidden_states, tuple):
            hidden_states, labels, cu_seqlens, max_seqlen, adacpsp_config = hidden_states
        
        # Call layer with appropriate arguments
        if cu_seqlens is not None:
            # Varlen mode
            hidden_states = self.layer(
                hidden_states, 
                attention_mask=None,  # varlen doesn't need attention_mask
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                adacpsp_config=adacpsp_config,
                rotary_embedding=rotary_embedding
            )
            return hidden_states, labels, cu_seqlens, max_seqlen, adacpsp_config
        else:
            # Non-varlen mode (original logic)
            hidden_states = self.layer(hidden_states, attention_mask=attention_mask, rotary_embedding=rotary_embedding)
            return hidden_states


class LlamaPreNorm_(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, labels=None, cu_seqlens=None, max_seqlen=None,
                adacpsp_config=None, attention_mask=None, rotary_embedding=None):
        """Forward pass for pre-normalization with AdaCPSP support."""
        # Check if this is varlen mode (tuple input)
        if isinstance(hidden_states, tuple):
            hidden_states, labels, cu_seqlens, max_seqlen, adacpsp_config = hidden_states
        
        hidden_states = self.norm(hidden_states)
        
        if cu_seqlens is not None:
            return hidden_states, labels, cu_seqlens, max_seqlen, adacpsp_config
        return hidden_states


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
        args = get_args()
        self.sequence_parallel = args.sequence_parallel
        self.tp_group = model.lm_head.tp_group
        self.sp_group = model.lm_head.sp_group
        self.cp_group = model.lm_head.cp_group
        self.cp_size = torch.distributed.get_world_size(self.cp_group) if self.cp_group is not None else 1
        self.sp_size = torch.distributed.get_world_size(self.sp_group) if self.sp_group is not None else 1
        self.lm_head = LlamaLoss_(model.lm_head.weight, self.sequence_parallel, self.tp_group)
        self.clone_scatter_output_in_embedding = args.clone_scatter_output_in_embedding
        self.parallel_loss = parallel_loss
        self.half_entropy = half_entropy
        self.use_adacpsp = getattr(args, 'use_adacpsp', False)
        self.global_train_batch_size = args.global_train_batch_size
        # seq_data_group is used for gradient balancing across data parallel ranks
        self.seq_data_group = getattr(model.lm_head, 'seq_data_group', None)
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

    def forward(self, hidden_states, labels=None, cu_seqlens=None, max_seqlen=None,
                adacpsp_config=None, attention_mask=None, rotary_embedding=None):
        """
        Forward pass for loss computation with AdaCPSP support.
        
        Args:
            hidden_states: Hidden states from the model
            labels: Ground truth labels
            cu_seqlens: Cumulative sequence lengths for varlen mode
            max_seqlen: Maximum sequence length for varlen mode
            adacpsp_config: AdaCPSP configuration for dynamic strategy
        """
        # Check if this is varlen mode (tuple input)
        if isinstance(hidden_states, tuple):
            hidden_states, labels, cu_seqlens, max_seqlen, adacpsp_config = hidden_states
        
        # Varlen mode
        if cu_seqlens is not None:
            return self._forward_varlen(hidden_states, labels, cu_seqlens, max_seqlen, adacpsp_config)
        
        # Non-varlen mode (original logic)
        if self.vocab_sp:
            labels = labels[:, self.seq_start_index : self.seq_end_index].contiguous()
        if not self.sequence_parallel:
            hidden_states = copy_to_tensor_model_parallel_region(hidden_states, self.tp_group)

        logits_parallel = self.lm_head(hidden_states)

        # [b s] -> [s b]
        labels = labels.transpose(0, 1).contiguous()

        # loss = tensor_parallel.vocab_parallel_cross_entropy(output.float(), input_ids)
        if not self.parallel_loss:
            output = gather_from_tensor_model_parallel_region(logits_parallel, self.tp_group)
            if not self.half_entropy:
                logits = output.float()
            else:
                logits = output
            loss = None
            # Shift so that tokens < n predict n
            shift_logits = logits.contiguous()  # logits[:-1, ..., :].contiguous()
            shift_labels = labels.contiguous()  # input_ids[1:, ...].contiguous()
            # Flatten the tokens
            from torch.nn import CrossEntropyLoss

            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        else:
            loss = fused_vocab_parallel_cross_entropy(logits_parallel, labels, self.half_entropy, tp_group=self.tp_group)
            # loss = tensor_parallel.vocab_parallel_cross_entropy(logits_parallel, labels, self.half_entropy, tp_group=self.tp_group)
            if self.vocab_sp:
                loss = gather_from_tensor_model_parallel_region(loss, self.sp_group)
            # loss = loss.mean()
        loss = loss.transpose(0, 1).contiguous()
        return loss
    
    def _forward_varlen(self, hidden_states, labels, cu_seqlens, max_seqlen, adacpsp_config):
        """Forward pass for varlen mode.
        
        - Compute loss on each rank's local data
        - Use vocab_sequence_parallel_cross_entropy for SP/CP
        - Apply gradient balancing at the end
        """
        from torch.nn import CrossEntropyLoss
        
        # Determine strategy
        if adacpsp_config is not None and adacpsp_config.use_adacpsp:
            use_ulysses = adacpsp_config.use_ulysses
            use_zigzag_cp = adacpsp_config.use_zigzag_cp
            sp_group = adacpsp_config.sp_group
            cp_group = adacpsp_config.cp_group
            ulysses_size = adacpsp_config.ulysses_size
            cp_size = adacpsp_config.cp_size
        else:
            use_ulysses = self.sp_size > 1
            use_zigzag_cp = self.cp_size > 1
            sp_group = self.sp_group
            cp_group = self.cp_group
            ulysses_size = self.sp_size
            cp_size = self.cp_size
        
        # Determine the parallel group for loss computation
        if use_ulysses and ulysses_size > 1:
            parallel_group = sp_group
            parallel_size = ulysses_size
        elif use_zigzag_cp and cp_size > 1:
            parallel_group = cp_group
            parallel_size = cp_size
        else:
            parallel_group = None
            parallel_size = 1
        
        ds_sequence_parallel = parallel_size > 1
        
        # Number of sequences in this batch (for packing)
        local_bsz = len(cu_seqlens) - 1
        
        # hidden_states shape: (total_seq, 1, hidden_size) -> (total_seq, hidden_size)
        if hidden_states.dim() == 3 and hidden_states.shape[1] == 1:
            hidden_states = hidden_states.squeeze(1)
        
        if not self.sequence_parallel:
            hidden_states = copy_to_tensor_model_parallel_region(hidden_states, self.tp_group)
        
        # Compute logits
        logits_parallel = self.lm_head(hidden_states)
        
        # labels shape: (total_seq,) -> (total_seq, 1) for compatibility with cross entropy
        labels = labels.unsqueeze(1)
        
        if not self.parallel_loss:
            # Gather logits and compute standard loss
            output = gather_from_tensor_model_parallel_region(logits_parallel, self.tp_group)
            if not self.half_entropy:
                logits = output.float()
            else:
                logits = output
            
            if ds_sequence_parallel:
                # Use sequence parallel cross entropy
                from megatron.core import sequence_parallel as sp_module
                loss = sp_module.vocab_sequence_parallel_cross_entropy(logits, labels, parallel_group)
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                loss = loss.unsqueeze(0)  # Make it consistent shape
        else:
            # Parallel loss computation
            if ds_sequence_parallel:
                # Use vocab_sequence_parallel_cross_entropy for both Ulysses SP and Zigzag CP
                # This computes loss on each rank's local data without gathering
                from megatron.core import sequence_parallel as sp_module
                if not self.half_entropy:
                    loss = sp_module.vocab_sequence_parallel_cross_entropy(
                        logits_parallel.float(), labels, parallel_group
                    )
                else:
                    loss = sp_module.vocab_sequence_parallel_cross_entropy(
                        logits_parallel, labels, parallel_group
                    )
            else:
                # No parallelism: standard vocab parallel cross entropy
                if not self.half_entropy:
                    loss = tensor_parallel.vocab_parallel_cross_entropy(
                        logits_parallel.float(), labels, tp_group=self.tp_group
                    )
                else:
                    loss = tensor_parallel.vocab_parallel_cross_entropy(
                        logits_parallel, labels, tp_group=self.tp_group
                    )
        
        # Mean over sequence
        loss = loss.mean()
        
        # For packing, divide by number of sequences
        loss = loss / local_bsz
        
        # Gradient balancing: scale loss by local_bsz / global_bsz
        # This ensures gradients are properly balanced across all ranks
        if self.seq_data_group is not None:
            seq_data_world_size = torch.distributed.get_world_size(self.seq_data_group)
        else:
            seq_data_world_size = 1
        
        loss = (loss * local_bsz * seq_data_world_size) / self.global_train_batch_size
        
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
