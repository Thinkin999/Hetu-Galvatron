import torch
from flash_attn.ops.rms_norm import RMSNorm as LlamaRMSNorm
# from transformers.models.llama.modeling_llama import LlamaRMSNorm
# from megatron.legacy.model.rms_norm import RMSNorm as LlamaRMSNorm
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding
from galvatron.core.runtime.tensor_parallel.mlp import MLP, MLPSubmodules
from galvatron.core.runtime.tensor_parallel.attention import SelfAttention, SelfAttentionSubmodules
from galvatron.core.runtime.tensor_parallel.attention_impl import (
    DotProductAttention, 
    FlashSelfOrCrossAttention, 
    FlashSelfAttentionVarlen,
    DistributedAttention,
    ZigzagRingFlashAttention,
    zigzag_ring_flash_attn_varlen_func,
)
from megatron.training.arguments import core_transformer_config_from_args
from torch import nn

from galvatron.core import get_args

class LlamaAttention_tp(nn.Module):
    def __init__(self, config, layer_number, tp_group=None, sp_group=None, cp_group=None):
        super().__init__()
        args = get_args()
        self.sequence_parallel = args.sequence_parallel
        self.use_ulysses = sp_group.size > 1
        self.use_zigzag_cp = cp_group.size > 1
        self.sp_size = sp_group.size if sp_group is not None else 1
        self.cp_size = cp_group.size if cp_group is not None else 1
        self.tp_size = tp_group.size if tp_group is not None else 1
        self.use_adacpsp = getattr(args, 'use_adacpsp', False)
        megatron_config = core_transformer_config_from_args(args)
        self.tp_group = tp_group.group if tp_group is not None else None
        self.sp_group = sp_group.group if sp_group is not None else None
        self.cp_group = cp_group.group if cp_group is not None else None
        self.cp_ranks = cp_group.ranks if cp_group is not None else None

        self.attention = SelfAttention(
            megatron_config,
            SelfAttentionSubmodules(
                linear_qkv=ColumnParallelLinear,
                core_attention=DotProductAttention,
                flash_attention=FlashSelfOrCrossAttention,
                dist_attention=DistributedAttention,
                zigzag_ring_flash_attn=ZigzagRingFlashAttention,
                linear_proj=RowParallelLinear,
            ),
            layer_number,
            attn_mask_type=AttnMaskType.causal,
            tp_group=self.tp_group,
            sp_group=self.sp_group,
            cp_group=self.cp_group,
            cp_ranks=self.cp_ranks,
        )
        
        # Varlen attention components for AdaCPSP
        self.flash_attention_varlen = FlashSelfAttentionVarlen(
            causal=True,
            attention_dropout=config.attention_dropout,
        )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.layer_idx = layer_number
        self.LayerNorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_pos_emb = RotaryEmbedding(
            self.head_dim, args.rotary_percent, seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor, 
            rotary_base=args.rotary_base,
            cp_group=self.cp_group, sp_group=self.sp_group
        )

    def forward(self, hidden_states, attention_mask=None, cu_seqlens=None, max_seqlen=None,
                adacpsp_config=None, rotary_embedding=None):
        """
        Forward pass for LlamaAttention with AdaCPSP support.
        
        Args:
            hidden_states: Input hidden states
            attention_mask: Attention mask (for non-varlen mode)
            cu_seqlens: Cumulative sequence lengths (for varlen mode)
            max_seqlen: Maximum sequence length (for varlen mode)
            adacpsp_config: AdaCPSP configuration for dynamic strategy
            rotary_embedding: Pre-computed rotary embeddings
        """
        input_tensor = hidden_states
        hidden_states = self.LayerNorm(hidden_states)
        
        # Varlen mode
        if cu_seqlens is not None:
            return self._forward_varlen(
                hidden_states, input_tensor, cu_seqlens, max_seqlen, adacpsp_config
            )
        
        # Non-varlen mode (original logic)
        if self.sequence_parallel:
            if self.use_ulysses:
                if self.use_zigzag_cp:
                    rotary_pos_emb = self.rotary_pos_emb(
                        hidden_states.shape[0] * self.cp_size * self.sp_size)
                else:
                    rotary_pos_emb = self.rotary_pos_emb(
                        hidden_states.shape[0] , offset=hidden_states.shape[0] * torch.distributed.get_rank(self.sp_group))
            else:
                if self.use_zigzag_cp:
                    rotary_pos_emb = self.rotary_pos_emb(
                        hidden_states.shape[0] * torch.distributed.get_world_size(self.tp_group) * self.cp_size)
                elif rotary_embedding is not None:
                    rotary_pos_emb = rotary_embedding
                else:
                    rotary_pos_emb = self.rotary_pos_emb(
                        hidden_states.shape[0] * self.tp_size
                    )
        else:
            if rotary_embedding is not None:
                rotary_pos_emb = rotary_embedding
            elif self.use_zigzag_cp:
                rotary_pos_emb = self.rotary_pos_emb(hidden_states.shape[0] * self.cp_size)
            else:
                rotary_pos_emb = self.rotary_pos_emb(hidden_states.shape[0])
        hidden_states, bias = self.attention(hidden_states, attention_mask, rotary_pos_emb=rotary_pos_emb)
        hidden_states = hidden_states + input_tensor
        return hidden_states
    
    def _forward_varlen(self, hidden_states, input_tensor, cu_seqlens, max_seqlen, adacpsp_config):
        """Forward pass for varlen mode with AdaCPSP support."""
        from megatron.core.models.common.embeddings.rope_utils import _apply_rotary_pos_emb_thd
        
        # Determine strategy
        if adacpsp_config is not None and adacpsp_config.use_adacpsp:
            use_ulysses = adacpsp_config.use_ulysses
            use_zigzag_cp = adacpsp_config.use_zigzag_cp
            sp_group = adacpsp_config.sp_group
            cp_group = adacpsp_config.cp_group
            ulysses_size = adacpsp_config.ulysses_size
            cp_size = adacpsp_config.cp_size
        else:
            use_ulysses = self.use_ulysses
            use_zigzag_cp = self.use_zigzag_cp
            sp_group = self.sp_group
            cp_group = self.cp_group
            ulysses_size = self.sp_size
            cp_size = self.cp_size
        
        # hidden_states shape: (total_seq, 1, hidden_size) -> (total_seq, hidden_size)
        if hidden_states.dim() == 3 and hidden_states.shape[1] == 1:
            hidden_states = hidden_states.squeeze(1)
        
        # Compute Q, K, V using the linear_qkv from attention
        qkv = self.attention.linear_qkv(hidden_states)[0]  # (total_seq, (q+k+v)*head_dim)
        
        # Split into Q, K, V
        # Shape: (total_seq, num_heads, head_dim) for Q
        # Shape: (total_seq, num_kv_heads, head_dim) for K, V
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        
        query = qkv[:, :q_size].view(-1, self.num_heads, self.head_dim)
        key = qkv[:, q_size:q_size + kv_size].view(-1, self.num_kv_heads, self.head_dim)
        value = qkv[:, q_size + kv_size:].view(-1, self.num_kv_heads, self.head_dim)
        
        # Compute and apply rotary embeddings for varlen
        # For varlen, generate position_ids for each short sequence within the pack
        query, key = self._apply_varlen_rotary_emb(
            query, key, cu_seqlens, max_seqlen, 
            use_ulysses, use_zigzag_cp, sp_group, cp_group, ulysses_size, cp_size
        )
        
        # Select attention implementation based on strategy
        if use_ulysses and ulysses_size > 1:
            # Ulysses SP with varlen: use DistributedAttention wrapping FlashSelfAttentionVarlen
            # For Ulysses, need to expand KV heads to sp_world_size for AlltoAll
            if self.num_kv_heads < ulysses_size:
                key_expanded = key.repeat_interleave(ulysses_size // self.num_kv_heads, dim=1)
                value_expanded = value.repeat_interleave(ulysses_size // self.num_kv_heads, dim=1)
            else:
                key_expanded = key
                value_expanded = value
            
            # Reshape for DistributedAttention: (total_seq, heads, dim) -> (total_seq, 1, heads, dim)
            # Then use batch_dim_idx=1
            q_for_dist = query.unsqueeze(1)  # (total_seq, 1, heads, dim)
            k_for_dist = key_expanded.unsqueeze(1)
            v_for_dist = value_expanded.unsqueeze(1)
            
            # Create DistributedAttention wrapping FlashSelfAttentionVarlen
            dist_attn = DistributedAttention(
                local_attention=self.flash_attention_varlen,
                sequence_process_group=sp_group,
                scatter_idx=2,  # scatter on head dim
                gather_idx=0,   # gather on seq dim
            )
            # Call with batch_dim_idx=1, pass cu_seqlens and max_seqlen as kwargs
            context = dist_attn(q_for_dist, k_for_dist, v_for_dist, batch_dim_idx=1,
                               cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            context = context.squeeze(1)  # (total_seq, heads, dim)
            
        elif use_zigzag_cp and cp_size > 1:
            # Zigzag CP with varlen: flash attention can handle GQA internally
            context = zigzag_ring_flash_attn_varlen_func(
                query, key, value,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                dropout_p=self.attention_dropout if self.training else 0.0,
                causal=True,
                group=cp_group,
            )
        else:
            # Standard flash attention varlen: flash attention can handle GQA internally
            context = self.flash_attention_varlen(
                query, key, value,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen
            )
        
        # Reshape context: (total_seq, num_heads, head_dim) -> (total_seq, hidden_size)
        context = context.view(context.shape[0], -1)
        
        # Output projection
        output, bias = self.attention.linear_proj(context)
        
        # Add residual and reshape back
        output = output + input_tensor.squeeze(1) if input_tensor.dim() == 3 else output + input_tensor
        output = output.unsqueeze(1)  # (total_seq, 1, hidden_size)
        
        return output
    
    def _apply_varlen_rotary_emb(self, query, key, cu_seqlens, max_seqlen,
                                  use_ulysses, use_zigzag_cp, sp_group, cp_group, 
                                  ulysses_size, cp_size):
        """Apply rotary embeddings for varlen sequences.
        
        For varlen (packed sequences), each short sequence has its own position ids starting from 0.
        Similar to: position_ids = torch.cat([torch.arange(cu_seqlens[i+1] - cu_seqlens[i]) for i in range(num_seqs)])
        """
        from megatron.core.models.common.embeddings.rope_utils import _rotate_half
        
        num_seqs = len(cu_seqlens) - 1
        total_seq_len = query.shape[0]
        
        # Generate position_ids for each short sequence
        # Each sequence starts from position 0
        position_ids_list = []
        for i in range(num_seqs):
            seq_len = (cu_seqlens[i + 1] - cu_seqlens[i]).item()
            position_ids_list.append(torch.arange(seq_len, device=query.device))
        position_ids = torch.cat(position_ids_list, dim=0)  # (total_seq,)
        
        # For zigzag CP, the position_ids need to follow zigzag pattern
        # The data has been zigzag-split, so we need zigzag position embeddings
        if use_zigzag_cp and cp_size > 1:
            # After zigzag split, each rank has positions [rank, 2*cp_size-1-rank, ...] for each sequence
            # We need to generate the correct position_ids based on zigzag pattern
            position_ids = self._get_zigzag_position_ids(cu_seqlens, cp_group, cp_size)
        
        # Get rotary embeddings for all positions
        # inv_freq shape: (dim/2,)
        inv_freq = self.rotary_pos_emb.inv_freq
        
        # position_ids: (total_seq,) -> (total_seq, 1)
        # freqs: (total_seq, dim/2)
        freqs = torch.outer(position_ids.float(), inv_freq)
        
        # emb: (total_seq, dim)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # cos, sin: (total_seq, 1, 1, dim)
        cos = emb.cos().unsqueeze(1).unsqueeze(1)
        sin = emb.sin().unsqueeze(1).unsqueeze(1)
        
        # Apply rotary embeddings
        # query, key: (total_seq, heads, dim) -> (total_seq, 1, heads, dim)
        # query = query.unsqueeze(1)
        # key = key.unsqueeze(1)
        
        # Apply rotation
        query_rot = (query * cos) + (_rotate_half(query) * sin)
        key_rot = (key * cos) + (_rotate_half(key) * sin)
        
        return query_rot.squeeze(1), key_rot.squeeze(1)
    
    def _get_zigzag_position_ids(self, cu_seqlens, cp_group, cp_size):
        """Generate position ids for zigzag CP pattern.
        
        After zigzag split, each rank has [chunk_rank, chunk_(2*cp_size-1-rank)] for each sequence.
        The position ids should reflect the original positions.
        """
        cp_rank = torch.distributed.get_rank(cp_group)
        num_seqs = len(cu_seqlens) - 1
        
        position_ids_list = []
        for i in range(num_seqs):
            local_seq_len = (cu_seqlens[i + 1] - cu_seqlens[i]).item()
            # Original sequence length before zigzag split
            original_seq_len = local_seq_len * cp_size
            chunk_size = original_seq_len // (2 * cp_size)
            
            # First half: positions from chunk_rank
            first_half_start = cp_rank * chunk_size
            first_half_positions = torch.arange(first_half_start, first_half_start + chunk_size, 
                                                 device=cu_seqlens.device)
            
            # Second half: positions from chunk_(2*cp_size-1-rank)
            second_half_idx = 2 * cp_size - 1 - cp_rank
            second_half_start = second_half_idx * chunk_size
            second_half_positions = torch.arange(second_half_start, second_half_start + chunk_size,
                                                  device=cu_seqlens.device)
            
            position_ids_list.extend([first_half_positions, second_half_positions])
        
        return torch.cat(position_ids_list, dim=0)

class LlamaMLP_tp(nn.Module):
    def __init__(self, config, tp_group=None):
        super().__init__()
        megatron_config = core_transformer_config_from_args(get_args())
        self.tp_group = tp_group.group if tp_group is not None else None
        self.mlp = MLP(
            megatron_config, 
            MLPSubmodules(
                linear_fc1=ColumnParallelLinear, 
                linear_fc2=RowParallelLinear), 
            tp_group=self.tp_group)
        self.LayerNorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None, rotary_embedding=None): # Adding attention_mask and rotary_embedding ensures input consistency when profiling the MLP layer independently
        input_tensor = hidden_states
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states, bias = self.mlp(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states

class LlamaLayer_tp(nn.Module):
    def __init__(self, config, layer_number, tp_group=None, sp_group=None, cp_group=None):
        super().__init__()
        self.attention = LlamaAttention_tp(config, layer_number, tp_group, sp_group, cp_group)
        self.mlp = LlamaMLP_tp(config, tp_group)
        self.idx = layer_number

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        cu_seqlens=None,
        max_seqlen=None,
        adacpsp_config=None,
        rotary_embedding=None,
    ):
        """
        Forward pass for LlamaLayer with AdaCPSP support.
        
        Args:
            hidden_states: Input hidden states
            attention_mask: Attention mask (for non-varlen mode)
            cu_seqlens: Cumulative sequence lengths (for varlen mode)
            max_seqlen: Maximum sequence length (for varlen mode)
            adacpsp_config: AdaCPSP configuration for dynamic strategy
            rotary_embedding: Pre-computed rotary embeddings
        """
        attention_output = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            adacpsp_config=adacpsp_config,
            rotary_embedding=rotary_embedding,
        )
        layer_output = self.mlp(attention_output)

        return layer_output


def construct_tensor_parallel_model(model, config, tp_groups_enc, sp_groups_enc, cp_groups_enc):
    args = get_args()
    if hasattr(args, "profile_unit") and args.profile_unit == "attention":
        layers_tp = nn.ModuleList(
            [
                LlamaAttention_tp(config, i, tp_group=tp_groups_enc[i + 1], sp_group=sp_groups_enc[i + 1], cp_group=cp_groups_enc[i + 1])
                for i in range(config.num_hidden_layers)
            ]
        )
    elif hasattr(args, "profile_unit") and args.profile_unit == "mlp":
        layers_tp = nn.ModuleList(
            [
                LlamaMLP_tp(config, tp_group=tp_groups_enc[i + 1])
                for i in range(config.num_hidden_layers)
            ]
        )
    else:
        layers_tp = nn.ModuleList(
            [
                LlamaLayer_tp(config, i, tp_group=tp_groups_enc[i + 1], sp_group=sp_groups_enc[i + 1], cp_group=cp_groups_enc[i + 1])
                for i in range(config.num_hidden_layers)
            ]
        )
    setattr(model.model, "layers", layers_tp)
    args = get_args()
    megatron_config = core_transformer_config_from_args(get_args())
    setattr(
        model.model,
        "embed_tokens",
        VocabParallelEmbedding(
            args.padded_vocab_size,
            megatron_config.hidden_size,
            config=megatron_config,
            init_method=megatron_config.init_method,
            reduce_scatter_embeddings=args.sequence_parallel,
            tp_group=tp_groups_enc[0].group,
            sp_group=sp_groups_enc[0].group,
            cp_group=cp_groups_enc[0].group
        ),
    )
    setattr(
        model,
        "lm_head",
        ColumnParallelLinear(
            megatron_config.hidden_size,
            args.padded_vocab_size,
            config=megatron_config,
            init_method=megatron_config.init_method,
            bias=False,
            tp_group=tp_groups_enc[-1].group,
            sp_group=sp_groups_enc[-1].group,
            cp_group=cp_groups_enc[-1].group,
        ),
    )

    return model
