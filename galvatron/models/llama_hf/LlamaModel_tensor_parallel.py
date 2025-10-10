import torch
from flash_attn.ops.rms_norm import RMSNorm as LlamaRMSNorm
# from transformers.models.llama.modeling_llama import LlamaRMSNorm
# from megatron.legacy.model.rms_norm import RMSNorm as LlamaRMSNorm
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding
from galvatron.core.runtime.tensor_parallel.mlp import MLP, MLPSubmodules
from galvatron.core.runtime.tensor_parallel.attention import SelfAttention, SelfAttentionSubmodules
from galvatron.core.runtime.tensor_parallel.attention_impl import DotProductAttention, FlashSelfOrCrossAttention, DistributedAttention, ZigzagRingFlashAttention
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

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.layer_idx = layer_number
        self.LayerNorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_pos_emb = RotaryEmbedding(
            self.head_dim, args.rotary_percent, seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor, 
            rotary_base=args.rotary_base,
            cp_group=self.cp_group, sp_group=self.sp_group
        )

    def forward(self, hidden_states, attention_mask, rotary_embedding):
        input_tensor = hidden_states
        hidden_states = self.LayerNorm(hidden_states)
        if self.sequence_parallel:
            if self.use_ulysses:
                if self.use_zigzag_cp:
                    # max_seq_len = hidden_states.shape[0] * self.cp_size * self.sp_size
                    # no offset for zigzag cp, because the offset is already included in the Megatron RotaryEmbedding
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
        rotary_embedding=None,
    ):
        attention_output = self.attention(
            hidden_states,
            attention_mask,
            rotary_embedding,
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
