"""
AdaCPSP Communication Group Manager

Follows FlexSP's `convert_microbatch_res` pattern:
  - Each microbatch contains MULTIPLE groups of potentially different sizes
  - Groups are mapped to CONSECUTIVE GPU ranks
  - Each rank belongs to exactly ONE group per microbatch  
  - Different groups can use different attn_types (Ulysses / Ring)

Example: 8 GPUs, microbatch = [("ulysses", 4, [s0,s1,s2]), ("ring", 4, [s3,s4,s5])]
  → Rank 0-3: Ulysses sp_size=4, processing sequences [s0,s1,s2]
  → Rank 4-7: Ring cp_size=4, processing sequences [s3,s4,s5]

Design:
  - tp_deg = 1 (no tensor parallelism, full weights on every GPU)
  - FSDP covers all GPUs (dp = world_size at construction)
  - sp_size and cp_size are BOTH dynamic, determined by solver
  - Groups are created lazily and cached (like FlexSP's global_group_set)
"""

import torch
import torch.distributed as dist
from typing import Dict, Tuple, Optional, List

from galvatron.core.runtime.tensor_parallel.attention import SelfAttention
from galvatron.core.runtime.tensor_parallel.attention_impl import (
    DistributedAttention,
    ZigzagRingFlashAttentionVarlen,
    FlashSelfAttentionVarlen,
)


# Global group cache (all ranks must participate in new_group creation)
_global_group_set: List[Tuple[int, ...]] = []  # list of rank tuples that have been created
_group_pool: Dict[Tuple[int, ...], dist.ProcessGroup] = {}  # rank_tuple -> ProcessGroup


def convert_microbatch_res(micro_res):
    """
    Convert solver's microbatch result to per-rank assignment.
    
    Follows FlexSP's convert_microbatch_res pattern:
    - Groups are laid out on consecutive GPU ranks
    - Each rank finds which group it belongs to
    - Creates communication groups lazily (all ranks MUST call new_group together)
    
    If the total parallel_sizes don't cover all N GPUs, remaining ranks are
    assigned to a single-rank dummy group (sp=1, cp=1, no sequences).
    
    Args:
        micro_res: List of (attn_type, parallel_size, [seq_id_list])
            e.g. [("ulysses", 4, [0,1,2]), ("ring", 4, [3,4,5])]
            The sum of all parallel_sizes should equal world_size.
    
    Returns:
        batch_indices: List of sequence IDs assigned to the current rank's group
        group: The ProcessGroup for this rank's communication (None for single-rank groups)
        attn_type: "ulysses" or "ring" 
        sp_size: Ulysses parallel size (parallel_size if ulysses, 1 if ring)
        cp_size: Ring parallel size (1 if ulysses, parallel_size if ring)
    """
    global _global_group_set, _group_pool
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    # Safety: pad micro_res to cover all N GPUs
    total_covered = sum(ps for _, ps, _ in micro_res)
    if total_covered < world_size:
        remaining = world_size - total_covered
        # Pad with single-rank dummy groups
        for _ in range(remaining):
            micro_res.append(("ulysses", 1, []))
    
    cum_cnt = 0
    my_group = None
    my_batch_indices = []
    my_attn_type = "ulysses"
    my_sp_size = 1
    my_cp_size = 1
    
    for res_tuple in micro_res:
        attn_type, parallel_size, seq_id_list = res_tuple
        rank_start = cum_cnt
        rank_end = cum_cnt + parallel_size
        ranks = list(range(rank_start, rank_end))
        
        # Only create multi-rank groups (single-rank doesn't need a group)
        if parallel_size > 1:
            if tuple(ranks) not in _global_group_set:
                new_group = dist.new_group(ranks)
                _global_group_set.append(tuple(ranks))
                if rank in ranks:
                    _group_pool[tuple(ranks)] = new_group
        
        cum_cnt += parallel_size
        
        if rank in ranks:
            if parallel_size > 1:
                my_group = _group_pool[tuple(ranks)]
            else:
                my_group = None  # Single-rank group, no communication needed
            my_batch_indices = seq_id_list
            my_attn_type = attn_type
            if attn_type == "ulysses":
                my_sp_size = parallel_size
                my_cp_size = 1
            elif attn_type == "ring":
                my_sp_size = 1
                my_cp_size = parallel_size
            else:
                raise ValueError(f"Unknown attn_type: {attn_type}")
    
    return my_batch_indices, my_group, my_attn_type, my_sp_size, my_cp_size


def set_model_strategy(
    model: torch.nn.Module,
    sp_size: int,
    cp_size: int,
    sp_group: Optional[dist.ProcessGroup],
    cp_group: Optional[dist.ProcessGroup],
    attn_type: str,
):
    """
    Reconfigure all attention and embedding modules in the model 
    for the given per-rank strategy. Called before each microbatch forward.
    
    Following FlexSP pattern:
        args.sp_group = args.sp_groups[i]  # then model reads it in forward
    
    But we update module attributes directly for both SP and CP:
    - SelfAttention: sp_group, cp_group, sp_size, cp_size, use_ulysses, use_zigzag_cp
    - DistributedAttention: spg (sp process group) 
    - ZigzagRingFlashAttentionVarlen: cp_process_group
    - LlamaEmbeddings_: sp_group, cp_group, sp_size, cp_size
    
    Args:
        model: The full model
        sp_size: Ulysses parallel size for this rank
        cp_size: Ring attention parallel size for this rank
        sp_group: SP process group (for Ulysses All-to-All), or None
        cp_group: CP process group (for Ring P2P), or None
        attn_type: "ulysses" or "ring"
    """
    use_ulysses = (attn_type == "ulysses") and (sp_size > 1)
    use_zigzag_cp = (attn_type == "ring") and (cp_size > 1)
    
    for name, module in model.named_modules():
        # Update SelfAttention modules
        if isinstance(module, SelfAttention):
            module.sp_group = sp_group if use_ulysses else None
            module.cp_group = cp_group if use_zigzag_cp else None
            module.sp_size = sp_size
            module.cp_size = cp_size
            module.use_ulysses = use_ulysses
            module.use_zigzag_cp = use_zigzag_cp
            
            # Update DistributedAttention's sp group
            # (like FlexSP: self.dist_attn.spg = args.sp_group)
            if hasattr(module, 'dist_attn') and use_ulysses:
                module.dist_attn.spg = sp_group
                # Also set local_attention to the correct backend
                if hasattr(module, 'flash_attention'):
                    module.dist_attn.local_attn = module.flash_attention
            
            # Update ZigzagRingFlashAttention's cp group
            if hasattr(module, 'zigzag_ring_flash_attn') and use_zigzag_cp:
                module.zigzag_ring_flash_attn.cp_process_group = cp_group
        
        # Update LlamaEmbeddings_ modules
        elif hasattr(module, 'embed_tokens') and hasattr(module, 'cp_size') and hasattr(module, 'sp_size'):
            module.sp_group = sp_group if use_ulysses else None
            module.cp_group = cp_group if use_zigzag_cp else None
            module.sp_size = sp_size
            module.cp_size = cp_size
