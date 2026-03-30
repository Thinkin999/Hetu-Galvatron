"""
AdaCPSP Communication Group Manager

Follows FlexSP's `convert_microbatch_res` pattern:
  - Each microbatch contains MULTIPLE groups of potentially different sizes
  - Groups are mapped to CONSECUTIVE GPU ranks
  - Each rank belongs to exactly ONE group per microbatch  
  - Different groups can use different attn_types (Ulysses / Ring / USP)

Example: 8 GPUs, microbatch = [("ulysses", 4, 4, 1, [s0,s1,s2]), ("ring", 4, 1, 4, [s3,s4,s5])]
  → Rank 0-3: Ulysses sp_size=4, processing sequences [s0,s1,s2]
  → Rank 4-7: Ring cp_size=4, processing sequences [s3,s4,s5]

USP example: 8 GPUs, microbatch = [("usp", 8, 2, 4, [s0,..,s5])]
  → All 8 GPUs: sp_size=2, cp_size=4
  → SP groups (stride cp_size): [0,4], [1,5], [2,6], [3,7]
  → CP groups (contiguous, size cp_size): [0,1,2,3], [4,5,6,7]

Design:
  - tp_deg = 1 (no tensor parallelism, full weights on every GPU)
  - FSDP covers all GPUs (dp = world_size at construction)
  - sp_size and cp_size are BOTH dynamic, determined by solver
  - Groups are created lazily by SIZE and cached (size-based, no SP/CP distinction)
  - For USP: a 2D mesh yields both SP groups and CP groups on the same ranks
"""

import torch
import torch.distributed as dist
from typing import Dict, List, Optional, Set, Tuple

from galvatron.core.runtime.tensor_parallel.attention import SelfAttention
from galvatron.core.runtime.tensor_parallel.attention_impl import (
    DistributedAttention,
    ZigzagRingFlashAttentionVarlen,
    FlashSelfAttentionVarlen,
)


# Per-process caches. Two layers:
# - _created_group_keys: every rank records that this rank-tuple has already been
#   created via a collective dist.new_group(), so subsequent calls must skip
#   new_group (fixes member/non-member _group_pool missync).
# - _group_pool: member ranks store the usable ProcessGroup handle only.
_created_group_keys: Set[Tuple[int, ...]] = set()
_group_pool: Dict[Tuple[int, ...], dist.ProcessGroup] = {}


def _get_or_create_group(ranks: List[int]) -> Optional[dist.ProcessGroup]:
    """
    Get or lazily create a ProcessGroup for the given rank list.

    IMPORTANT: This is a collective operation — ALL ranks in the world must
    call dist.new_group() with the same ranks, even if they're not in the group.
    The function returns the group only for ranks that are members.

    For parallel_size == 1, returns None (no communication needed).
    """
    global _group_pool, _created_group_keys

    if len(ranks) <= 1:
        return None  # No group needed for single-rank

    key = tuple(ranks)
    rank = dist.get_rank()

    if key not in _created_group_keys:
        # Collective: all ranks must take this branch once per key, in lockstep.
        new_group = dist.new_group(ranks)
        _created_group_keys.add(key)
        if rank in ranks:
            _group_pool[key] = new_group

    if rank in ranks:
        return _group_pool.get(key)
    return None


def convert_microbatch_res(micro_res):
    """
    Convert solver's microbatch result to per-rank assignment.

    Follows FlexSP's convert_microbatch_res pattern:
    - Groups are laid out on consecutive GPU ranks
    - Each rank finds which group it belongs to
    - Creates communication groups lazily (all ranks MUST call new_group together)

    If the total parallel_sizes don't cover all N GPUs, remaining ranks are
    assigned to single-rank dummy groups (sp=1, cp=1, no sequences).

    Args:
        micro_res: List of tuples. Supports both formats:
            5-tuple (legacy): (attn_type, parallel_size, sp_size, cp_size, [seq_ids])
            6-tuple (placement-aware): (attn_type, parallel_size, sp_size, cp_size,
                                        placement, [seq_ids])

    Returns:
        batch_indices, sp_group, cp_group, attn_type, sp_size, cp_size, placement
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Normalize to 6-tuple format
    normalized = []
    for res_tuple in micro_res:
        if len(res_tuple) == 5:
            at, ps, sp, cp, sids = res_tuple
            normalized.append((at, ps, sp, cp, "context_first", sids))
        else:
            normalized.append(res_tuple)

    total_covered = sum(ps for _, ps, _, _, _, _ in normalized)
    if total_covered < world_size:
        remaining = world_size - total_covered
        for _ in range(remaining):
            normalized.append(("ulysses", 1, 1, 1, "context_first", []))

    cum_cnt = 0
    my_sp_group = None
    my_cp_group = None
    my_batch_indices = []
    my_attn_type = "ulysses"
    my_sp_size = 1
    my_cp_size = 1
    my_placement = "context_first"

    all_groups_to_create = []

    for res_tuple in normalized:
        attn_type, parallel_size, sp_size, cp_size, placement, seq_id_list = res_tuple
        base_rank = cum_cnt
        group_ranks = list(range(base_rank, base_rank + parallel_size))

        if attn_type in ("ulysses", "ring"):
            if parallel_size > 1:
                all_groups_to_create.append(("simple", group_ranks, attn_type, sp_size, cp_size, placement, seq_id_list))
            else:
                all_groups_to_create.append(("none", group_ranks, attn_type, sp_size, cp_size, placement, seq_id_list))
        elif attn_type == "usp":
            all_groups_to_create.append(("usp", group_ranks, attn_type, sp_size, cp_size, placement, seq_id_list))
        else:
            raise ValueError(f"Unknown attn_type: {attn_type}")

        cum_cnt += parallel_size

    for entry in all_groups_to_create:
        kind = entry[0]
        group_ranks = entry[1]
        attn_type = entry[2]
        sp_size = entry[3]
        cp_size = entry[4]
        placement = entry[5]
        seq_id_list = entry[6]
        base_rank = group_ranks[0]

        if kind == "none":
            if rank in group_ranks:
                my_batch_indices = seq_id_list
                my_attn_type = attn_type
                my_sp_size = sp_size
                my_cp_size = cp_size
                my_placement = placement
                my_sp_group = None
                my_cp_group = None

        elif kind == "simple":
            grp = _get_or_create_group(group_ranks)
            if rank in group_ranks:
                my_batch_indices = seq_id_list
                my_attn_type = attn_type
                my_sp_size = sp_size
                my_cp_size = cp_size
                my_placement = placement
                if attn_type == "ulysses":
                    my_sp_group = grp
                    my_cp_group = None
                else:
                    my_sp_group = None
                    my_cp_group = grp

        elif kind == "usp":
            if placement == "head_first":
                # Head-first: SP groups consecutive, CP groups strided
                # rank(cp_idx, sp_idx) = base_rank + cp_idx * sp_size + sp_idx
                for cp_idx in range(cp_size):
                    sp_ranks = [base_rank + cp_idx * sp_size + j for j in range(sp_size)]
                    sp_grp = _get_or_create_group(sp_ranks)
                    if rank in sp_ranks:
                        my_sp_group = sp_grp

                for sp_idx in range(sp_size):
                    cp_ranks = [base_rank + cp_idx * sp_size + sp_idx for cp_idx in range(cp_size)]
                    cp_grp = _get_or_create_group(cp_ranks)
                    if rank in cp_ranks:
                        my_cp_group = cp_grp
            else:
                # Context-first (default): CP groups consecutive, SP groups strided
                # rank(sp_idx, cp_idx) = base_rank + sp_idx * cp_size + cp_idx
                for sp_idx in range(sp_size):
                    cp_ranks = [base_rank + sp_idx * cp_size + j for j in range(cp_size)]
                    cp_grp = _get_or_create_group(cp_ranks)
                    if rank in cp_ranks:
                        my_cp_group = cp_grp

                for cp_idx in range(cp_size):
                    sp_ranks = [base_rank + sp_idx * cp_size + cp_idx for sp_idx in range(sp_size)]
                    sp_grp = _get_or_create_group(sp_ranks)
                    if rank in sp_ranks:
                        my_sp_group = sp_grp

            if rank in group_ranks:
                my_batch_indices = seq_id_list
                my_attn_type = "usp"
                my_sp_size = sp_size
                my_cp_size = cp_size
                my_placement = placement

    return my_batch_indices, my_sp_group, my_cp_group, my_attn_type, my_sp_size, my_cp_size, my_placement


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

    Supports three modes:
      - attn_type="ulysses": use_ulysses=True, use_zigzag_cp=False
        → dist_attn wraps flash_attention (All-to-All only)
      - attn_type="ring": use_ulysses=False, use_zigzag_cp=True
        → zigzag_ring_flash_attn handles P2P ring
      - attn_type="usp": use_ulysses=True, use_zigzag_cp=True
        → dist_attn wraps zigzag_ring_flash_attn
          (All-to-All scatter → Ring P2P → All-to-All gather)

    Args:
        model: The full model
        sp_size: Ulysses parallel size for this rank
        cp_size: Ring attention parallel size for this rank
        sp_group: SP process group (for Ulysses All-to-All), or None
        cp_group: CP process group (for Ring P2P), or None
        attn_type: "ulysses", "ring", or "usp"
    """
    use_ulysses = (attn_type in ("ulysses", "usp")) and (sp_size > 1)
    use_zigzag_cp = (attn_type in ("ring", "usp")) and (cp_size > 1)

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
            if hasattr(module, 'dist_attn') and use_ulysses:
                module.dist_attn.spg = sp_group
                # For USP: dist_attn's local_attn = zigzag_ring_flash_attn
                # For pure Ulysses: dist_attn's local_attn = flash_attention
                if use_zigzag_cp and hasattr(module, 'zigzag_ring_flash_attn'):
                    module.dist_attn.local_attn = module.zigzag_ring_flash_attn
                elif hasattr(module, 'flash_attention'):
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
