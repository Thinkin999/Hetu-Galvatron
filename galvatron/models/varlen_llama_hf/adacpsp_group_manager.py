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


def _enumerate_all_possible_group_tuples(
    world_size: int,
    gpus_per_node: int,
    max_parallel_size: int,
    min_parallel_size: int = 1,
    allowed_attn_types: Tuple[str, ...] = ("ulysses", "ring", "usp"),
) -> List[Tuple[int, ...]]:
    """
    Enumerate every rank-tuple that `convert_microbatch_res` can ever pass to
    `_get_or_create_group`, given the solver's search space.

    Mirrors `AdaCPSPOptimizer._generate_gpu_partitions` and
    `_strategies_for_group_size` semantics:
      * Groups laid out on CONSECUTIVE rank ranges
      * Group size is a power of 2 in [min_parallel_size, max_parallel_size]
      * Base ranks are multiples of the group size that come out of partitions
        of N=world_size into descending powers of 2 — i.e. for each size s,
        bases {0, s, 2s, ...} where base+s <= world_size.
      * USP creates additional strided SP groups inside each block; CP groups
        in context_first happen to be consecutive (subset of simple groups);
        head_first swaps the roles.

    Returns a deterministic-ordered list of unique rank tuples.
    """
    max_ps = min(max_parallel_size or world_size, world_size)
    min_ps = max(2, min_parallel_size)

    sizes: List[int] = []
    s = 2
    while s <= max_ps:
        if s >= min_ps:
            sizes.append(s)
        s *= 2

    tuples: Set[Tuple[int, ...]] = set()

    # 1) Simple consecutive groups (Ulysses / Ring of size `size`)
    for size in sizes:
        for base in range(0, world_size - size + 1, size):
            tuples.add(tuple(range(base, base + size)))

    # 2) USP sub-groups within each consecutive block of size `block`
    if "usp" in allowed_attn_types:
        for block in sizes:
            if block < 4:
                continue  # USP requires sp>=2 AND cp>=2 → block >= 4
            # placements
            placements = ["context_first"]
            if block > gpus_per_node:
                placements.append("head_first")
            for base in range(0, world_size - block + 1, block):
                sp = 2
                while sp <= block // 2:
                    if block % sp != 0:
                        sp *= 2
                        continue
                    cp = block // sp
                    if cp < 2:
                        sp *= 2
                        continue
                    for placement in placements:
                        if placement == "context_first":
                            # rank(sp_idx, cp_idx) = base + sp_idx*cp + cp_idx
                            # CP groups consecutive, SP groups strided
                            for sp_idx in range(sp):
                                cp_ranks = tuple(
                                    base + sp_idx * cp + j for j in range(cp)
                                )
                                tuples.add(cp_ranks)
                            for cp_idx in range(cp):
                                sp_ranks = tuple(
                                    base + sp_idx2 * cp + cp_idx
                                    for sp_idx2 in range(sp)
                                )
                                tuples.add(sp_ranks)
                        else:
                            # head_first: rank(cp_idx, sp_idx) = base + cp_idx*sp + sp_idx
                            # SP groups consecutive, CP groups strided
                            for cp_idx in range(cp):
                                sp_ranks = tuple(
                                    base + cp_idx * sp + j for j in range(sp)
                                )
                                tuples.add(sp_ranks)
                            for sp_idx in range(sp):
                                cp_ranks = tuple(
                                    base + cp_idx2 * sp + sp_idx
                                    for cp_idx2 in range(cp)
                                )
                                tuples.add(cp_ranks)
                    sp *= 2

    # Deterministic, NCCL-friendly order: smaller groups first (cheaper init),
    # then by lexicographic rank order.
    return sorted(tuples, key=lambda t: (len(t), t))


def precreate_all_groups(
    world_size: Optional[int] = None,
    gpus_per_node: int = 8,
    max_parallel_size: Optional[int] = None,
    min_parallel_size: int = 1,
    allowed_attn_types: Tuple[str, ...] = ("ulysses", "ring", "usp"),
    warm_with_allreduce: bool = True,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    Eagerly create every NCCL ProcessGroup that the solver could ever dispatch
    to, then force NCCL communicator initialization via a tiny all_reduce so
    that subsequent steady-state iterations incur ZERO group-creation latency.

    MUST be called collectively on ALL ranks (same arguments) before any
    training step, and after `dist.init_process_group()` + CUDA device set.

    The cost of this call is amortized over the entire run:
      - For W=16, ~60-80 unique groups, ~3-5 min one-time
      - Eliminates 5-100s "first-touch" spikes during training
      - Solver decisions reflect TRUE steady-state cost, not warmup outliers

    Returns: dict with counters {tuples_total, tuples_new, warmups_run}
    """
    import time

    if world_size is None:
        world_size = dist.get_world_size()
    if max_parallel_size is None:
        max_parallel_size = world_size

    rank = dist.get_rank()
    tuples = _enumerate_all_possible_group_tuples(
        world_size=world_size,
        gpus_per_node=gpus_per_node,
        max_parallel_size=max_parallel_size,
        min_parallel_size=min_parallel_size,
        allowed_attn_types=allowed_attn_types,
    )

    if verbose and rank == 0:
        # Bucket by size for readable logging
        by_size: Dict[int, int] = {}
        for t in tuples:
            by_size[len(t)] = by_size.get(len(t), 0) + 1
        print(
            f"[GroupPrecreate] World={world_size}, gpn={gpus_per_node}, "
            f"max_ps={max_parallel_size}, attn={allowed_attn_types}"
        )
        print(
            f"[GroupPrecreate] Enumerated {len(tuples)} unique rank tuples: "
            + ", ".join(f"size{k}×{v}" for k, v in sorted(by_size.items()))
        )

    device = torch.cuda.current_device() if torch.cuda.is_available() else None
    dummy = (
        torch.zeros(1, device=f"cuda:{device}", dtype=torch.float32)
        if device is not None
        else None
    )

    t0 = time.time()
    new_groups = 0
    warmups = 0
    for idx, t in enumerate(tuples):
        existed = t in _created_group_keys
        # COLLECTIVE: every rank must enter this for each tuple, in identical order.
        grp = _get_or_create_group(list(t))
        if not existed:
            new_groups += 1
        # Force NCCL communicator init by issuing one tiny allreduce on members.
        # Non-members skip (no collective participation needed for a sub-group).
        if warm_with_allreduce and grp is not None and dummy is not None:
            dist.all_reduce(dummy, group=grp)
            warmups += 1
        # Progress log (rank 0 only, throttled)
        if verbose and rank == 0 and (idx + 1) % 16 == 0:
            elapsed = time.time() - t0
            print(
                f"[GroupPrecreate] {idx + 1}/{len(tuples)} groups warmed "
                f"({elapsed:.1f}s elapsed)",
                flush=True,
            )

    # Synchronize all NCCL streams before the global barrier so that any
    # pending warm-up allreduces are fully drained.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dist.barrier()
    elapsed = time.time() - t0

    if verbose and rank == 0:
        print(
            f"[GroupPrecreate] Done in {elapsed:.2f}s. "
            f"new_groups={new_groups}, warmups_on_this_rank={warmups}, "
            f"total_tuples={len(tuples)}"
        )

    return {
        "tuples_total": len(tuples),
        "tuples_new": new_groups,
        "warmups_run": warmups,
        "elapsed_s": elapsed,
    }


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
