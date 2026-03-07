import torch
import numpy as np
import random 
import torch.distributed as dist
import torch.distributed
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def get_args():
    from galvatron.core import get_args as _get_args
    return _get_args()

flexSP_optimizer = None
adaCPSP_optimizer = None
adaCPSP_forced_strategy = None  # For testing: override solver with hardcoded heterogeneous groups

def set_seed():
    seed = 123
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def collate_fn(batch):
    """
    Collate function that handles:
    1. Non-packing mode: standard padding
    2. Packing mode without solver: simple concatenation
    3. AdaCPSP mode: solver determines HETEROGENEOUS groups per microbatch
       following FlexSP's convert_microbatch_res pattern.
       
       Key design (tp=1):
       - Each microbatch can have MULTIPLE groups of DIFFERENT sizes
       - Groups map to CONSECUTIVE GPU ranks
       - Each rank belongs to exactly ONE group per microbatch
       - Different groups can use different attn_types (Ulysses / Ring / USP)
       - All ranks in same group receive the SAME packed sequences
       - The embedding layer then splits by SP or CP for per-rank data
    """
    max_len = max([len(seq) for seq in batch])
    world_size = torch.distributed.get_world_size()
    max_len = ((max_len - 1) // world_size + 1) * world_size
    args = get_args()
    max_len = min(max_len, args.seq_length)
    
    if not args.use_packing:
        padded_batch = torch.zeros((len(batch), max_len), dtype=torch.long, device=batch[0].device)
        for i, seq in enumerate(batch):
            padded_batch[i, :len(seq)] = seq
        return padded_batch
    
    if not adaCPSP_optimizer:
        # Simple packing (no solver)
        cu_seqlens = torch.empty(len(batch) + 1, dtype=torch.int64)
        cu_seqlens[0] = 0
        for i in range(1, len(cu_seqlens)):
            cu_seqlens[i] = cu_seqlens[i-1] + len(batch[i - 1])
        packed = torch.concat(batch)
        return [packed, cu_seqlens]
    
    # ═══════════════════════════════════════════════════════════════
    # AdaCPSP mode: heterogeneous groups per microbatch
    # Supports ulysses, ring, AND usp (combined ulysses + ring)
    # ═══════════════════════════════════════════════════════════════
    from galvatron.models.varlen_llama_hf.adacpsp_solver import Sequence
    from galvatron.models.varlen_llama_hf.adacpsp_group_manager import convert_microbatch_res
    
    rank = dist.get_rank()
    
    # Reset per-iteration strategy storage
    args.adacpsp_strategies = []     # per-microbatch strategy for this rank
    args.adacpsp_sp_groups = []      # per-microbatch SP group for this rank
    args.adacpsp_cp_groups = []      # per-microbatch CP group for this rank
    
    # ─── Step 1: Rank 0 runs solver (or use forced strategy) ───
    all_groups = None
    if rank == 0:
        seqs = [Sequence(seq=s.shape[0], id=i) for i, s in enumerate(batch)]
        
        if adaCPSP_forced_strategy is not None:
            # Forced heterogeneous strategy for testing
            all_groups = _build_forced_groups(seqs, world_size, adaCPSP_forced_strategy)
            if rank == 0:
                print(f"[AdaCPSP] Using FORCED strategy: {adaCPSP_forced_strategy}")
        else:
            all_groups, all_results = adaCPSP_optimizer.solve_globalbatch(seqs)
        
        if len(all_groups) == 0:
            print("[AdaCPSP] Solver failed, fallback to Ulysses×" + str(world_size))
            from galvatron.models.varlen_llama_hf.adacpsp_solver import ParallelStrategy
            fallback_strat = ParallelStrategy("ulysses", world_size)
            all_groups = [[(fallback_strat, seqs)]]
    
    # ─── Step 2: Broadcast results (ALL ranks participate in every broadcast) ───
    # Broadcast number of microbatches
    num_mb_val = len(all_groups) if rank == 0 else 0
    num_mb_t = torch.LongTensor([num_mb_val]).cuda()
    dist.broadcast(num_mb_t, 0)
    num_mb = num_mb_t.item()
    
    # For each microbatch: broadcast groups, then convert_microbatch_res
    all_micro_res = []
    for mb_idx in range(num_mb):
        # Broadcast number of groups in this microbatch
        ng_val = len(all_groups[mb_idx]) if rank == 0 else 0
        ng_t = torch.LongTensor([ng_val]).cuda()
        dist.broadcast(ng_t, 0)
        n_groups = ng_t.item()
        
        micro_res = []
        for g_idx in range(n_groups):
            # Broadcast group info:
            #   [attn_code, parallel_size, sp_size, cp_size, num_seqs, seq_id_0, ...]
            #   attn_code: 0=ulysses, 1=ring, 2=usp
            if rank == 0:
                strat, group_seqs = all_groups[mb_idx][g_idx]
                attn_code = {"ulysses": 0, "ring": 1, "usp": 2}[strat.attn_type]
                seq_ids = [seq.id for seq in group_seqs]
                info = torch.LongTensor(
                    [attn_code, strat.parallel_size, strat.sp_size, strat.cp_size,
                     len(seq_ids)] + seq_ids
                ).cuda()
                info_len_t = torch.LongTensor([len(info)]).cuda()
            else:
                info_len_t = torch.LongTensor([0]).cuda()
            
            dist.broadcast(info_len_t, 0)
            
            if rank != 0:
                info = torch.zeros(info_len_t.item(), dtype=torch.long).cuda()
            dist.broadcast(info, 0)
            
            # Decode
            info_list = info.cpu().tolist()
            attn_code = int(info_list[0])
            parallel_size = int(info_list[1])
            sp_size = int(info_list[2])
            cp_size = int(info_list[3])
            num_seqs = int(info_list[4])
            seq_ids = [int(x) for x in info_list[5:5+num_seqs]]
            attn_type = {0: "ulysses", 1: "ring", 2: "usp"}[attn_code]
            micro_res.append((attn_type, parallel_size, sp_size, cp_size, seq_ids))
        
        all_micro_res.append(micro_res)
    
    # ─── Step 3: Convert solver output to per-rank assignment ───
    # ALL ranks call convert_microbatch_res (which calls dist.new_group, a collective)
    microbatches = []
    for mb_idx, micro_res in enumerate(all_micro_res):
        my_seq_ids, my_sp_group, my_cp_group, my_attn_type, my_sp_size, my_cp_size = \
            convert_microbatch_res(micro_res)
        
        # Store per-rank strategy
        strat_info = {
            "sp_size": my_sp_size,
            "cp_size": my_cp_size,
            "attn_type": my_attn_type,
        }
        args.adacpsp_strategies.append(strat_info)
        
        # Store per-rank groups (USP has BOTH sp_group and cp_group)
        args.adacpsp_sp_groups.append(my_sp_group)
        args.adacpsp_cp_groups.append(my_cp_group)
        
        # ─── Build packed tokens + cu_seqlens for THIS GROUP's sequences ───
        # All ranks in the same group get the SAME packed sequences.
        # The embedding layer then splits by SP or CP for per-rank data.
        group_seqs = [batch[sid] for sid in my_seq_ids]
        if len(group_seqs) == 0:
            # Safety: this rank's group has no sequences (shouldn't happen normally)
            packed_tokens = torch.zeros(1, dtype=torch.long, device=batch[0].device)
            cu_seqlens = torch.zeros(2, dtype=torch.int64)
            cu_seqlens[1] = 1
        else:
            cu_seqlens = torch.zeros(len(group_seqs) + 1, dtype=torch.int64)
            for j, seq in enumerate(group_seqs):
                cu_seqlens[j + 1] = cu_seqlens[j] + len(seq)
            packed_tokens = torch.cat(group_seqs)
        
        microbatches.append([[packed_tokens, cu_seqlens]])
    
    if rank == 0:
        for mb_idx, strat in enumerate(args.adacpsp_strategies):
            print(f"  [AdaCPSP] MB{mb_idx}: type={strat['attn_type']}, "
                  f"sp={strat['sp_size']}, cp={strat['cp_size']}")
    
    return microbatches


def _build_forced_groups(seqs, world_size, forced_config):
    """
    Build forced heterogeneous groups for testing.
    
    forced_config: list of tuples, each of which is EITHER:
        (attn_type, parallel_size)                    — for ulysses / ring
        (attn_type, parallel_size, sp_size, cp_size)  — for usp
    
    Examples:
        [("ulysses", 8)]                    → 1 group of 8 GPUs
        [("ring", 2)]                       → 4 groups of 2 GPUs (auto-replicated)
        [("ulysses", 4), ("ring", 4)]       → ranks 0-3: Ulysses×4, ranks 4-7: Ring×4
        [("usp", 8, 2, 4)]                  → 1 group of 8 GPUs: sp=2, cp=4
    
    If the sum of parallel_sizes < world_size and there is exactly one entry,
    the entry is replicated to fill all GPUs.
    """
    from galvatron.models.varlen_llama_hf.adacpsp_solver import ParallelStrategy
    
    # Normalise entries to (attn_type, parallel_size, sp_size, cp_size)
    normalised = []
    for entry in forced_config:
        if len(entry) == 2:
            attn_type, ps = entry
            if attn_type == "ulysses":
                normalised.append((attn_type, ps, ps, 1))
            elif attn_type == "ring":
                normalised.append((attn_type, ps, 1, ps))
            else:
                raise ValueError(f"USP requires 4-tuple: (attn_type, parallel_size, sp_size, cp_size)")
        elif len(entry) == 4:
            normalised.append(tuple(entry))
        else:
            raise ValueError(f"Unexpected forced_config entry: {entry}")
    
    total_ps = sum(ps for _, ps, _, _ in normalised)
    
    # Auto-replicate: if single strategy doesn't cover all GPUs, repeat it
    if total_ps < world_size and len(normalised) == 1:
        _, ps, sp, cp = normalised[0]
        at = normalised[0][0]
        assert world_size % ps == 0, \
            f"world_size ({world_size}) must be divisible by parallel_size ({ps})"
        num_replicas = world_size // ps
        normalised = [(at, ps, sp, cp)] * num_replicas
        total_ps = sum(ps for _, ps, _, _ in normalised)
    
    assert total_ps == world_size, \
        f"Forced strategy total parallel_sizes ({total_ps}) != world_size ({world_size})"
    
    num_groups = len(normalised)
    
    # Distribute sequences round-robin across groups
    group_seqs = [[] for _ in range(num_groups)]
    for i, seq in enumerate(seqs):
        group_seqs[i % num_groups].append(seq)
    
    # Build the groups list (single microbatch containing all groups)
    groups = []
    for (attn_type, parallel_size, sp_size, cp_size), g_seqs in zip(normalised, group_seqs):
        strat = ParallelStrategy(
            attn_type=attn_type,
            parallel_size=parallel_size,
            sp_size=sp_size,
            cp_size=cp_size,
        )
        groups.append((strat, g_seqs))
    
    # All sequences in one microbatch
    return [groups]


def distributed_dataloader(dataset, global_bsz, shuffle=True, args=None, group=None, 
                           adaCPSP_optimizer_=None, adaCPSP_forced_strategy_=None):
    global adaCPSP_optimizer, adaCPSP_forced_strategy
    adaCPSP_optimizer = adaCPSP_optimizer_
    adaCPSP_forced_strategy = adaCPSP_forced_strategy_
    
    if args is not None and getattr(args, 'use_adaCPSP', False):
        # ═══════════════════════════════════════════════════════════
        # AdaCPSP: ALL ranks must load the SAME full global batch
        # because the solver makes a GLOBAL decision and then
        # convert_microbatch_res distributes sequences to rank groups.
        # 
        # This is the same pattern as FlexSP where dp=1 (all sp/cp):
        # every rank sees the full batch, solver partitions it.
        # ═══════════════════════════════════════════════════════════
        rank = torch.distributed.get_rank()
        train_batch_size_input = global_bsz
        trainloader = DataLoader(
            dataset=dataset,
            batch_size=train_batch_size_input,
            sampler=DistributedSampler(dataset, shuffle=shuffle, num_replicas=1, rank=0),
            collate_fn=collate_fn
        )
        return trainloader
    else:
        rank = torch.distributed.get_rank(group)
        world_size = torch.distributed.get_world_size(group)
        train_batch_size_input = global_bsz // world_size
        trainloader = DataLoader(
            dataset=dataset,
            batch_size=train_batch_size_input,
            sampler=DistributedSampler(dataset, shuffle=shuffle, num_replicas=world_size, rank=rank),
            collate_fn=collate_fn
        )
        return trainloader


def print_loss(args, loss, ep, iter):
    if args.check_loss or args.profile:
        if loss is None:
            return
        if isinstance(loss, (list, tuple)):
            if len(loss) == 0:
                return
            if isinstance(loss[0], torch.Tensor):
                loss = np.mean([l.item() for l in loss])
            else:
                loss = np.mean(loss)
        else:
            loss = loss.item() if isinstance(loss, torch.Tensor) else loss
        print('[Epoch %d] (Iteration %d): Loss = %.3f' % (ep, iter, loss))
