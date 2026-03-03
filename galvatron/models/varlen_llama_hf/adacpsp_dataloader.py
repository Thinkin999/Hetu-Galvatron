"""
AdaCPSP DataLoader: Data loading with adaptive CP/SP strategy selection.

This module follows the pattern:
- Global state management
- collate_fn returns microbatches in format: [[[tokens, cu_seqlens]], ...]
- Communication groups passed through args.sp_groups and args.cp_groups

Strategy representation: (cp_size, sp_size, [seq_ids...])
- cp_size: Context Parallel size (Zigzag Ring Attention)
- sp_size: Ulysses SP size (AlltoAll)
- seq_ids: List of sequence indices assigned to this group
- Total parallel size = cp_size * sp_size
"""

import torch
import torch.distributed as dist
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import List, Dict, Tuple, Optional
import multiprocessing as mp

from galvatron.core import get_args
from .adacpsp_solver import (
    Sequence, AdaCPSPCostModel, AdaCPSPOptimizer, 
    AdaCPSPConfig
)


# Global state
global_group_set = []  # indicate whether a group is created
sp_group_pool = {}  # SP communication groups (Ulysses)
cp_group_pool = {}  # CP communication groups (Zigzag Ring)
is_first_iter = True
solve_process = None
mp_manager = mp.Manager()
solved_globalbatch_gps = mp_manager.list()  # solved microbatch assignments
adacpsp_optimizer = None
prev_batch = None


def solve_target(seqs: List[Sequence], shared_globalbatch_gps, strategy: str):
    """Solver target function for multiprocessing.
    
    Args:
        seqs: List of Sequence objects
        shared_globalbatch_gps: Shared list for results
        strategy: 'adaptive', 'ulysses', or 'cp'
        
    Output format: List of microbatches, each microbatch is a list of tuples:
        (cp_size, sp_size, [seq_ids...])
    """
    global adacpsp_optimizer
    
    if adacpsp_optimizer is None:
        return
    
    adacpsp_optimizer.token_info(seqs)
    import time
    
    start = time.time()
    print(f'Running AdaCPSP Solver (strategy={strategy})...')
    
    if strategy == 'adaptive':
        results = adacpsp_optimizer.solve_heterogeneous(seqs)
    else:
        results = adacpsp_optimizer.solve_simple(seqs, strategy=strategy)
    
    end = time.time()
    print(f'Solver Time Cost: {end-start:.4f}s')
    
    # Print solution
    adacpsp_optimizer.print_solution(results)
    
    # Convert to globalbatch format using triplet: (cp_size, sp_size, [seq_ids...])
    # Each microbatch contains groups that can run in parallel
    globalbatch_groups = []
    
    for config, group_seqs in results:
        # Triplet representation: (cp_size, sp_size, seq_ids)
        group_info = (
            config.cp_size,      # CP size (Zigzag Ring)
            config.sp_size,      # SP size (Ulysses AlltoAll)
            [seq.id for seq in group_seqs]  # Sequence IDs
        )
        globalbatch_groups.append(group_info)
    
    # All groups form one microbatch (they run on different GPU subsets)
    if globalbatch_groups:
        shared_globalbatch_gps.append(globalbatch_groups)


def convert_microbatch_res(micro_res):
    """Convert microbatch result to batch_indices, sp_group, cp_group.
    
    Args:
        micro_res: List of (cp_size, sp_size, seq_ids) triplets
        
    Returns:
        batch_indices: List of sequence indices for this rank
        sp_group: SP communication group for this rank (or None)
        cp_group: CP communication group for this rank (or None)
        cp_size: CP size for this rank
        sp_size: SP size for this rank
    """
    global global_group_set, sp_group_pool, cp_group_pool
    
    cum_cnt = 0
    sp_group = None
    cp_group = None
    batch_indices = []
    my_cp_size = 1
    my_sp_size = 1
    
    for res_tuple in micro_res:
        # Triplet format: (cp_size, sp_size, seq_ids)
        cp_size, sp_size, seq_id_list = res_tuple
        total_size = cp_size * sp_size
        
        rank_start = cum_cnt
        rank_end = cum_cnt + total_size
        ranks = list(range(rank_start, rank_end))
        
        current_rank = torch.distributed.get_rank()
        
        # Create SP groups (Ulysses AlltoAll) if sp_size > 1
        # SP groups: ranks within same CP position
        # Layout: [sp0_cp0, sp1_cp0, ..., sp0_cp1, sp1_cp1, ...]
        if sp_size > 1:
            for cp_idx in range(cp_size):
                sp_start = rank_start + cp_idx * sp_size
                sp_ranks = list(range(sp_start, sp_start + sp_size))
                
                if tuple(sp_ranks) not in global_group_set:
                    sp_group_ = torch.distributed.new_group(sp_ranks)
                    global_group_set.append(tuple(sp_ranks))
                    if current_rank in sp_ranks:
                        sp_group_pool[tuple(sp_ranks)] = sp_group_
        
        # Create CP groups (Zigzag Ring) if cp_size > 1
        # CP groups: ranks with same SP position across CP
        if cp_size > 1:
            for sp_idx in range(sp_size):
                cp_ranks = [rank_start + sp_idx + i * sp_size for i in range(cp_size)]
                
                if tuple(cp_ranks) not in global_group_set:
                    cp_group_ = torch.distributed.new_group(cp_ranks)
                    global_group_set.append(tuple(cp_ranks))
                    if current_rank in cp_ranks:
                        cp_group_pool[tuple(cp_ranks)] = cp_group_
        
        # Get groups for current rank
        if current_rank in ranks:
            local_rank = current_rank - rank_start
            batch_indices = seq_id_list
            my_cp_size = cp_size
            my_sp_size = sp_size
            
            # Get SP group for this rank
            if sp_size > 1:
                cp_idx = local_rank // sp_size
                sp_start = rank_start + cp_idx * sp_size
                sp_ranks = tuple(range(sp_start, sp_start + sp_size))
                sp_group = sp_group_pool.get(sp_ranks)
            
            # Get CP group for this rank
            if cp_size > 1:
                sp_idx = local_rank % sp_size
                cp_ranks = tuple([rank_start + sp_idx + i * sp_size for i in range(cp_size)])
                cp_group = cp_group_pool.get(cp_ranks)
        
        cum_cnt += total_size
    
    return batch_indices, sp_group, cp_group, my_cp_size, my_sp_size


def pad_sequence_for_cp(seq_tensor: torch.Tensor, cp_size: int) -> torch.Tensor:
    """Pad sequence to be divisible by 2 * cp_size for zigzag splitting.
    
    For Zigzag CP, each sequence must have length divisible by 2 * cp_size.
    """
    original_len = seq_tensor.shape[0]
    divisor = 2 * cp_size
    
    if original_len % divisor == 0:
        return seq_tensor
    
    pad_len = divisor - (original_len % divisor)
    padded = torch.nn.functional.pad(seq_tensor, (0, pad_len), value=0)
    return padded


def extract_local_for_zigzag(
    packed_tensor: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cp_rank: int,
    cp_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract local portion of packed sequences for zigzag CP.
    
    For zigzag pattern, each rank gets:
    - First half: positions [rank * chunk_size : (rank + 1) * chunk_size]
    - Second half: positions [(2*cp_size - 1 - rank) * chunk_size : (2*cp_size - rank) * chunk_size]
    """
    num_seqs = cu_seqlens.shape[0] - 1
    local_chunks = []
    local_cu_seqlens = [0]
    
    for i in range(num_seqs):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        seq_len = end - start
        chunk_size = seq_len // (2 * cp_size)
        
        # First half chunk
        first_start = start + cp_rank * chunk_size
        first_end = first_start + chunk_size
        
        # Second half chunk (zigzag pattern)
        second_idx = 2 * cp_size - 1 - cp_rank
        second_start = start + second_idx * chunk_size
        second_end = second_start + chunk_size
        
        # Extract chunks
        first_chunk = packed_tensor[first_start:first_end]
        second_chunk = packed_tensor[second_start:second_end]
        
        local_chunks.append(first_chunk)
        local_chunks.append(second_chunk)
        local_cu_seqlens.append(local_cu_seqlens[-1] + 2 * chunk_size)
    
    local_tensor = torch.cat(local_chunks, dim=0)
    local_cu_seqlens = torch.tensor(local_cu_seqlens, dtype=cu_seqlens.dtype, device=cu_seqlens.device)
    
    return local_tensor, local_cu_seqlens


def set_seed():
    """Set random seed for reproducibility."""
    seed = 123
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def collate_fn(batch):
    """Collate function for AdaCPSP (following FlexSP pattern).
    
    Args:
        batch: List of sequence tensors from dataset
        
    Returns:
        - None for first iteration
        - List of microbatches: [[[tokens, cu_seqlens]], ...]
        - Or simple [tokens, cu_seqlens] if not using AdaCPSP
        
    Strategy representation: (cp_size, sp_size, [seq_ids...])
    """
    global is_first_iter, solve_process, solved_globalbatch_gps, prev_batch, adacpsp_optimizer
    
    max_len = max([len(seq) for seq in batch])
    world_size = torch.distributed.get_world_size()
    max_len = ((max_len - 1) // world_size + 1) * world_size
    args = get_args()
    max_len = min(max_len, args.seq_length)
    
    if not args.use_packing:
        # Padding mode (no packing)
        padded_batch = torch.zeros((len(batch), max_len), dtype=torch.long, device=batch[0].device)
        for i, seq in enumerate(batch):
            padded_batch[i, :len(seq)] = seq
        return padded_batch
    
    # Packing mode with AdaCPSP
    if adacpsp_optimizer:
        seqs = [Sequence(sentence.shape[0], seq_id=i) for i, sentence in enumerate(batch)]
        rank = dist.get_rank()
        strategy = getattr(args, 'adacpsp_strategy', 'adaptive')
        
        if rank == 0:
            if not is_first_iter:
                solve_process.join()
                globalbatch_groups = list(solved_globalbatch_gps)
                solved_globalbatch_gps[:] = []  # Clear the list
            solve_process = mp.Process(target=solve_target, args=(seqs, solved_globalbatch_gps, strategy))
            solve_process.start()
        
        if is_first_iter:
            is_first_iter = False
            prev_batch = batch
            return None
        
        # Synchronize
        torch.distributed.barrier()
        
        # Initialize args attributes for groups
        args.sp_groups = []      # Ulysses SP groups
        args.cp_groups = []      # Zigzag CP groups
        args.cp_sizes = []       # CP size for each microbatch
        args.sp_sizes = []       # SP size for each microbatch
        
        microbatches = []
        n_mbatch = 0
        
        # Broadcast number of microbatches
        if rank == 0:
            micro_bsz = torch.LongTensor([len(globalbatch_groups)]).cuda()
            dist.broadcast(micro_bsz, 0)
        else:
            micro_bsz = torch.LongTensor([0]).cuda()
            dist.broadcast(micro_bsz, 0)
        
        for mb_idx in range(micro_bsz.item()):
            n_mbatch += 1
            
            if rank == 0:
                microbatch_group = globalbatch_groups[mb_idx]
                ele_num = torch.LongTensor([len(microbatch_group)]).cuda()
                dist.broadcast(ele_num, 0)
                
                for i in range(ele_num.item()):
                    # Triplet format: (cp_size, sp_size, seq_ids)
                    cp_size, sp_size, seq_ids = microbatch_group[i]
                    
                    # Broadcast group info
                    cp_size_t = torch.LongTensor([cp_size]).cuda()
                    sp_size_t = torch.LongTensor([sp_size]).cuda()
                    num_seqs = torch.LongTensor([len(seq_ids)]).cuda()
                    seq_ids_t = torch.LongTensor(seq_ids).cuda()
                    
                    dist.broadcast(cp_size_t, 0)
                    dist.broadcast(sp_size_t, 0)
                    dist.broadcast(num_seqs, 0)
                    dist.broadcast(seq_ids_t, 0)
            else:
                microbatch_group = []
                ele_num = torch.LongTensor([0]).cuda()
                dist.broadcast(ele_num, 0)
                
                for i in range(ele_num.item()):
                    cp_size_t = torch.LongTensor([0]).cuda()
                    sp_size_t = torch.LongTensor([0]).cuda()
                    num_seqs = torch.LongTensor([0]).cuda()
                    
                    dist.broadcast(cp_size_t, 0)
                    dist.broadcast(sp_size_t, 0)
                    dist.broadcast(num_seqs, 0)
                    
                    seq_ids_t = torch.LongTensor(num_seqs.item()).cuda()
                    dist.broadcast(seq_ids_t, 0)
                    
                    seq_ids = [j.item() for j in seq_ids_t]
                    # Triplet format: (cp_size, sp_size, seq_ids)
                    microbatch_group.append((
                        cp_size_t.item(),
                        sp_size_t.item(),
                        seq_ids
                    ))
            
            # Convert result to get batch indices and groups
            batch_indices, sp_group, cp_group, cp_size, sp_size = convert_microbatch_res(microbatch_group)
            
            # Get sequences for this rank
            m_batch = [prev_batch[idx] for idx in batch_indices]
            
            # Store groups and sizes in args
            args.sp_groups.append(sp_group)
            args.cp_groups.append(cp_group)
            args.cp_sizes.append(cp_size)
            args.sp_sizes.append(sp_size)
            
            torch.distributed.barrier()
            
            # Pad sequences for CP if needed
            if cp_size > 1 and cp_group is not None:
                m_batch = [pad_sequence_for_cp(seq, cp_size) for seq in m_batch]
            
            # Create cu_seqlens
            cu_seqlens = torch.empty(len(m_batch) + 1, dtype=torch.int64)
            cu_seqlens[0] = 0
            for i in range(1, len(cu_seqlens)):
                cu_seqlens[i] = cu_seqlens[i-1] + len(m_batch[i-1])
            
            # Concatenate sequences
            m_batch_tensor = torch.concat(m_batch)
            
            # Extract local portion for zigzag CP if needed
            if cp_size > 1 and cp_group is not None:
                cp_rank = torch.distributed.get_rank(cp_group)
                m_batch_tensor, cu_seqlens = extract_local_for_zigzag(
                    m_batch_tensor, cu_seqlens, cp_rank, cp_size
                )
            
            # Format: [[[tokens, cu_seqlens]]] to match FlexSP
            microbatches.append([[m_batch_tensor, cu_seqlens]])
        
        prev_batch = batch
        return microbatches
    
    else:
        # Simple packing without AdaCPSP
        cu_seqlens = torch.empty(len(batch) + 1, dtype=torch.int64)
        cu_seqlens[0] = 0
        for i in range(1, len(cu_seqlens)):
            cu_seqlens[i] = cu_seqlens[i-1] + len(batch[i-1])
        batch = torch.concat(batch)
        return [batch, cu_seqlens]


def distributed_dataloader(dataset, global_bsz, shuffle=True, args=None, group=None, adacpsp_optimizer_=None):
    """Create distributed dataloader with AdaCPSP support.
    
    Args:
        dataset: Dataset to load from
        global_bsz: Global batch size
        shuffle: Whether to shuffle
        args: Training arguments
        group: Communication group for sampling
        adacpsp_optimizer_: AdaCPSP optimizer instance
        
    Returns:
        DataLoader with AdaCPSP collate function
    """
    rank = torch.distributed.get_rank(group)
    world_size = torch.distributed.get_world_size(group)
    
    global adacpsp_optimizer
    adacpsp_optimizer = adacpsp_optimizer_
    
    train_batch_size_input = global_bsz // world_size
    
    trainloader = DataLoader(
        dataset=dataset,
        batch_size=train_batch_size_input,
        sampler=DistributedSampler(dataset, shuffle=shuffle, num_replicas=world_size, rank=rank),
        collate_fn=collate_fn
    )
    
    return trainloader


def print_loss(args, loss, ep, iter):
    """Print loss (following FlexSP pattern)."""
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
