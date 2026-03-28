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

def set_seed():
    seed = 123
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def collate_fn(batch):
    """
    Collate function — pure data packing, no distributed ops.

    Returns:
      - Non-packing mode: padded tensor [B, max_len]
      - Packing mode (with or without AdaCPSP): [packed_tokens, cu_seqlens]
        The training loop handles solver + group assignment for AdaCPSP.
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

    cu_seqlens = torch.empty(len(batch) + 1, dtype=torch.int64)
    cu_seqlens[0] = 0
    for i in range(1, len(cu_seqlens)):
        cu_seqlens[i] = cu_seqlens[i - 1] + len(batch[i - 1])
    packed = torch.concat(batch)
    return [packed, cu_seqlens]


def distributed_dataloader(dataset, global_bsz, shuffle=True, args=None, group=None,
                           adaCPSP_optimizer_=None, adaCPSP_forced_strategy_=None):
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
