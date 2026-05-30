"""Minimal cross-node NCCL sanity check.

Runs all_reduce on a 1M-fp32 tensor across all ranks for a few iterations
and prints per-rank bandwidth. If init_process_group itself hangs, the
configured TORCH_NCCL_BLOCKING_WAIT timeout will surface the error within
~5 minutes instead of the default 30.

Usage (via torchrun, see 10_sanity_nccl_dispatch.sh):
  torchrun --nnodes 2 --nproc_per_node 8 --master_addr ... 10_sanity_nccl.py
"""

import os
import socket
import time

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    host = socket.gethostname()
    print(f"[sanity] {host} local_rank={local_rank} starting init_process_group", flush=True)
    t0 = time.time()
    dist.init_process_group(backend="nccl")
    init_dt = time.time() - t0
    rank = dist.get_rank()
    world = dist.get_world_size()
    if rank == 0:
        print(f"[sanity] init_process_group OK ({init_dt:.1f}s) world={world}", flush=True)

    numel = 1 << 20  # 1M fp32 = 4 MiB
    x = torch.ones(numel, device="cuda")
    for _ in range(5):
        dist.all_reduce(x)
    torch.cuda.synchronize()

    n_iter = 50
    t0 = time.time()
    for _ in range(n_iter):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / n_iter

    bytes_per_iter = numel * 4
    bw_gbps = bytes_per_iter / dt / 1e9
    if rank == 0:
        print(
            f"[sanity] all_reduce({numel} fp32) world={world}  "
            f"mean={dt*1000:.3f} ms  alg_bw={bw_gbps:.2f} GB/s",
            flush=True,
        )
        print("[sanity] PASS", flush=True)

    dist.barrier()


if __name__ == "__main__":
    main()
