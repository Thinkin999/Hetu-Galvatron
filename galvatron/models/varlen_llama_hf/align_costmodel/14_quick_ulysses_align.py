"""Quick fresh measurement of Ulysses attention layer time across sp×seq grid.

Mirrors 12_quick_ring_align.py but for Ulysses. Loops over sp_size and seq_len
in a single torchrun process to save 5-7 minutes of init time per case.

For Qwen2.5-7B (n_q=28, n_kv=4):
  - sp=2, 4: no head padding (n_kv % sp == 0)
  - sp=8: kv_factor=2.0, q_factor=2.0 (replicates 4 extra kv heads)
  - sp=16: kv_factor=4.0, q_factor=4.0

We use *consecutive* topology (sp groups within a node when sp<=8, crossing
the node boundary when sp=16).

Outputs JSON: { "measured_per_layer_ms": {sp: {seq: ms}} }
"""

import argparse
import gc
import importlib.util
import json
import os
import socket
import statistics
import sys
from typing import Dict, List

import torch
import torch.distributed as dist

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(MODEL_DIR, "../../.."))
SITE_PACKAGE_DIR = os.path.join(REPO_ROOT, "galvatron", "site_package")
for p in (SITE_PACKAGE_DIR, REPO_ROOT, MODEL_DIR):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)


def _load_attn_impl():
    path = os.path.join(REPO_ROOT, "galvatron", "core", "runtime",
                         "tensor_parallel", "attention_impl.py")
    spec = importlib.util.spec_from_file_location("align_attn_impl", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ATTN = _load_attn_impl()
DistributedAttention = _ATTN.DistributedAttention
FlashSelfAttentionVarlen = _ATTN.FlashSelfAttentionVarlen


def _consecutive_sp_groups(world_size: int, sp_size: int):
    """Return a list of rank-lists, one per sp group, consecutive topology."""
    assert world_size % sp_size == 0, f"world {world_size} not divisible by sp {sp_size}"
    return [list(range(i, i + sp_size)) for i in range(0, world_size, sp_size)]


def make_cu_seqlens(num_seqs: int, seq_len: int, device):
    return torch.arange(0, (num_seqs + 1) * seq_len, step=seq_len,
                        dtype=torch.int32, device=device)


def build_inputs(seq_len: int, num_seqs: int, n_heads: int, n_kv_heads: int,
                 head_dim: int, sp_size: int, device, dtype):
    """For Ulysses: each rank holds seq_len/sp_size of tokens per sequence,
    with full n_heads/n_kv_heads count. After Q a2a it becomes seq_len full
    with n/sp heads.
    """
    total_tokens = num_seqs * seq_len
    if total_tokens % sp_size != 0:
        raise ValueError(f"total_tokens={total_tokens} not divisible by sp={sp_size}")
    local_tokens = total_tokens // sp_size
    cu = make_cu_seqlens(num_seqs, seq_len, device)
    max_seqlen = seq_len
    q = torch.randn(1, local_tokens, n_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(1, local_tokens, n_kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(1, local_tokens, n_kv_heads, head_dim, device=device, dtype=dtype)
    return q, k, v, cu, max_seqlen


def run_one(module, q, k, v, cu, max_seqlen) -> None:
    q_ = q.detach().clone().requires_grad_(True)
    k_ = k.detach().clone().requires_grad_(True)
    v_ = v.detach().clone().requires_grad_(True)
    out = module(q_, k_, v_, 0, cu, max_seqlen)
    loss = out.float().square().mean()
    loss.backward()


_SP_GROUP_CACHE: Dict[int, "dist.ProcessGroup"] = {}


def get_or_build_sp_group(sp_size: int, world_size: int, rank: int):
    """Cache sp_groups by size to avoid NCCL communicator leak across cases."""
    if sp_size in _SP_GROUP_CACHE:
        return _SP_GROUP_CACHE[sp_size]
    my_sp_group = None
    for ranks in _consecutive_sp_groups(world_size, sp_size):
        g = dist.new_group(ranks=ranks)
        if rank in ranks:
            my_sp_group = g
    _SP_GROUP_CACHE[sp_size] = my_sp_group
    return my_sp_group


def benchmark_case(sp_size: int, seq_len: int, num_seqs: int,
                   n_heads: int, n_kv_heads: int, head_dim: int,
                   warmup: int, iters: int,
                   world_size: int, rank: int):
    """Build sp groups (cached), run benchmark, return max-across-ranks median ms."""
    my_sp_group = get_or_build_sp_group(sp_size, world_size, rank)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    q, k, v, cu, max_seqlen = build_inputs(
        seq_len, num_seqs, n_heads, n_kv_heads, head_dim, sp_size, device, dtype,
    )
    local_attn = FlashSelfAttentionVarlen(causal=True, attention_dropout=0.0)
    module = DistributedAttention(
        local_attention=local_attn,
        sequence_process_group=my_sp_group,
    )

    for _ in range(warmup):
        run_one(module, q, k, v, cu, max_seqlen)
    torch.cuda.synchronize()
    dist.barrier()

    per_iter = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run_one(module, q, k, v, cu, max_seqlen)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        t = torch.tensor([ms], device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        per_iter.append(float(t.item()))

    median_ms = statistics.median(per_iter)
    # Explicit cleanup so the next case starts from a clean cuda state.
    del q, k, v, cu, module, local_attn
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    dist.barrier()
    return median_ms, per_iter


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sps", default="2,4,8,16")
    p.add_argument("--seqs", default="4096,8192,16384,32768")
    p.add_argument("--num-seqs", type=int, default=16)
    p.add_argument("--n-heads", type=int, default=28)
    p.add_argument("--n-kv-heads", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--output-json", default=None)
    args = p.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    if rank == 0:
        print(f"[bench] world={world} host={socket.gethostname()}", flush=True)

    sps: List[int] = [int(x) for x in args.sps.split(",")]
    seqs: List[int] = [int(x) for x in args.seqs.split(",")]

    results = {}
    for sp in sps:
        if world % sp != 0:
            if rank == 0:
                print(f"[bench] skip sp={sp}: world={world} not divisible", flush=True)
            continue
        results[sp] = {}
        for seq in seqs:
            tag = f"sp={sp} seq={seq}"
            if rank == 0:
                print(f"\n[bench] {tag} num_seqs={args.num_seqs}", flush=True)
            try:
                measured, per_iter = benchmark_case(
                    sp, seq, args.num_seqs,
                    args.n_heads, args.n_kv_heads, args.head_dim,
                    args.warmup, args.iters,
                    world, rank,
                )
                results[sp][seq] = measured
                if rank == 0:
                    p50 = statistics.median(per_iter)
                    p90 = sorted(per_iter)[min(len(per_iter)-1,
                                                int(0.9 * len(per_iter)) - 1)]
                    print(f"[bench] {tag} median={measured:.2f}ms "
                          f"(p50/p90/max={p50:.2f}/{p90:.2f}/{max(per_iter):.2f})",
                          flush=True)
            except Exception as e:
                if rank == 0:
                    print(f"[bench] {tag} FAILED: {e}", flush=True)
                results[sp][seq] = None

    if rank == 0 and args.output_json:
        # serialize sp keys to strings for valid json
        ser = {str(sp): {str(seq): v for seq, v in vals.items()}
               for sp, vals in results.items()}
        with open(args.output_json, "w") as f:
            json.dump({"measured_per_layer_ms": ser,
                       "num_seqs": args.num_seqs,
                       "world": world,
                       "topology": "consecutive"}, f, indent=2)
        print(f"[bench] wrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
