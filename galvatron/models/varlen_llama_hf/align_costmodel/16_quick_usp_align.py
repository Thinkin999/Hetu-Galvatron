"""Quick fresh measurement of USP (Ulysses+Ring) attention layer time.

Mirrors 14_quick_ulysses_align but for USP. Loops over (sp, cp, seq) grid in
a single torchrun process to save init time per case.

We use head_first placement (sp-major within node, then cp across nodes when
needed). For Qwen2.5-7B (n_q=28, n_kv=4):
  - sp=2, 4: no head padding
  - sp=8: q_factor=2, kv_factor=2
  - sp=16: q_factor=4, kv_factor=4

Output JSON: { "measured_per_layer_ms": {sp: {cp: {seq: ms}}} }
"""

import argparse
import gc
import importlib.util
import json
import os
import socket
import statistics
import sys
from typing import Dict, List, Optional, Tuple

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
ZigzagRingFlashAttentionVarlen = _ATTN.ZigzagRingFlashAttentionVarlen


# ---- group construction (head_first placement, consecutive topology) ----
# For 16-GPU world with sp_size, cp_size where sp*cp == parallel_size:
#   parallel_groups = world / parallel_size groups of size parallel_size
#   Within each parallel group, head_first means sp groups are contiguous
#   (intra-node when possible), then cp connects across sp groups.
# Example for sp=2,cp=4,world=16: 2 parallel groups of size 8.
#   group 0 ranks: [0,1,2,3,4,5,6,7]
#       sp groups (head_first): [0,1], [2,3], [4,5], [6,7]
#       cp groups (head_first): [0,2,4,6], [1,3,5,7]


_GROUP_CACHE: Dict[Tuple[int, int], Tuple[Optional[dist.ProcessGroup],
                                          Optional[dist.ProcessGroup]]] = {}


def get_or_build_sp_cp_groups(sp_size: int, cp_size: int, world: int, rank: int
                                ) -> Tuple[Optional[dist.ProcessGroup],
                                            Optional[dist.ProcessGroup],
                                            Optional[List[int]]]:
    """Build sp and cp groups under head_first/consecutive placement.

    Returns (sp_group, cp_group, cp_ranks_for_my_rank) for the rank's groups.
    """
    key = (sp_size, cp_size)
    cached = _GROUP_CACHE.get(key)
    parallel = sp_size * cp_size
    assert world % parallel == 0, (
        f"world={world} not divisible by sp*cp={parallel}")
    num_parallel = world // parallel

    my_sp_group = None
    my_cp_group = None
    my_cp_ranks = None
    for pg_idx in range(num_parallel):
        base = pg_idx * parallel
        base_ranks = list(range(base, base + parallel))
        # sp_groups within this parallel group: head_first => sp groups are
        # contiguous blocks of size sp.
        sp_groups_in_pg = []
        for cp_idx in range(cp_size):
            sp_ranks = [base_ranks[cp_idx * sp_size + j] for j in range(sp_size)]
            sp_groups_in_pg.append(sp_ranks)
        # cp_groups: take one rank from each sp_group, in order
        cp_groups_in_pg = []
        for sp_idx in range(sp_size):
            cp_ranks = [base_ranks[cp_idx * sp_size + sp_idx]
                        for cp_idx in range(cp_size)]
            cp_groups_in_pg.append(cp_ranks)
        for sp_ranks in sp_groups_in_pg:
            g = dist.new_group(ranks=sp_ranks)
            if rank in sp_ranks:
                my_sp_group = g
        for cp_ranks in cp_groups_in_pg:
            g = dist.new_group(ranks=cp_ranks)
            if rank in cp_ranks:
                my_cp_group = g
                my_cp_ranks = cp_ranks
    _GROUP_CACHE[key] = (my_sp_group, my_cp_group)
    return my_sp_group, my_cp_group, my_cp_ranks


def make_cu_seqlens(num_seqs: int, seq_len: int, device):
    return torch.arange(0, (num_seqs + 1) * seq_len, step=seq_len,
                        dtype=torch.int32, device=device)


def build_inputs_usp(seq_len: int, num_seqs: int, n_heads: int, n_kv_heads: int,
                     head_dim: int, sp_size: int, cp_size: int, device, dtype):
    """USP: each rank holds (seq/cp tokens) / sp = seq/(sp*cp) tokens per sequence.

    Pre-a2a Q shape: (1, local_tokens_per_seq*num_seqs, n_q, hd) where
    local_tokens_per_seq = seq / cp / sp (note: cp is at the outer ring layer,
    sp partitions further within).

    Actually, in galvatron's USP layout: the ring partitions seq into cp chunks,
    then ulysses scatters each chunk's heads. So pre-a2a tokens per rank
    = (seq/cp) * num_seqs (after ring chunking but before a2a)?

    Looking at 03_benchmark_attention build_inputs for usp branch:
        cp_local_seq = seq_len // cp_size
        total_cp_local_tokens = num_seqs * cp_local_seq
        local_tokens = total_cp_local_tokens // sp_size  # after Ulysses a2a (scatter heads, gather seq)
    Wait that's after a2a. Let me re-check.

    For USP DistributedAttention wraps ZigzagRingFlashAttentionVarlen as local_attn.
    The input to DistributedAttention.forward is (1, local_tokens, n_heads, hd)
    where local_tokens = pre-a2a tokens for this rank. Ulysses a2a will then
    redistribute and feed to Ring (which has its own per-rank seq chunking).

    Per 03_benchmark_attention:
        local_tokens = total_cp_local_tokens // sp_size
    where total_cp_local_tokens = num_seqs * cp_local_seq = num_seqs * seq/cp.
    So local_tokens = num_seqs * seq / (cp * sp).
    cu_seqlens uses cp_local_seq = seq/cp as the per-sequence chunk size after ring.
    max_seqlen = cp_local_seq.

    Q is (1, local_tokens, n_q, hd) which after a2a becomes
    (1, total_cp_local_tokens, n_q/sp, hd), then Ring expects (seq/cp)-chunked
    input per its own rank.
    """
    if seq_len % (2 * cp_size) != 0:
        raise ValueError(f"seq_len={seq_len} must be divisible by 2*cp_size={2*cp_size}")
    cp_local_seq = seq_len // cp_size
    total_cp_local_tokens = num_seqs * cp_local_seq
    if total_cp_local_tokens % sp_size != 0:
        raise ValueError(
            f"num_seqs*seq/cp={total_cp_local_tokens} not divisible by sp={sp_size}")
    local_tokens = total_cp_local_tokens // sp_size
    cu = make_cu_seqlens(num_seqs, cp_local_seq, device)
    max_seqlen = cp_local_seq
    q = torch.randn(1, local_tokens, n_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(1, local_tokens, n_kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(1, local_tokens, n_kv_heads, head_dim, device=device, dtype=dtype)
    return q, k, v, cu, max_seqlen


def run_one(module, q, k, v, cu, max_seqlen):
    q_ = q.detach().clone().requires_grad_(True)
    k_ = k.detach().clone().requires_grad_(True)
    v_ = v.detach().clone().requires_grad_(True)
    out = module(q_, k_, v_, 0, cu, max_seqlen)
    loss = out.float().square().mean()
    loss.backward()


def benchmark_case(sp_size: int, cp_size: int, seq_len: int, num_seqs: int,
                   n_heads: int, n_kv_heads: int, head_dim: int,
                   warmup: int, iters: int,
                   world: int, rank: int):
    sp_group, cp_group, cp_ranks = get_or_build_sp_cp_groups(
        sp_size, cp_size, world, rank)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    q, k, v, cu, max_seqlen = build_inputs_usp(
        seq_len, num_seqs, n_heads, n_kv_heads, head_dim, sp_size, cp_size,
        device, dtype,
    )
    local_attn = ZigzagRingFlashAttentionVarlen(
        attention_dropout=0.0,
        cp_group=cp_group,
        cp_ranks=cp_ranks,
        causal=True,
    )
    module = DistributedAttention(
        local_attention=local_attn,
        sequence_process_group=sp_group,
    )

    for _ in range(warmup):
        run_one(module, q, k, v, cu, max_seqlen)
    torch.cuda.synchronize()
    dist.barrier()

    per_iter = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        run_one(module, q, k, v, cu, max_seqlen)
        e.record()
        torch.cuda.synchronize()
        ms = s.elapsed_time(e)
        t = torch.tensor([ms], device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        per_iter.append(float(t.item()))

    median_ms = statistics.median(per_iter)
    del q, k, v, cu, module, local_attn
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    dist.barrier()
    return median_ms, per_iter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", default="2x8,4x4,8x2,2x4,4x2,2x2",
                   help="comma-separated sp x cp pairs")
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

    cases: List[Tuple[int, int]] = []
    for tok in args.cases.split(","):
        sp_s, cp_s = tok.split("x")
        cases.append((int(sp_s), int(cp_s)))
    seqs: List[int] = [int(x) for x in args.seqs.split(",")]

    # Filter invalid cases
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for sp, cp in cases:
        if world % (sp * cp) != 0:
            if rank == 0:
                print(f"[bench] skip sp={sp} cp={cp}: world={world} not div by {sp*cp}",
                      flush=True)
            continue
        results.setdefault(str(sp), {}).setdefault(str(cp), {})
        for seq in seqs:
            tag = f"sp={sp} cp={cp} seq={seq}"
            if rank == 0:
                print(f"\n[bench] {tag} num_seqs={args.num_seqs}", flush=True)
            try:
                measured, per_iter = benchmark_case(
                    sp, cp, seq, args.num_seqs,
                    args.n_heads, args.n_kv_heads, args.head_dim,
                    args.warmup, args.iters,
                    world, rank,
                )
                results[str(sp)][str(cp)][str(seq)] = measured
                if rank == 0:
                    p50 = statistics.median(per_iter)
                    p90 = sorted(per_iter)[min(len(per_iter)-1,
                                                int(0.9 * len(per_iter))-1)]
                    print(f"[bench] {tag} median={measured:.2f}ms "
                          f"(p50/p90/max={p50:.2f}/{p90:.2f}/"
                          f"{max(per_iter):.2f})", flush=True)
            except Exception as e:
                if rank == 0:
                    print(f"[bench] {tag} FAILED: {e}", flush=True)
                results[str(sp)][str(cp)][str(seq)] = None

    if rank == 0 and args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({
                "measured_per_layer_ms": results,
                "num_seqs": args.num_seqs,
                "world": world,
                "placement": "head_first",
                "topology": "consecutive",
            }, f, indent=2)
        print(f"[bench] wrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
