#!/usr/bin/env python3
"""Trace USP (Ulysses + Ring) wrapper with PyTorch profiler.

Cases like "seq:sp:cp". Uses GALVATRON_ULYSSES_TRACE for the outer
DistributedAttention a2a markers; the inner ZigzagRingFlashAttentionVarlen
calls show up under "uly_fwd_local_attn" / "uly_bwd_*_a2a".
"""

import argparse
import gc
import importlib.util
import os
import sys
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("GALVATRON_ULYSSES_TRACE", "1")

import torch
import torch.distributed as dist

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(MODEL_DIR, "../../.."))
for path in (REPO_ROOT, MODEL_DIR,
              os.path.join(REPO_ROOT, "galvatron", "site_package")):
    if path not in sys.path:
        sys.path.insert(0, path)


def load_attention_impl():
    module_path = os.path.join(
        REPO_ROOT, "galvatron", "core", "runtime", "tensor_parallel",
        "attention_impl.py")
    spec = importlib.util.spec_from_file_location("trace_attn", module_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_GROUP_CACHE: Dict[Tuple[int, int], Tuple] = {}


def get_groups(sp_size: int, cp_size: int, world: int, rank: int):
    key = (sp_size, cp_size)
    parallel = sp_size * cp_size
    num_parallel = world // parallel
    my_sp_group = None
    my_cp_group = None
    my_cp_ranks = None
    for pg_idx in range(num_parallel):
        base = pg_idx * parallel
        base_ranks = list(range(base, base + parallel))
        for cp_idx in range(cp_size):
            sp_ranks = [base_ranks[cp_idx * sp_size + j] for j in range(sp_size)]
            g = dist.new_group(ranks=sp_ranks)
            if rank in sp_ranks:
                my_sp_group = g
        for sp_idx in range(sp_size):
            cp_ranks = [base_ranks[ci * sp_size + sp_idx] for ci in range(cp_size)]
            g = dist.new_group(ranks=cp_ranks)
            if rank in cp_ranks:
                my_cp_group = g
                my_cp_ranks = cp_ranks
    return my_sp_group, my_cp_group, my_cp_ranks


def make_cu(num_seqs, seq_len, device):
    return torch.arange(0, (num_seqs + 1) * seq_len, step=seq_len,
                         dtype=torch.int32, device=device)


def run_once(module, q_b, k_b, v_b, cu, max_seqlen):
    q = q_b.detach().clone().requires_grad_(True)
    k = k_b.detach().clone().requires_grad_(True)
    v = v_b.detach().clone().requires_grad_(True)
    with torch.profiler.record_function("usp_fwd"):
        out = module(q, k, v, 0, cu, max_seqlen)
    with torch.profiler.record_function("usp_bwd"):
        loss = out.float().square().mean()
        loss.backward()


def parse_cases(spec: str) -> List[Tuple[int, int, int]]:
    cases = []
    # accept space- or comma-separated tokens.
    tokens = spec.replace(",", " ").split()
    for tok in tokens:
        if not tok:
            continue
        seq, sp, cp = tok.split(":")
        cases.append((int(seq), int(sp), int(cp)))
    return cases


def run_case(attn_impl, rank, world, device, dtype, seq_len, sp_size, cp_size,
              num_seqs, n_heads, n_kv_heads, head_dim, warmup, active,
              trace_dir, profile_ranks):
    if world % (sp_size * cp_size) != 0:
        if rank == 0:
            print(f"[skip] world={world} not div by sp*cp={sp_size*cp_size}")
        return
    if seq_len % (2 * cp_size) != 0:
        if rank == 0:
            print(f"[skip] seq={seq_len} not div by 2*cp={2*cp_size}")
        return
    sp_group, cp_group, cp_ranks = get_groups(sp_size, cp_size, world, rank)
    cp_local_seq = seq_len // cp_size
    total_cp_local = num_seqs * cp_local_seq
    if total_cp_local % sp_size != 0:
        if rank == 0:
            print(f"[skip] total_cp_local={total_cp_local} not div by sp={sp_size}")
        return
    local_tokens = total_cp_local // sp_size

    local_attn = attn_impl.ZigzagRingFlashAttentionVarlen(
        attention_dropout=0.0, cp_group=cp_group, cp_ranks=cp_ranks, causal=True,
    )
    module = attn_impl.DistributedAttention(
        local_attention=local_attn, sequence_process_group=sp_group)

    cu = make_cu(num_seqs, cp_local_seq, device)
    q = torch.randn(1, local_tokens, n_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(1, local_tokens, n_kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(1, local_tokens, n_kv_heads, head_dim, dtype=dtype, device=device)

    for _ in range(warmup):
        run_once(module, q, k, v, cu, cp_local_seq)
    torch.cuda.synchronize()
    dist.barrier()

    case_dir = os.path.join(trace_dir, f"seq{seq_len}_sp{sp_size}_cp{cp_size}")
    if rank == 0:
        os.makedirs(case_dir, exist_ok=True)
    dist.barrier()

    should_profile = rank in set(profile_ranks)
    prof = None
    if should_profile:
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                         torch.profiler.ProfilerActivity.CUDA],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                case_dir, worker_name=f"rank{rank}"),
        )
        prof.start()
    for _ in range(active):
        run_once(module, q, k, v, cu, cp_local_seq)
        torch.cuda.synchronize()
        if prof is not None:
            prof.step()
    if prof is not None:
        prof.stop()
        if rank == min(profile_ranks):
            print(f"[trace] seq={seq_len} sp={sp_size} cp={cp_size} -> {case_dir}")
    del module, local_attn, q, k, v, cu
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trace-dir", required=True)
    p.add_argument("--cases", default="4096:4:2 4096:2:4")
    p.add_argument("--num-seqs", type=int, default=16)
    p.add_argument("--n-heads", type=int, default=28)
    p.add_argument("--n-kv-heads", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--active", type=int, default=2)
    p.add_argument("--profile-ranks", type=int, nargs="+", default=[0])
    args = p.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        os.makedirs(args.trace_dir, exist_ok=True)
        print(f"[trace] world={world} cases={args.cases}")
    dist.barrier()

    attn_impl = load_attention_impl()
    cases = parse_cases(args.cases)
    for seq, sp, cp in cases:
        if rank == 0:
            print(f"[trace] === seq={seq} sp={sp} cp={cp} ===", flush=True)
        run_case(attn_impl, rank, world, device, torch.bfloat16,
                  seq, sp, cp,
                  args.num_seqs, args.n_heads, args.n_kv_heads, args.head_dim,
                  args.warmup, args.active, args.trace_dir, args.profile_ranks)
    dist.barrier()


if __name__ == "__main__":
    main()
