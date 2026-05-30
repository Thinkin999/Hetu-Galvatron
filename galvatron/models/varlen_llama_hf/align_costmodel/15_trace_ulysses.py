#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trace Ulysses (DistributedAttention) wrapper with PyTorch profiler.

Mirrors 08_trace_ring_attention.py but for Ulysses. Each case is `seq_len:sp_size`.
Per-op record_function markers (uly_fwd_q_a2a, uly_fwd_k_a2a, uly_fwd_v_a2a,
uly_fwd_local_attn, uly_fwd_o_a2a; uly_bwd_{q,k,v,o}_a2a) are emitted inside
attention_impl.py when GALVATRON_ULYSSES_TRACE=1.
"""

import argparse
import importlib.util
import os
import sys
from typing import List, Optional, Tuple

os.environ.setdefault("GALVATRON_ULYSSES_TRACE", "1")

import torch
import torch.distributed as dist

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(MODEL_DIR, "../../.."))
for path in (REPO_ROOT, MODEL_DIR, os.path.join(REPO_ROOT, "galvatron", "site_package")):
    if path not in sys.path:
        sys.path.insert(0, path)


def load_attention_impl():
    module_path = os.path.join(
        REPO_ROOT, "galvatron", "core", "runtime", "tensor_parallel", "attention_impl.py"
    )
    spec = importlib.util.spec_from_file_location("trace_attention_impl", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load attention_impl from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_cu_seqlens(num_seqs: int, seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.arange(0, (num_seqs + 1) * seq_len, step=seq_len, dtype=torch.int32, device=device)


def consecutive_sp_groups(world_size: int, sp_size: int) -> List[List[int]]:
    assert world_size % sp_size == 0
    return [list(range(i, i + sp_size)) for i in range(0, world_size, sp_size)]


def build_sp_group(world_size: int, rank: int, sp_size: int
                    ) -> Tuple[Optional[dist.ProcessGroup], Optional[List[int]]]:
    my_group = None
    my_ranks = None
    for ranks in consecutive_sp_groups(world_size, sp_size):
        group = dist.new_group(ranks=ranks)
        if rank in ranks:
            my_group = group
            my_ranks = ranks
    return my_group, my_ranks


def run_once(module, q_base, k_base, v_base, cu, max_seqlen) -> float:
    q = q_base.detach().clone().requires_grad_(True)
    k = k_base.detach().clone().requires_grad_(True)
    v = v_base.detach().clone().requires_grad_(True)
    with torch.profiler.record_function("uly_fwd"):
        out = module(q, k, v, 0, cu, max_seqlen)
    with torch.profiler.record_function("uly_bwd"):
        loss = out.float().square().mean()
        loss.backward()
    return float(loss.item())


def parse_cases(spec: str) -> List[Tuple[int, int]]:
    cases = []
    for tok in spec.split():
        if not tok:
            continue
        seq, sp = tok.split(":")
        cases.append((int(seq), int(sp)))
    return cases


def run_case(
    attn_impl,
    rank: int,
    world_size: int,
    device: torch.device,
    dtype: torch.dtype,
    seq_len: int,
    sp_size: int,
    num_seqs: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    warmup: int,
    active: int,
    trace_dir: str,
    profile_ranks: List[int],
) -> None:
    if sp_size <= 0 or world_size % sp_size != 0:
        if rank == 0:
            print(f"[Trace][skip] sp_size={sp_size} does not divide world_size={world_size}")
        return

    sp_group, sp_ranks = build_sp_group(world_size, rank, sp_size)
    total_tokens = num_seqs * seq_len
    if total_tokens % sp_size != 0:
        if rank == 0:
            print(f"[Trace][skip] total_tokens={total_tokens} not divisible by sp={sp_size}")
        return
    local_tokens = total_tokens // sp_size

    local_attn = attn_impl.FlashSelfAttentionVarlen(causal=True, attention_dropout=0.0)
    module = attn_impl.DistributedAttention(
        local_attention=local_attn,
        sequence_process_group=sp_group,
    ).to(device)
    module.train()

    cu = make_cu_seqlens(num_seqs, seq_len, device)
    q = torch.randn(1, local_tokens, n_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(1, local_tokens, n_kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(1, local_tokens, n_kv_heads, head_dim, dtype=dtype, device=device)

    for _ in range(warmup):
        run_once(module, q, k, v, cu, seq_len)
    torch.cuda.synchronize()
    dist.barrier(group=sp_group)

    case_trace_dir = os.path.join(trace_dir, f"seq{seq_len}_sp{sp_size}")
    if rank == 0:
        os.makedirs(case_trace_dir, exist_ok=True)
    dist.barrier()

    should_profile = rank in set(profile_ranks)
    prof = None
    if should_profile:
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            with_stack=False,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                case_trace_dir,
                worker_name=f"rank{rank}",
            ),
        )
        prof.start()

    for _ in range(active):
        run_once(module, q, k, v, cu, seq_len)
        torch.cuda.synchronize()
        if prof is not None:
            prof.step()

    if prof is not None:
        prof.stop()
        if rank == min(profile_ranks):
            print(f"[Trace] seq={seq_len} sp={sp_size} -> {case_trace_dir}")

    torch.cuda.synchronize()
    dist.barrier()
    del module


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace Ulysses (DistributedAttention) wrapper")
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument(
        "--cases",
        default="8192:16 32768:16",
        help='Space-separated "seq:sp" pairs, e.g. "4096:2 8192:8 32768:16"',
    )
    parser.add_argument("--num-seqs", type=int, default=16)
    parser.add_argument("--n-heads", type=int, default=28)
    parser.add_argument("--n-kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--active", type=int, default=3)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--profile-ranks", type=int, nargs="+", default=[0, 8])
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    cases = parse_cases(args.cases)
    if rank == 0:
        os.makedirs(args.trace_dir, exist_ok=True)
        print(f"[Trace] world_size={world_size}, cases={cases}")
        print(f"[Trace] trace_dir={args.trace_dir}")
    dist.barrier()

    attn_impl = load_attention_impl()
    if not getattr(attn_impl, "_ULYSSES_TRACE", False):
        if rank == 0:
            print(
                "[Trace][warn] attention_impl was imported without "
                "GALVATRON_ULYSSES_TRACE=1; per-op markers will be missing.",
                file=sys.stderr,
            )

    for seq_len, sp_size in cases:
        if rank == 0:
            print(f"[Trace] === case seq={seq_len} sp={sp_size} ===", flush=True)
        run_case(
            attn_impl=attn_impl,
            rank=rank,
            world_size=world_size,
            device=device,
            dtype=dtype,
            seq_len=seq_len,
            sp_size=sp_size,
            num_seqs=args.num_seqs,
            n_heads=args.n_heads,
            n_kv_heads=args.n_kv_heads,
            head_dim=args.head_dim,
            warmup=args.warmup,
            active=args.active,
            trace_dir=args.trace_dir,
            profile_ranks=args.profile_ranks,
        )

    dist.barrier()


if __name__ == "__main__":
    main()
