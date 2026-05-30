#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trace RingAttention wrapper with PyTorch profiler.

Supports a multi-case sweep in a single torchrun (saves init_process_group cost).
Each case is `seq_len:cp_size`; the script:

  1. builds the cp_group for this case
  2. warms up
  3. opens a profiler and runs `active` ring fwd+bwd iterations
  4. closes the profiler, writes Chrome trace JSON per rank
  5. moves to the next case

Per-step `record_function` markers (`ring_fwd_step_{N}_{phase}` /
`ring_bwd_step_{N}_{phase}`) are emitted inside attention_impl.py only when
`GALVATRON_RING_TRACE_PER_STEP=1`. This script sets it automatically.
"""

import argparse
import importlib.util
import os
import sys
from typing import List, Optional, Tuple

# Enable per-step record_function markers BEFORE attention_impl is imported.
os.environ.setdefault("GALVATRON_RING_TRACE_PER_STEP", "1")

import torch
import torch.distributed as dist


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(MODEL_DIR, "../../.."))
for path in (REPO_ROOT, MODEL_DIR, os.path.join(REPO_ROOT, "galvatron", "site_package")):
    if path not in sys.path:
        sys.path.insert(0, path)

from profile_topo_utils import build_group_ranks_list  # noqa: E402


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


def build_ring_group(
    world_size: int, rank: int, cp_size: int, topology: str
) -> Tuple[Optional[dist.ProcessGroup], Optional[List[int]]]:
    """Build all cp_size-sized groups; return the one containing my rank."""
    groups = build_group_ranks_list(world_size, cp_size, topology)
    my_group = None
    my_ranks = None
    for ranks in groups:
        group = dist.new_group(ranks=ranks)
        if rank in ranks:
            my_group = group
            my_ranks = ranks
    return my_group, my_ranks


def run_once(module, q_base, k_base, v_base, cu, max_seqlen) -> float:
    q = q_base.detach().clone().requires_grad_(True)
    k = k_base.detach().clone().requires_grad_(True)
    v = v_base.detach().clone().requires_grad_(True)
    with torch.profiler.record_function("ring_fwd"):
        out = module(q, k, v, cu, max_seqlen)
    with torch.profiler.record_function("ring_bwd"):
        loss = out.float().square().mean()
        loss.backward()
    return float(loss.item())


def parse_cases(spec: str) -> List[Tuple[int, int]]:
    """Parse `"seq:cp seq:cp ..."` -> [(seq, cp), ...]."""
    cases = []
    for tok in spec.split():
        if not tok:
            continue
        seq, cp = tok.split(":")
        cases.append((int(seq), int(cp)))
    return cases


def run_case(
    attn_impl,
    rank: int,
    world_size: int,
    device: torch.device,
    dtype: torch.dtype,
    seq_len: int,
    cp_size: int,
    num_seqs: int,
    topology: str,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    warmup: int,
    active: int,
    trace_dir: str,
    profile_ranks: List[int],
    record_shapes: bool,
    with_stack: bool,
) -> None:
    if cp_size <= 0 or world_size % cp_size != 0:
        if rank == 0:
            print(f"[Trace][skip] cp_size={cp_size} does not divide world_size={world_size}")
        return
    if seq_len % (2 * cp_size) != 0:
        if rank == 0:
            print(f"[Trace][skip] seq_len={seq_len} not divisible by 2*cp_size={2 * cp_size}")
        return

    cp_group, cp_ranks = build_ring_group(world_size, rank, cp_size, topology)
    if cp_group is None or cp_ranks is None:
        if rank == 0:
            print(f"[Trace][skip] rank {rank} not in any cp_group for cp_size={cp_size}")
        return

    module = attn_impl.ZigzagRingFlashAttentionVarlen(
        attention_dropout=0.0,
        cp_group=cp_group,
        cp_ranks=cp_ranks,
        causal=True,
    ).to(device)
    module.train()

    local_seq = seq_len // cp_size
    local_tokens = num_seqs * local_seq
    cu = make_cu_seqlens(num_seqs, local_seq, device)
    q = torch.randn(1, local_tokens, n_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(1, local_tokens, n_kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(1, local_tokens, n_kv_heads, head_dim, dtype=dtype, device=device)

    for _ in range(warmup):
        run_once(module, q, k, v, cu, local_seq)
    torch.cuda.synchronize()
    dist.barrier(group=cp_group)

    case_trace_dir = os.path.join(trace_dir, f"seq{seq_len}_cp{cp_size}_{topology}")
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
            record_shapes=record_shapes,
            with_stack=with_stack,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                case_trace_dir,
                worker_name=f"rank{rank}",
            ),
        )
        prof.start()

    for _ in range(active):
        run_once(module, q, k, v, cu, local_seq)
        torch.cuda.synchronize()
        if prof is not None:
            prof.step()

    if prof is not None:
        prof.stop()
        if rank == min(profile_ranks):
            print(f"[Trace] seq={seq_len} cp={cp_size} -> {case_trace_dir}")

    torch.cuda.synchronize()
    dist.barrier()
    # NOTE: do NOT call dist.destroy_process_group(cp_group) here. NCCL's
    # background watchdog races with subsequent CUDA primary-context teardown
    # and aborts with "CUDA driver error: unknown error". Leaking the per-case
    # cp_group across the sweep is fine in practice (each case allocates
    # `world_size / cp_size` extra communicators, which is tiny).
    del module


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace RingAttention wrapper (multi-case sweep)")
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument(
        "--cases",
        default="8192:16",
        help='Space-separated "seq:cp" pairs, e.g. "4096:2 8192:8 16384:16"',
    )
    parser.add_argument("--num-seqs", type=int, default=16)
    parser.add_argument("--topology", choices=["consecutive", "strided"], default="consecutive")
    parser.add_argument("--n-heads", type=int, default=28)
    parser.add_argument("--n-kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--active", type=int, default=3)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--profile-ranks", type=int, nargs="+", default=[0, 8, 15])
    parser.add_argument("--record-shapes", action="store_true")
    parser.add_argument("--with-stack", action="store_true")
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
    # Hard fail early if GALVATRON_RING_TRACE_PER_STEP wasn't picked up.
    if not getattr(attn_impl, "_RING_TRACE_PER_STEP", False):
        if rank == 0:
            print(
                "[Trace][warn] attention_impl was imported without "
                "GALVATRON_RING_TRACE_PER_STEP=1; per-step markers will be missing.",
                file=sys.stderr,
            )

    for seq_len, cp_size in cases:
        if rank == 0:
            print(f"[Trace] === case seq={seq_len} cp={cp_size} ===", flush=True)
        run_case(
            attn_impl=attn_impl,
            rank=rank,
            world_size=world_size,
            device=device,
            dtype=dtype,
            seq_len=seq_len,
            cp_size=cp_size,
            num_seqs=args.num_seqs,
            topology=args.topology,
            n_heads=args.n_heads,
            n_kv_heads=args.n_kv_heads,
            head_dim=args.head_dim,
            warmup=args.warmup,
            active=args.active,
            trace_dir=args.trace_dir,
            profile_ranks=args.profile_ranks,
            record_shapes=args.record_shapes,
            with_stack=args.with_stack,
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
