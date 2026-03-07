#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Overlap & Forward/Backward Profiling for AdaCPSP Cost Model
=============================================================

Measures:
  1. Flash Attention forward vs backward ratio (bwd_fwd_ratio)
  2. Ring Attention overlap: per-step compute vs per-step comm vs end-to-end
     → derive overlap effectiveness
  3. Ulysses All-to-All: measure actual blocking vs pipelining behavior

Usage:
  torchrun --nproc_per_node=8 profile_overlap.py \
      --n_heads 32 --n_kv_heads 32 --head_dim 128 --hidden_size 4096

Output:
  JSON with overlap factors + fwd/bwd ratio for cost model integration
"""

import os
import json
import argparse
import math
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import torch
import torch.distributed as dist

# ─── Flash Attention imports ──────────────────────────────────────────
HAS_FLASH_ATTN = False
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
    HAS_FLASH_ATTN = True
except ImportError:
    flash_attn_func = None
    flash_attn_varlen_func = None
    _flash_attn_forward = None
    _flash_attn_backward = None


# ═══════════════════════════════════════════════════════════════════════
# PART 1: Forward vs Backward Flash Attention Profiling
# ═══════════════════════════════════════════════════════════════════════

def profile_fwd_bwd_attention(
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    seq_lengths: List[int],
    warmup: int = 5,
    iters: int = 20,
    device: str = "cuda",
    dtype=torch.bfloat16,
) -> List[Dict]:
    """
    Measure forward and backward flash attention times separately.
    Returns list of dicts with fwd_ms, bwd_ms, bwd_fwd_ratio per seq_len.
    
    Key insight: backward ≈ 2-2.5x forward for FlashAttention-2.
    This ratio is important for accurate overlap modeling in Ring Attention,
    where backward ring steps have more compute per step.
    """
    if not HAS_FLASH_ATTN:
        print("[WARN] flash_attn not available, skipping fwd/bwd profiling")
        return []

    results = []
    print(f"\n[Fwd/Bwd Profiling] n_heads={n_heads}, n_kv_heads={n_kv_heads}, "
          f"head_dim={head_dim}")

    for seq_len in seq_lengths:
        try:
            # Create tensors with gradient tracking for backward
            q = torch.randn(1, seq_len, n_heads, head_dim,
                            dtype=dtype, device=device, requires_grad=True)
            k = torch.randn(1, seq_len, n_kv_heads, head_dim,
                            dtype=dtype, device=device, requires_grad=True)
            v = torch.randn(1, seq_len, n_kv_heads, head_dim,
                            dtype=dtype, device=device, requires_grad=True)

            # ── Forward profiling ──
            for _ in range(warmup):
                out = flash_attn_func(q, k, v, causal=True)
            torch.cuda.synchronize()

            fwd_start = torch.cuda.Event(enable_timing=True)
            fwd_end = torch.cuda.Event(enable_timing=True)
            fwd_start.record()
            for _ in range(iters):
                out = flash_attn_func(q, k, v, causal=True)
            fwd_end.record()
            torch.cuda.synchronize()
            fwd_ms = fwd_start.elapsed_time(fwd_end) / iters

            # ── Backward profiling ──
            # Need a fresh forward pass for each backward
            grad_out = torch.randn_like(out)

            # Warmup backward
            for _ in range(warmup):
                out = flash_attn_func(q, k, v, causal=True)
                out.backward(grad_out, retain_graph=False)
                q.grad = None
                k.grad = None
                v.grad = None

            torch.cuda.synchronize()

            bwd_start = torch.cuda.Event(enable_timing=True)
            bwd_end = torch.cuda.Event(enable_timing=True)
            bwd_start.record()
            for _ in range(iters):
                out = flash_attn_func(q, k, v, causal=True)
                out.backward(grad_out, retain_graph=False)
                q.grad = None
                k.grad = None
                v.grad = None
            bwd_end.record()
            torch.cuda.synchronize()
            # bwd_total includes fwd + bwd; subtract fwd to get pure bwd
            bwd_total_ms = bwd_start.elapsed_time(bwd_end) / iters
            bwd_ms = bwd_total_ms - fwd_ms

            ratio = bwd_ms / fwd_ms if fwd_ms > 0 else 0

            results.append({
                "seq_len": seq_len,
                "fwd_ms": fwd_ms,
                "bwd_ms": bwd_ms,
                "fwd_bwd_total_ms": bwd_total_ms,
                "bwd_fwd_ratio": ratio,
            })
            print(f"  seq={seq_len:>6}: fwd={fwd_ms:.4f}ms, bwd={bwd_ms:.4f}ms, "
                  f"ratio={ratio:.3f}")

            del q, k, v, out, grad_out
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  seq={seq_len}: OOM")
                torch.cuda.empty_cache()
                break
            raise

    if results:
        ratios = [r["bwd_fwd_ratio"] for r in results]
        avg_ratio = np.mean(ratios)
        print(f"\n  Average bwd/fwd ratio: {avg_ratio:.3f}")

    return results


# ═══════════════════════════════════════════════════════════════════════
# PART 2: Ring Attention Per-Step Overlap Measurement
# ═══════════════════════════════════════════════════════════════════════

def profile_ring_overlap(
    cp_group,
    cp_size: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    seq_lengths: List[int],
    warmup: int = 5,
    iters: int = 10,
    dtype=torch.bfloat16,
) -> List[Dict]:
    """
    For each sequence length, measure:
      1. compute_per_step: pure flash attention on local chunk (no comm)
      2. comm_per_step: pure P2P send/recv of KV (no compute)
      3. overlapped_per_step: realistic ring step (compute + overlapped comm)
    
    The overlap effectiveness = (compute + comm - overlapped) / min(compute, comm)
    Theoretically: overlapped ≈ max(compute, comm) if perfect overlap.
    
    Ring Attention forward pattern:
      for step in range(cp_size):
          if step + 1 != cp_size:
              next_k, next_v = comm.send_recv_kv(k, v)  # non-blocking
          flash_attn_forward(q, k, v)                     # compute
          if step + 1 != cp_size:
              comm.wait()                                  # wait for comm
              k, v = next_k, next_v
    
    So comm and compute run concurrently on different CUDA streams (P2P uses NCCL
    stream, flash_attn uses default stream). Overlap depends on whether the NCCL
    transfer finishes before flash_attn completes.
    """
    rank = dist.get_rank()
    local_rank = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{local_rank}")
    cp_rank = dist.get_rank(cp_group)
    bpe = 2 if dtype == torch.bfloat16 else 4

    next_cp_rank = dist.get_global_rank(cp_group, (cp_rank + 1) % cp_size)
    prev_cp_rank = dist.get_global_rank(cp_group, (cp_rank - 1) % cp_size)

    results = []

    for seq_len in seq_lengths:
        local_seq = seq_len // cp_size
        if local_seq < 128:
            continue

        # Allocate tensors
        q = torch.randn(local_seq, n_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(local_seq, n_kv_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(local_seq, n_kv_heads, head_dim, dtype=dtype, device=device)
        recv_k = torch.empty_like(k)
        recv_v = torch.empty_like(v)
        cu_seqlens = torch.tensor([0, local_seq], dtype=torch.int32, device=device)

        kv_bytes = k.numel() * bpe + v.numel() * bpe
        kv_mb = kv_bytes / 1024 / 1024

        # ── Measure 1: Pure compute (flash_attn on local chunk) ──
        if flash_attn_varlen_func is not None:
            for _ in range(warmup):
                flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens,
                                       local_seq, local_seq, causal=True)
            torch.cuda.synchronize()

            se = torch.cuda.Event(enable_timing=True)
            ee = torch.cuda.Event(enable_timing=True)
            se.record()
            for _ in range(iters):
                flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens,
                                       local_seq, local_seq, causal=True)
            ee.record()
            torch.cuda.synchronize()
            compute_ms = se.elapsed_time(ee) / iters
        else:
            compute_ms = 0.0
            if rank == 0:
                print(f"  [WARN] flash_attn_varlen_func not available")

        # ── Measure 2: Pure comm (P2P ring step, KV transfer) ──
        def do_p2p_step():
            ops = [
                dist.P2POp(dist.isend, k, next_cp_rank, group=cp_group),
                dist.P2POp(dist.isend, v, next_cp_rank, group=cp_group),
                dist.P2POp(dist.irecv, recv_k, prev_cp_rank, group=cp_group),
                dist.P2POp(dist.irecv, recv_v, prev_cp_rank, group=cp_group),
            ]
            reqs = dist.batch_isend_irecv(ops)
            for r in reqs:
                r.wait()

        for _ in range(warmup):
            do_p2p_step()
        torch.cuda.synchronize()
        dist.barrier(group=cp_group)

        se = torch.cuda.Event(enable_timing=True)
        ee = torch.cuda.Event(enable_timing=True)
        se.record()
        for _ in range(iters):
            do_p2p_step()
        ee.record()
        torch.cuda.synchronize()
        comm_ms = se.elapsed_time(ee) / iters

        # ── Measure 3: Overlapped step (comm + compute, mimic ring attention) ──
        def do_overlapped_step():
            # Launch non-blocking comm
            ops = [
                dist.P2POp(dist.isend, k, next_cp_rank, group=cp_group),
                dist.P2POp(dist.isend, v, next_cp_rank, group=cp_group),
                dist.P2POp(dist.irecv, recv_k, prev_cp_rank, group=cp_group),
                dist.P2POp(dist.irecv, recv_v, prev_cp_rank, group=cp_group),
            ]
            reqs = dist.batch_isend_irecv(ops)
            # Compute while comm is in flight
            if flash_attn_varlen_func is not None:
                flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens,
                                       local_seq, local_seq, causal=True)
            # Wait for comm
            for r in reqs:
                r.wait()

        for _ in range(warmup):
            do_overlapped_step()
        torch.cuda.synchronize()
        dist.barrier(group=cp_group)

        se = torch.cuda.Event(enable_timing=True)
        ee = torch.cuda.Event(enable_timing=True)
        se.record()
        for _ in range(iters):
            do_overlapped_step()
        ee.record()
        torch.cuda.synchronize()
        overlapped_ms = se.elapsed_time(ee) / iters

        # ── Derive overlap metrics ──
        theoretical_no_overlap = compute_ms + comm_ms
        ideal_overlap = max(compute_ms, comm_ms)

        # Overlap savings: how much of the additive cost is saved
        saved_ms = theoretical_no_overlap - overlapped_ms
        # Overlap efficiency: 1.0 = perfect overlap, 0.0 = no overlap
        max_possible_savings = min(compute_ms, comm_ms)
        overlap_efficiency = saved_ms / max_possible_savings if max_possible_savings > 0 else 0
        overlap_efficiency = max(0, min(1.0, overlap_efficiency))

        # Practical overlap factor: fraction of comm hidden behind compute
        # effective_time = compute + (1 - overlap_factor) * comm
        # → overlap_factor = (compute + comm - overlapped) / comm
        overlap_factor = saved_ms / comm_ms if comm_ms > 0 else 0
        overlap_factor = max(0, min(1.0, overlap_factor))

        result = {
            "seq_len": seq_len,
            "local_seq": local_seq,
            "cp_size": cp_size,
            "compute_ms": compute_ms,
            "comm_ms": comm_ms,
            "overlapped_ms": overlapped_ms,
            "no_overlap_ms": theoretical_no_overlap,
            "ideal_overlap_ms": ideal_overlap,
            "overlap_efficiency": overlap_efficiency,
            "overlap_factor": overlap_factor,
            "kv_mb": kv_mb,
            "compute_dominant": compute_ms > comm_ms,
        }
        results.append(result)

        if rank == 0:
            status = "compute>comm" if compute_ms > comm_ms else "comm>compute"
            print(f"  cp={cp_size} seq={seq_len:>6} (local={local_seq:>5}): "
                  f"compute={compute_ms:.3f}ms, comm={comm_ms:.3f}ms, "
                  f"overlap={overlapped_ms:.3f}ms, "
                  f"η={overlap_efficiency:.2f}, factor={overlap_factor:.2f} [{status}]")

        del q, k, v, recv_k, recv_v, cu_seqlens
        torch.cuda.empty_cache()

    return results


# ═══════════════════════════════════════════════════════════════════════
# PART 3: Ring Attention End-to-End Profiling (full forward pass)
# ═══════════════════════════════════════════════════════════════════════

def profile_ring_e2e(
    cp_group,
    cp_size: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    seq_lengths: List[int],
    warmup: int = 3,
    iters: int = 5,
    dtype=torch.bfloat16,
) -> List[Dict]:
    """
    Measure full Ring Attention forward pass end-to-end.
    This gives us the actual time for a complete ring pass of cp_size steps,
    which we can compare against the cost model's prediction.
    
    We implement the standard ring attention loop (without zigzag for simplicity).
    """
    if not HAS_FLASH_ATTN:
        return []

    rank = dist.get_rank()
    local_rank = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{local_rank}")
    cp_rank = dist.get_rank(cp_group)
    bpe = 2 if dtype == torch.bfloat16 else 4

    next_cp_rank = dist.get_global_rank(cp_group, (cp_rank + 1) % cp_size)
    prev_cp_rank = dist.get_global_rank(cp_group, (cp_rank - 1) % cp_size)

    results = []

    for seq_len in seq_lengths:
        local_seq = seq_len // cp_size
        if local_seq < 128:
            continue

        q = torch.randn(local_seq, n_heads, head_dim, dtype=dtype, device=device)
        k_orig = torch.randn(local_seq, n_kv_heads, head_dim, dtype=dtype, device=device)
        v_orig = torch.randn(local_seq, n_kv_heads, head_dim, dtype=dtype, device=device)
        cu = torch.tensor([0, local_seq], dtype=torch.int32, device=device)

        def ring_forward():
            """One full ring attention forward pass."""
            k = k_orig.clone()
            v = v_orig.clone()
            recv_k = torch.empty_like(k)
            recv_v = torch.empty_like(v)

            for step in range(cp_size):
                # Start non-blocking comm (except last step)
                reqs = None
                if step + 1 != cp_size:
                    ops = [
                        dist.P2POp(dist.isend, k, next_cp_rank, group=cp_group),
                        dist.P2POp(dist.isend, v, next_cp_rank, group=cp_group),
                        dist.P2POp(dist.irecv, recv_k, prev_cp_rank, group=cp_group),
                        dist.P2POp(dist.irecv, recv_v, prev_cp_rank, group=cp_group),
                    ]
                    reqs = dist.batch_isend_irecv(ops)

                # Compute attention (always compute for simplicity - ignoring causal skip)
                flash_attn_varlen_func(
                    q, k, v, cu, cu, local_seq, local_seq,
                    causal=(step == 0),  # only first step is causal
                )

                # Wait for comm
                if reqs is not None:
                    for r in reqs:
                        r.wait()
                    k = recv_k.clone()
                    v = recv_v.clone()

        # Warmup
        for _ in range(warmup):
            ring_forward()
        torch.cuda.synchronize()
        dist.barrier(group=cp_group)

        # Profile
        se = torch.cuda.Event(enable_timing=True)
        ee = torch.cuda.Event(enable_timing=True)
        se.record()
        for _ in range(iters):
            ring_forward()
        ee.record()
        torch.cuda.synchronize()
        ring_e2e_ms = se.elapsed_time(ee) / iters

        # Also measure pure compute for all cp_size steps (no comm)
        se.record()
        for _ in range(iters):
            for _ in range(cp_size):
                flash_attn_varlen_func(
                    q, k_orig, v_orig, cu, cu, local_seq, local_seq, causal=True)
        ee.record()
        torch.cuda.synchronize()
        pure_compute_ms = se.elapsed_time(ee) / iters

        result = {
            "seq_len": seq_len,
            "local_seq": local_seq,
            "cp_size": cp_size,
            "ring_e2e_ms": ring_e2e_ms,
            "pure_compute_all_steps_ms": pure_compute_ms,
            "overhead_ms": ring_e2e_ms - pure_compute_ms,
            "overhead_ratio": (ring_e2e_ms - pure_compute_ms) / pure_compute_ms if pure_compute_ms > 0 else 0,
        }
        results.append(result)

        if rank == 0:
            print(f"  cp={cp_size} seq={seq_len:>6}: "
                  f"ring_e2e={ring_e2e_ms:.3f}ms, "
                  f"pure_compute={pure_compute_ms:.3f}ms, "
                  f"overhead={result['overhead_ratio']*100:.1f}%")

        del q, k_orig, v_orig, cu
        torch.cuda.empty_cache()

    return results


# ═══════════════════════════════════════════════════════════════════════
# PART 4: Backward Ring Attention Comm Profiling
# ═══════════════════════════════════════════════════════════════════════

def profile_ring_bwd_comm(
    cp_group,
    cp_size: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    seq_lengths: List[int],
    warmup: int = 5,
    iters: int = 10,
    dtype=torch.bfloat16,
) -> List[Dict]:
    """
    Backward ring attention involves TWO rings:
      1. kv_ring: send K,V to next, recv from prev (same as forward)
      2. dkv_ring: send dK,dV to prev, recv from next (reverse direction)
    
    Both run concurrently. This doubles the communication load.
    Measure the actual backward comm time for one ring step.
    """
    rank = dist.get_rank()
    local_rank = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{local_rank}")
    cp_rank = dist.get_rank(cp_group)
    bpe = 2 if dtype == torch.bfloat16 else 4

    next_r = dist.get_global_rank(cp_group, (cp_rank + 1) % cp_size)
    prev_r = dist.get_global_rank(cp_group, (cp_rank - 1) % cp_size)

    results = []

    for seq_len in seq_lengths:
        local_seq = seq_len // cp_size
        if local_seq < 128:
            continue

        # KV tensors (forward ring)
        k = torch.randn(local_seq, n_kv_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(local_seq, n_kv_heads, head_dim, dtype=dtype, device=device)
        recv_k = torch.empty_like(k)
        recv_v = torch.empty_like(v)
        # dKV tensors (backward ring, reverse direction)
        dk = torch.randn(local_seq, n_kv_heads, head_dim, dtype=dtype, device=device)
        dv = torch.randn(local_seq, n_kv_heads, head_dim, dtype=dtype, device=device)
        recv_dk = torch.empty_like(dk)
        recv_dv = torch.empty_like(dv)

        kv_bytes = k.numel() * bpe * 2  # K + V

        def do_bwd_step():
            """One backward ring step: forward KV ring + reverse dKV ring."""
            ops = [
                # Forward KV ring
                dist.P2POp(dist.isend, k, next_r, group=cp_group),
                dist.P2POp(dist.isend, v, next_r, group=cp_group),
                dist.P2POp(dist.irecv, recv_k, prev_r, group=cp_group),
                dist.P2POp(dist.irecv, recv_v, prev_r, group=cp_group),
                # Reverse dKV ring
                dist.P2POp(dist.isend, dk, prev_r, group=cp_group),
                dist.P2POp(dist.isend, dv, prev_r, group=cp_group),
                dist.P2POp(dist.irecv, recv_dk, next_r, group=cp_group),
                dist.P2POp(dist.irecv, recv_dv, next_r, group=cp_group),
            ]
            reqs = dist.batch_isend_irecv(ops)
            for r in reqs:
                r.wait()

        for _ in range(warmup):
            do_bwd_step()
        torch.cuda.synchronize()
        dist.barrier(group=cp_group)

        se = torch.cuda.Event(enable_timing=True)
        ee = torch.cuda.Event(enable_timing=True)
        se.record()
        for _ in range(iters):
            do_bwd_step()
        ee.record()
        torch.cuda.synchronize()
        bwd_comm_ms = se.elapsed_time(ee) / iters

        # Also measure single-direction (fwd-only) for comparison
        def do_fwd_step():
            ops = [
                dist.P2POp(dist.isend, k, next_r, group=cp_group),
                dist.P2POp(dist.isend, v, next_r, group=cp_group),
                dist.P2POp(dist.irecv, recv_k, prev_r, group=cp_group),
                dist.P2POp(dist.irecv, recv_v, prev_r, group=cp_group),
            ]
            reqs = dist.batch_isend_irecv(ops)
            for r in reqs:
                r.wait()

        for _ in range(warmup):
            do_fwd_step()
        torch.cuda.synchronize()
        dist.barrier(group=cp_group)

        se.record()
        for _ in range(iters):
            do_fwd_step()
        ee.record()
        torch.cuda.synchronize()
        fwd_comm_ms = se.elapsed_time(ee) / iters

        result = {
            "seq_len": seq_len,
            "local_seq": local_seq,
            "cp_size": cp_size,
            "fwd_comm_ms": fwd_comm_ms,
            "bwd_comm_ms": bwd_comm_ms,
            "bwd_fwd_comm_ratio": bwd_comm_ms / fwd_comm_ms if fwd_comm_ms > 0 else 0,
            "kv_bytes": kv_bytes,
        }
        results.append(result)

        if rank == 0:
            print(f"  cp={cp_size} seq={seq_len:>6}: "
                  f"fwd_comm={fwd_comm_ms:.3f}ms, bwd_comm={bwd_comm_ms:.3f}ms, "
                  f"ratio={result['bwd_fwd_comm_ratio']:.2f}x")

        del k, v, dk, dv, recv_k, recv_v, recv_dk, recv_dv
        torch.cuda.empty_cache()

    return results


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Overlap & Fwd/Bwd Profiling for AdaCPSP Cost Model")
    parser.add_argument("--n_heads", type=int, default=32)
    parser.add_argument("--n_kv_heads", type=int, default=32)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="./configs")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=-1)
    parser.add_argument("--mode", type=str, default="all",
                        choices=["fwd_bwd", "ring_overlap", "ring_e2e",
                                 "ring_bwd_comm", "all"])
    args, _ = parser.parse_known_args()

    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    if rank == 0:
        print("=" * 80)
        print(" Overlap & Forward/Backward Profiler for AdaCPSP")
        print("=" * 80)
        print(f" World size: {world_size}")
        print(f" Config: n_heads={args.n_heads}, n_kv_heads={args.n_kv_heads}, "
              f"head_dim={args.head_dim}, hidden_size={args.hidden_size}")
        print("=" * 80)

    all_output = {
        "type": "overlap_profiling",
        "world_size": world_size,
        "config": {
            "n_heads": args.n_heads,
            "n_kv_heads": args.n_kv_heads,
            "head_dim": args.head_dim,
            "hidden_size": args.hidden_size,
        },
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    # ═══ PART 1: Forward vs Backward Attention ═══
    if args.mode in ["fwd_bwd", "all"]:
        if rank == 0:
            print(f"\n{'='*80}")
            print(" PART 1: Forward vs Backward Flash Attention")
            print(f"{'='*80}")

            seq_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768]
            fwd_bwd_results = profile_fwd_bwd_attention(
                args.n_heads, args.n_kv_heads, args.head_dim,
                seq_lengths,
                warmup=args.warmup, iters=max(args.iters, 20),
            )

            if fwd_bwd_results:
                ratios = [r["bwd_fwd_ratio"] for r in fwd_bwd_results]
                all_output["fwd_bwd"] = {
                    "results": fwd_bwd_results,
                    "avg_bwd_fwd_ratio": float(np.mean(ratios)),
                    "median_bwd_fwd_ratio": float(np.median(ratios)),
                    # Ratio for large seqs (more representative)
                    "large_seq_ratio": float(np.mean([
                        r["bwd_fwd_ratio"] for r in fwd_bwd_results
                        if r["seq_len"] >= 4096
                    ])) if any(r["seq_len"] >= 4096 for r in fwd_bwd_results) else None,
                }

                print(f"\n  → Recommended bwd_fwd_ratio for cost model: "
                      f"{all_output['fwd_bwd']['avg_bwd_fwd_ratio']:.3f}")

        dist.barrier()

    # ═══ PART 2: Ring Attention Overlap ═══
    if args.mode in ["ring_overlap", "all"]:
        if rank == 0:
            print(f"\n{'='*80}")
            print(" PART 2: Ring Attention Per-Step Overlap Measurement")
            print(f"{'='*80}")

        ring_overlap_results = {}

        for cp_size in [2, 4, 8]:
            if cp_size > world_size:
                continue

            # Create CP group
            num_groups = world_size // cp_size
            my_group = None
            for g in range(num_groups):
                group_ranks = list(range(g * cp_size, (g + 1) * cp_size))
                group = dist.new_group(ranks=group_ranks)
                if rank in group_ranks:
                    my_group = group

            if rank == 0:
                print(f"\n--- cp_size={cp_size} ---")

            # Sequence lengths where local_seq >= 128
            seq_lengths = [s for s in [2048, 4096, 8192, 16384, 32768, 65536]
                           if s // cp_size >= 128]

            results = profile_ring_overlap(
                my_group, cp_size,
                args.n_heads, args.n_kv_heads, args.head_dim,
                seq_lengths,
                warmup=args.warmup, iters=args.iters,
            )
            ring_overlap_results[cp_size] = results
            dist.barrier()

        if rank == 0:
            # Compute summary statistics
            summary = {}
            for cp_size, results in ring_overlap_results.items():
                if results:
                    factors = [r["overlap_factor"] for r in results]
                    efficiencies = [r["overlap_efficiency"] for r in results]
                    summary[cp_size] = {
                        "avg_overlap_factor": float(np.mean(factors)),
                        "avg_overlap_efficiency": float(np.mean(efficiencies)),
                        "compute_dominant_pct": float(
                            np.mean([1 if r["compute_dominant"] else 0 for r in results]) * 100),
                    }

            all_output["ring_overlap"] = {
                "results": {str(k): v for k, v in ring_overlap_results.items()},
                "summary": {str(k): v for k, v in summary.items()},
            }

            print(f"\n--- Ring Overlap Summary ---")
            for cp_size, s in summary.items():
                print(f"  cp={cp_size}: avg_factor={s['avg_overlap_factor']:.3f}, "
                      f"avg_efficiency={s['avg_overlap_efficiency']:.3f}, "
                      f"compute_dominant={s['compute_dominant_pct']:.0f}%")

    # ═══ PART 3: Ring Attention End-to-End ═══
    if args.mode in ["ring_e2e", "all"]:
        if rank == 0:
            print(f"\n{'='*80}")
            print(" PART 3: Ring Attention End-to-End (Full Forward Pass)")
            print(f"{'='*80}")

        ring_e2e_results = {}

        for cp_size in [2, 4, 8]:
            if cp_size > world_size:
                continue

            num_groups = world_size // cp_size
            my_group = None
            for g in range(num_groups):
                group_ranks = list(range(g * cp_size, (g + 1) * cp_size))
                group = dist.new_group(ranks=group_ranks)
                if rank in group_ranks:
                    my_group = group

            if rank == 0:
                print(f"\n--- cp_size={cp_size} (e2e) ---")

            seq_lengths = [s for s in [2048, 4096, 8192, 16384, 32768, 65536]
                           if s // cp_size >= 128]

            results = profile_ring_e2e(
                my_group, cp_size,
                args.n_heads, args.n_kv_heads, args.head_dim,
                seq_lengths,
                warmup=max(2, args.warmup // 2),
                iters=max(3, args.iters // 2),
            )
            ring_e2e_results[cp_size] = results
            dist.barrier()

        if rank == 0:
            all_output["ring_e2e"] = {
                str(k): v for k, v in ring_e2e_results.items()
            }

    # ═══ PART 4: Ring Backward Comm ═══
    if args.mode in ["ring_bwd_comm", "all"]:
        if rank == 0:
            print(f"\n{'='*80}")
            print(" PART 4: Ring Attention Backward Communication (Dual Ring)")
            print(f"{'='*80}")

        ring_bwd_results = {}

        for cp_size in [2, 4, 8]:
            if cp_size > world_size:
                continue

            num_groups = world_size // cp_size
            my_group = None
            for g in range(num_groups):
                group_ranks = list(range(g * cp_size, (g + 1) * cp_size))
                group = dist.new_group(ranks=group_ranks)
                if rank in group_ranks:
                    my_group = group

            if rank == 0:
                print(f"\n--- cp_size={cp_size} (bwd comm) ---")

            seq_lengths = [s for s in [2048, 4096, 8192, 16384, 32768, 65536]
                           if s // cp_size >= 128]

            results = profile_ring_bwd_comm(
                my_group, cp_size,
                args.n_heads, args.n_kv_heads, args.head_dim,
                seq_lengths,
                warmup=args.warmup, iters=args.iters,
            )
            ring_bwd_results[cp_size] = results
            dist.barrier()

        if rank == 0:
            # Summary
            for cp_size, results in ring_bwd_results.items():
                if results:
                    ratios = [r["bwd_fwd_comm_ratio"] for r in results]
                    print(f"  cp={cp_size}: avg bwd/fwd comm ratio = {np.mean(ratios):.2f}x")

            all_output["ring_bwd_comm"] = {
                "results": {str(k): v for k, v in ring_bwd_results.items()},
                "summary": {
                    str(cp): {
                        "avg_bwd_fwd_comm_ratio": float(np.mean(
                            [r["bwd_fwd_comm_ratio"] for r in res]))
                    }
                    for cp, res in ring_bwd_results.items() if res
                },
            }

    # ═══ Save results ═══
    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(args.save_dir,
                                 f"overlap_profile_{world_size}gpus_{ts}.json")
        with open(save_path, "w") as f:
            json.dump(all_output, f, indent=2, default=str)

        print(f"\n{'='*80}")
        print(f" Results saved to: {save_path}")
        print(f"{'='*80}")

        # ── Generate summary for cost model integration ──
        print(f"\n{'='*80}")
        print(" COST MODEL INTEGRATION RECOMMENDATIONS")
        print(f"{'='*80}")

        if "fwd_bwd" in all_output:
            fb = all_output["fwd_bwd"]
            print(f"\n  bwd_fwd_ratio = {fb['avg_bwd_fwd_ratio']:.3f}")
            print(f"    (Flash Attention backward is ~{fb['avg_bwd_fwd_ratio']:.1f}x forward)")

        if "ring_overlap" in all_output:
            ro = all_output["ring_overlap"]["summary"]
            print(f"\n  Ring Attention overlap factors:")
            for cp, s in ro.items():
                print(f"    cp={cp}: overlap_factor = {s['avg_overlap_factor']:.3f}")
            print(f"    → In cost model: effective_ring_time = compute + (1-factor)*comm")
            print(f"    → Or use max-based: per_step = max(compute_step, comm_step)")

        if "ring_bwd_comm" in all_output and "summary" in all_output["ring_bwd_comm"]:
            rb = all_output["ring_bwd_comm"]["summary"]
            print(f"\n  Ring backward comm multiplier:")
            for cp, s in rb.items():
                print(f"    cp={cp}: bwd_comm = {s['avg_bwd_fwd_comm_ratio']:.2f}x fwd_comm")

        print(f"\n  Recommended cost model update:")
        print(f"    Ring total_time = fwd_time + bwd_time  (per layer × L)")
        print(f"    fwd_time = (cp-1) * max(compute_step, fwd_comm_step) + compute_step")
        print(f"    bwd_time = (cp-1) * max(bwd_ratio*compute_step, bwd_comm_step) + bwd_ratio*compute_step")
        print(f"{'='*80}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

