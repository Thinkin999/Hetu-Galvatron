#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
All-to-All Communication Profiling for AdaCPSP (Ulysses SP)

Profile all-to-all collective bandwidth for various SP group sizes.
This is used to parameterize the AdaCPSP cost model's Ulysses communication time.

Usage:
    torchrun --nproc_per_node=8 profile_alltoall.py \
        --hidden_size 4096 --num_attention_heads 32 --num_layers 32

Output:
    JSON file with alltoall bandwidth (GB/s) for each sp_size.
"""

import os
import json
import argparse
import time
from datetime import datetime

import torch
import torch.distributed as dist


def profile_alltoall(
    sp_group,
    sp_size: int,
    hidden_size: int,
    num_heads: int,
    num_layers: int,
    seq_lengths: list,
    warmup_iters: int = 5,
    profile_iters: int = 20,
    dtype=torch.bfloat16,
):
    """
    Profile all-to-all communication for Ulysses SP.

    In Ulysses SP, all-to-all is used to redistribute Q, K, V tensors:
      - Scatter: (seq_len, batch, num_heads, head_dim) → each rank gets (seq_len/sp, batch, num_heads, head_dim)
        but across heads: each rank gets all seq tokens for num_heads/sp heads
      - Gather: reverse after attention

    The message size per all-to-all for one QKV set (forward only):
      msg_bytes = seq_len * hidden_size * 2 (bf16) * 3 (Q,K,V) * 2 (scatter+gather)
    For both forward and backward: multiply by 3 (fwd + 2x bwd)

    But for cost modeling we measure raw bandwidth at various sizes.
    """
    rank = dist.get_rank()
    local_rank = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{local_rank}")

    head_dim = hidden_size // num_heads
    heads_per_rank = num_heads // sp_size
    bytes_per_element = 2 if dtype == torch.bfloat16 else 4

    results = []

    for seq_len in seq_lengths:
        # Simulate the all-to-all tensor shape:
        # Each rank holds: (seq_len/sp_size, batch=1, num_heads, head_dim)
        # After all-to-all: (seq_len, batch=1, heads_per_rank, head_dim)
        # We profile the scatter direction
        local_seq = seq_len // sp_size
        # Input chunks: sp_size chunks each of shape (local_seq, 1, heads_per_rank, head_dim)
        input_tensor = torch.randn(
            local_seq * sp_size, 1, heads_per_rank, head_dim,
            dtype=dtype, device=device
        )
        output_tensor = torch.empty_like(input_tensor)

        # Split into sp_size chunks for all_to_all
        input_chunks = list(input_tensor.chunk(sp_size, dim=0))
        output_chunks = list(output_tensor.chunk(sp_size, dim=0))

        # Warmup
        for _ in range(warmup_iters):
            dist.all_to_all(output_chunks, input_chunks, group=sp_group)
        torch.cuda.synchronize()

        # Profile
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(profile_iters):
            dist.all_to_all(output_chunks, input_chunks, group=sp_group)
        end_event.record()
        torch.cuda.synchronize()

        elapsed_ms = start_event.elapsed_time(end_event) / profile_iters

        # Calculate bandwidth
        # Total data moved: each rank sends (sp_size-1)/sp_size of its data
        total_elements = input_tensor.numel()
        total_bytes = total_elements * bytes_per_element
        data_moved = total_bytes * (sp_size - 1) / sp_size  # effective data moved
        bandwidth_gbs = data_moved / (elapsed_ms / 1000) / 1e9  # GB/s (bus bandwidth)

        results.append({
            "seq_len": seq_len,
            "local_seq": local_seq,
            "tensor_shape": list(input_tensor.shape),
            "total_bytes_MB": total_bytes / 1024 / 1024,
            "time_ms": elapsed_ms,
            "bandwidth_GBs": bandwidth_gbs,
        })

        if rank == 0:
            print(f"  sp={sp_size}, seq={seq_len:>7}: {elapsed_ms:.4f} ms, BW={bandwidth_gbs:.2f} GB/s")

        del input_tensor, output_tensor, input_chunks, output_chunks
        torch.cuda.empty_cache()

    return results


def profile_alltoall_single_tensor(
    sp_group,
    sp_size: int,
    message_sizes_mb: list,
    warmup_iters: int = 5,
    profile_iters: int = 20,
    dtype=torch.bfloat16,
):
    """
    Profile all-to-all with raw tensor sizes (not tied to model config).
    Useful for measuring pure communication bandwidth.
    """
    rank = dist.get_rank()
    local_rank = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{local_rank}")
    bytes_per_element = 2 if dtype == torch.bfloat16 else 4

    results = []

    for msg_mb in message_sizes_mb:
        num_elements = int(msg_mb * 1024 * 1024 / bytes_per_element)
        # Make divisible by sp_size
        num_elements = (num_elements // sp_size) * sp_size

        input_tensor = torch.randn(num_elements, dtype=dtype, device=device)
        output_tensor = torch.empty_like(input_tensor)

        input_chunks = list(input_tensor.chunk(sp_size))
        output_chunks = list(output_tensor.chunk(sp_size))

        # Warmup
        for _ in range(warmup_iters):
            dist.all_to_all(output_chunks, input_chunks, group=sp_group)
        torch.cuda.synchronize()

        # Profile
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(profile_iters):
            dist.all_to_all(output_chunks, input_chunks, group=sp_group)
        end_event.record()
        torch.cuda.synchronize()

        elapsed_ms = start_event.elapsed_time(end_event) / profile_iters

        total_bytes = num_elements * bytes_per_element
        data_moved = total_bytes * (sp_size - 1) / sp_size
        bandwidth_gbs = data_moved / (elapsed_ms / 1000) / 1e9

        results.append({
            "msg_size_MB": msg_mb,
            "actual_bytes": total_bytes,
            "time_ms": elapsed_ms,
            "bandwidth_GBs": bandwidth_gbs,
        })

        if rank == 0:
            print(f"  sp={sp_size}, msg={msg_mb:>6.1f} MB: {elapsed_ms:.4f} ms, BW={bandwidth_gbs:.2f} GB/s")

        del input_tensor, output_tensor, input_chunks, output_chunks
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="All-to-All Communication Profiling for AdaCPSP")
    parser.add_argument("--hidden_size", type=int, default=4096)
    parser.add_argument("--num_attention_heads", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--save_dir", type=str, default="./configs")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=-1)  # for torch.distributed.launch
    parser.add_argument("--mode", type=str, default="both", choices=["model", "raw", "both"],
                        help="model: profile with model-specific tensor shapes; "
                             "raw: profile with raw message sizes; both: do both")
    args, _ = parser.parse_known_args()

    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    if rank == 0:
        print("=" * 70)
        print(" All-to-All Communication Profiler for AdaCPSP")
        print("=" * 70)
        print(f" World size: {world_size}")
        print(f" Hidden size: {args.hidden_size}, Heads: {args.num_attention_heads}, Layers: {args.num_layers}")
        print("=" * 70)

    # Create SP groups for each power-of-2 size
    all_results = {}
    sp_size = 2
    while sp_size <= world_size:
        num_groups = world_size // sp_size
        # Create groups: consecutive ranks form a group
        for g in range(num_groups):
            group_ranks = list(range(g * sp_size, (g + 1) * sp_size))
            group = dist.new_group(ranks=group_ranks)
            if rank in group_ranks:
                my_group = group

        if rank == 0:
            print(f"\n--- Profiling All-to-All with sp_size={sp_size} ({num_groups} groups) ---")

        sp_results = {}

        if args.mode in ["model", "both"]:
            # Model-specific profiling: simulate actual Ulysses all-to-all shapes
            seq_lengths = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
            seq_lengths = [s for s in seq_lengths if s >= sp_size * 2]  # need at least 2 tokens per rank

            if rank == 0:
                print(f"  [Model-specific profiling]")
            model_results = profile_alltoall(
                my_group, sp_size,
                args.hidden_size, args.num_attention_heads, args.num_layers,
                seq_lengths,
                warmup_iters=args.warmup,
                profile_iters=args.iters,
            )
            sp_results["model"] = model_results

        if args.mode in ["raw", "both"]:
            # Raw message size profiling
            msg_sizes_mb = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

            if rank == 0:
                print(f"  [Raw message size profiling]")
            raw_results = profile_alltoall_single_tensor(
                my_group, sp_size,
                msg_sizes_mb,
                warmup_iters=args.warmup,
                profile_iters=args.iters,
            )
            sp_results["raw"] = raw_results

        # Compute summary bandwidth (average of raw results for messages >= 16MB)
        if "raw" in sp_results:
            large_msg_bw = [r["bandwidth_GBs"] for r in sp_results["raw"] if r["msg_size_MB"] >= 16]
            summary_bw = sum(large_msg_bw) / len(large_msg_bw) if large_msg_bw else 0
        elif "model" in sp_results:
            bws = [r["bandwidth_GBs"] for r in sp_results["model"]]
            summary_bw = sum(bws) / len(bws) if bws else 0
        else:
            summary_bw = 0

        sp_results["summary_bandwidth_GBs"] = summary_bw
        all_results[sp_size] = sp_results

        if rank == 0:
            print(f"  → Summary bandwidth for sp={sp_size}: {summary_bw:.2f} GB/s")

        dist.barrier()
        sp_size *= 2

    # Save results
    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = {
            "type": "alltoall_profiling",
            "world_size": world_size,
            "hidden_size": args.hidden_size,
            "num_attention_heads": args.num_attention_heads,
            "num_layers": args.num_layers,
            "timestamp": timestamp,
            "results": {str(k): v for k, v in all_results.items()},
            # Summary dict for cost model: sp_size -> bandwidth in GB/s
            "bandwidth_dict_GBs": {str(k): v["summary_bandwidth_GBs"] for k, v in all_results.items()},
        }
        # Add sp=1 (no communication)
        output["bandwidth_dict_GBs"]["1"] = 1e10

        save_path = os.path.join(args.save_dir, f"alltoall_profile_{world_size}gpus_{timestamp}.json")
        with open(save_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\n{'=' * 70}")
        print(f" Results saved to: {save_path}")
        print(f"\n Bandwidth Summary (for cost model):")
        for sp_str, bw in output["bandwidth_dict_GBs"].items():
            bw_display = f"{bw:.2f}" if bw < 1e9 else "inf (no comm)"
            print(f"   sp_size={sp_str:>3}: {bw_display} GB/s")
        print(f"{'=' * 70}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

