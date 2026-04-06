#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
P2P Ring Communication Profiling for AdaCPSP (Ring Attention / Context Parallel)

Profile point-to-point ring communication bandwidth for various CP group sizes.
This is used to parameterize the AdaCPSP cost model's Ring Attention communication time.

In Ring Attention, each step involves:
  - Send KV to next rank in the ring
  - Recv KV from previous rank in the ring
  - These happen concurrently with attention computation (overlap)

Usage:
    torchrun --nproc_per_node=8 profile_p2p_ring.py \
        --hidden_size 4096 --num_attention_heads 32 --num_layers 32

Output:
    JSON file with P2P ring bandwidth (GB/s) for each cp_size.
"""

import os
import json
import argparse
from datetime import datetime

import torch
import torch.distributed as dist

from profile_topo_utils import build_group_ranks_list, linear_fit, topo_key


def profile_p2p_ring(
    cp_group,
    cp_size: int,
    hidden_size: int,
    num_kv_heads: int,
    seq_lengths: list,
    warmup_iters: int = 5,
    profile_iters: int = 20,
    dtype=torch.bfloat16,
):
    """
    Profile P2P ring communication for Ring Attention.

    In ring attention, at each step each rank sends its K,V to the next rank
    and receives K,V from the previous rank. This repeats (cp_size - 1) times.

    KV tensor per step: (seq_len / cp_size, batch=1, num_kv_heads, head_dim) * 2 (K and V)
    Total KV bytes per step = 2 * (seq_len/cp_size) * num_kv_heads * head_dim * sizeof(dtype)
    Total steps = cp_size - 1
    """
    rank = dist.get_rank()
    local_rank = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{local_rank}")

    cp_rank = dist.get_rank(cp_group)
    head_dim = hidden_size // num_kv_heads  # Use num_kv_heads for GQA
    bytes_per_element = 2 if dtype == torch.bfloat16 else 4

    next_rank = dist.get_global_rank(cp_group, (cp_rank + 1) % cp_size)
    prev_rank = dist.get_global_rank(cp_group, (cp_rank - 1) % cp_size)

    results = []

    for seq_len in seq_lengths:
        local_seq = seq_len // cp_size
        if local_seq < 1:
            continue

        # KV tensor: (local_seq, batch=1, num_kv_heads, head_dim) for K and V
        kv_shape = (local_seq, 1, num_kv_heads, head_dim)
        send_k = torch.randn(kv_shape, dtype=dtype, device=device)
        send_v = torch.randn(kv_shape, dtype=dtype, device=device)
        recv_k = torch.empty_like(send_k)
        recv_v = torch.empty_like(send_v)

        kv_bytes = send_k.numel() * bytes_per_element * 2  # K + V

        def do_ring_step():
            """Simulate one ring step: send KV to next, recv KV from prev."""
            ops = []
            ops.append(dist.P2POp(dist.isend, send_k, next_rank, group=cp_group))
            ops.append(dist.P2POp(dist.isend, send_v, next_rank, group=cp_group))
            ops.append(dist.P2POp(dist.irecv, recv_k, prev_rank, group=cp_group))
            ops.append(dist.P2POp(dist.irecv, recv_v, prev_rank, group=cp_group))
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

        # Warmup
        for _ in range(warmup_iters):
            do_ring_step()
        torch.cuda.synchronize()
        dist.barrier(group=cp_group)

        # Profile: measure time for (cp_size - 1) ring steps (full ring)
        num_steps = cp_size - 1
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(profile_iters):
            for _ in range(num_steps):
                do_ring_step()
        end_event.record()
        torch.cuda.synchronize()

        total_elapsed_ms = start_event.elapsed_time(end_event) / profile_iters
        elapsed_tensor = torch.tensor([total_elapsed_ms], device=device, dtype=torch.float64)
        dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX, group=cp_group)
        total_elapsed_ms = float(elapsed_tensor.item())
        per_step_ms = total_elapsed_ms / num_steps if num_steps > 0 else 0

        # Bandwidth: data moved per step / time per step
        bandwidth_gbs = kv_bytes / (per_step_ms / 1000) / 1e9 if per_step_ms > 0 else 0

        results.append({
            "seq_len": seq_len,
            "local_seq": local_seq,
            "kv_shape": list(kv_shape),
            "kv_bytes_per_step_MB": kv_bytes / 1024 / 1024,
            "num_ring_steps": num_steps,
            "total_ring_time_ms": total_elapsed_ms,
            "per_step_time_ms": per_step_ms,
            "bandwidth_GBs": bandwidth_gbs,
        })

        if rank == 0:
            print(f"  cp={cp_size}, seq={seq_len:>7}: total_ring={total_elapsed_ms:.4f} ms, "
                  f"per_step={per_step_ms:.4f} ms, BW={bandwidth_gbs:.2f} GB/s")

        del send_k, send_v, recv_k, recv_v
        torch.cuda.empty_cache()

    return results


def profile_p2p_raw(
    cp_group,
    cp_size: int,
    message_sizes_mb: list,
    warmup_iters: int = 5,
    profile_iters: int = 20,
    dtype=torch.bfloat16,
):
    """
    Profile raw P2P ring bandwidth with varying message sizes.
    """
    rank = dist.get_rank()
    local_rank = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{local_rank}")

    cp_rank = dist.get_rank(cp_group)
    bytes_per_element = 2 if dtype == torch.bfloat16 else 4

    next_rank = dist.get_global_rank(cp_group, (cp_rank + 1) % cp_size)
    prev_rank = dist.get_global_rank(cp_group, (cp_rank - 1) % cp_size)

    results = []

    for msg_mb in message_sizes_mb:
        num_elements = int(msg_mb * 1024 * 1024 / bytes_per_element)

        send_tensor = torch.randn(num_elements, dtype=dtype, device=device)
        recv_tensor = torch.empty_like(send_tensor)

        total_bytes = num_elements * bytes_per_element

        def do_ring_step():
            ops = [
                dist.P2POp(dist.isend, send_tensor, next_rank, group=cp_group),
                dist.P2POp(dist.irecv, recv_tensor, prev_rank, group=cp_group),
            ]
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

        # Warmup
        for _ in range(warmup_iters):
            do_ring_step()
        torch.cuda.synchronize()
        dist.barrier(group=cp_group)

        # Profile single step
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(profile_iters):
            do_ring_step()
        end_event.record()
        torch.cuda.synchronize()

        elapsed_ms = start_event.elapsed_time(end_event) / profile_iters
        elapsed_tensor = torch.tensor([elapsed_ms], device=device, dtype=torch.float64)
        dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX, group=cp_group)
        elapsed_ms = float(elapsed_tensor.item())
        bandwidth_gbs = total_bytes / (elapsed_ms / 1000) / 1e9 if elapsed_ms > 0 else 0

        results.append({
            "msg_size_MB": msg_mb,
            "total_bytes": total_bytes,
            "time_ms": elapsed_ms,
            "bandwidth_GBs": bandwidth_gbs,
        })

        if rank == 0:
            print(f"  cp={cp_size}, msg={msg_mb:>6.1f} MB: {elapsed_ms:.4f} ms, BW={bandwidth_gbs:.2f} GB/s")

        del send_tensor, recv_tensor
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="P2P Ring Communication Profiling for AdaCPSP")
    parser.add_argument("--hidden_size", type=int, default=4096)
    parser.add_argument("--num_attention_heads", type=int, default=32)
    parser.add_argument("--num_kv_heads", type=int, default=32, help="Number of KV heads (for GQA)")
    parser.add_argument("--num_layers", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--save_dir", type=str, default="./configs")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=-1)
    parser.add_argument("--mode", type=str, default="both", choices=["model", "raw", "both"])
    parser.add_argument("--topology", type=str, default="both",
                        choices=["consecutive", "strided", "both"],
                        help="Topology to profile: consecutive, strided, or both")
    parser.add_argument("--gpus_per_node", type=int, default=8,
                        help="Number of GPUs per node (for topology description)")
    args, _ = parser.parse_known_args()

    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    if rank == 0:
        print("=" * 70)
        print(" P2P Ring Communication Profiler for AdaCPSP")
        print("=" * 70)
        print(f" World size: {world_size}")
        print(f" Hidden size: {args.hidden_size}, Heads: {args.num_attention_heads}, "
              f"KV Heads: {args.num_kv_heads}, Layers: {args.num_layers}")
        print("=" * 70)

    topologies_to_profile = (
        ["consecutive", "strided"] if args.topology == "both"
        else [args.topology]
    )

    all_results = {}       # topo_key -> per-cp/topo results
    bw_dict_consec = {}    # str(cp_size) -> bandwidth (consecutive)
    bw_dict_strided = {}   # str(cp_size) -> bandwidth (strided)
    linear_fits = {}       # topo_key -> {alpha, beta, r_squared}

    cp_size = 2
    while cp_size <= world_size:
        for topo in topologies_to_profile:
            group_ranks_list = build_group_ranks_list(world_size, cp_size, topo)
            num_groups = len(group_ranks_list)

            my_group = None
            for grp_ranks in group_ranks_list:
                group = dist.new_group(ranks=grp_ranks)
                if rank in grp_ranks:
                    my_group = group

            if rank == 0:
                example_ranks = group_ranks_list[0]
                print(f"\n--- P2P Ring cp={cp_size}, topo={topo} ({num_groups} groups, e.g. {example_ranks[:6]}...) ---")

            cp_results = {}

            if args.mode in ["model", "both"]:
                seq_lengths = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
                seq_lengths = [s for s in seq_lengths if s >= cp_size * 2]

                if rank == 0:
                    print(f"  [Model-specific profiling]")
                model_results = profile_p2p_ring(
                    my_group, cp_size,
                    args.hidden_size, args.num_kv_heads,
                    seq_lengths,
                    warmup_iters=args.warmup,
                    profile_iters=args.iters,
                )
                cp_results["model"] = model_results

            if args.mode in ["raw", "both"]:
                msg_sizes_mb = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

                if rank == 0:
                    print(f"  [Raw message size profiling]")
                raw_results = profile_p2p_raw(
                    my_group, cp_size,
                    msg_sizes_mb,
                    warmup_iters=args.warmup,
                    profile_iters=args.iters,
                )
                cp_results["raw"] = raw_results

            if "raw" in cp_results:
                large_msg_bw = [r["bandwidth_GBs"] for r in cp_results["raw"] if r["msg_size_MB"] >= 16]
                summary_bw = sum(large_msg_bw) / len(large_msg_bw) if large_msg_bw else 0
                xs = [r["msg_size_MB"] for r in cp_results["raw"]]
                ys = [r["time_ms"] for r in cp_results["raw"]]
                fit = linear_fit(xs, ys)
            elif "model" in cp_results:
                bws = [r["bandwidth_GBs"] for r in cp_results["model"]]
                summary_bw = sum(bws) / len(bws) if bws else 0
                xs = [r["kv_bytes_per_step_MB"] for r in cp_results["model"]]
                ys = [r["per_step_time_ms"] for r in cp_results["model"]]
                fit = linear_fit(xs, ys)
            else:
                summary_bw = 0
                fit = {"alpha": 0.0, "beta": 0.0, "r_squared": 0.0}

            cp_results["summary_bandwidth_GBs"] = summary_bw
            cp_results["topology"] = topo
            tk = topo_key(cp_size, topo)
            all_results[tk] = cp_results
            linear_fits[tk] = fit

            if topo == "consecutive":
                bw_dict_consec[str(cp_size)] = summary_bw
            else:
                bw_dict_strided[str(cp_size)] = summary_bw

            if rank == 0:
                print(f"  → cp={cp_size} topo={topo}: BW={summary_bw:.2f} GB/s, "
                      f"fit(alpha={fit['alpha']:.6f}, beta={fit['beta']:.4f}, R²={fit['r_squared']:.4f})")

            dist.barrier()
        cp_size *= 2

    # Save results
    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        legacy_bw = dict(bw_dict_consec)
        if not legacy_bw and bw_dict_strided:
            legacy_bw = dict(bw_dict_strided)
        legacy_bw["1"] = 1e10

        output = {
            "type": "p2p_ring_profiling",
            "world_size": world_size,
            "gpus_per_node": args.gpus_per_node,
            "hidden_size": args.hidden_size,
            "num_attention_heads": args.num_attention_heads,
            "num_kv_heads": args.num_kv_heads,
            "num_layers": args.num_layers,
            "timestamp": timestamp,
            "topologies_profiled": topologies_to_profile,
            "results": all_results,
            "linear_fits": linear_fits,
            "bandwidth_dict_GBs": legacy_bw,
            "bandwidth_dict_consec_GBs": {**bw_dict_consec, "1": 1e10},
            "bandwidth_dict_strided_GBs": {**bw_dict_strided, "1": 1e10},
        }

        save_path = os.path.join(args.save_dir, f"p2p_ring_profile_{world_size}gpus_{timestamp}.json")
        with open(save_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\n{'=' * 70}")
        print(f" Results saved to: {save_path}")
        print(f"\n P2P Ring Bandwidth Summary (for cost model):")
        for tk_str, fit in linear_fits.items():
            bw = all_results[tk_str]["summary_bandwidth_GBs"]
            bw_display = f"{bw:.2f}" if bw < 1e9 else "inf"
            print(f"   {tk_str:>20}: BW={bw_display} GB/s, alpha={fit['alpha']:.6f}, beta={fit['beta']:.4f}")
        print(f"{'=' * 70}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

