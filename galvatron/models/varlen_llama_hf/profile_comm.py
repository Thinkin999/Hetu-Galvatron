#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified topology-aware communication profiling for AdaCPSP.

This driver profiles:
  1. Ulysses all-to-all communication
  2. Ring-attention P2P communication

Unlike the legacy simplified communication path, this script:
  - profiles both consecutive and strided topologies
  - treats the slowest rank within each group as the group latency
  - aggregates results across all groups of the same size/topology
  - records group-level distributions instead of only a single implicit average
  - emits a single JSON that can be consumed by training / validation / analysis
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist

from profile_alltoall import profile_alltoall, profile_alltoall_single_tensor
from profile_p2p_ring import profile_p2p_raw, profile_p2p_ring
from profile_topo_utils import build_group_ranks_list, linear_fit, topo_key


def summarize_values(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "mean": float(np.mean(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
    }


def select_stat(values: List[float], policy: str) -> float:
    arr = np.array(values, dtype=np.float64)
    if policy == "mean":
        return float(np.mean(arr))
    if policy == "median":
        return float(np.percentile(arr, 50))
    if policy == "p90":
        return float(np.percentile(arr, 90))
    if policy == "max":
        return float(np.max(arr))
    raise ValueError(f"Unknown aggregation policy: {policy}")


def aggregate_alltoall_section(
    payloads: List[Dict],
    section: str,
    sp_size: int,
    policy: str,
) -> Optional[List[Dict]]:
    section_payloads = [p for p in payloads if p.get(section)]
    if not section_payloads:
        return None

    first_points = section_payloads[0][section]
    aggregated = []
    for idx, first in enumerate(first_points):
        group_points = [p[section][idx] for p in section_payloads]
        time_values = [float(pt["time_ms"]) for pt in group_points]
        agg_time = select_stat(time_values, policy)

        if section == "raw":
            total_bytes = float(first["actual_bytes"])
            data_moved_bytes = total_bytes * (sp_size - 1) / sp_size
            bandwidth_gbs = data_moved_bytes / (agg_time / 1000.0) / 1e9 if agg_time > 0 else 0.0
        else:
            total_bytes_mb = float(first["total_bytes_MB"])
            data_moved_bytes = total_bytes_mb * 1024 * 1024 * (sp_size - 1) / sp_size
            bandwidth_gbs = data_moved_bytes / (agg_time / 1000.0) / 1e9 if agg_time > 0 else 0.0

        aggregated.append({
            **first,
            "time_ms": agg_time,
            "bandwidth_GBs": float(bandwidth_gbs),
            "group_stats": {
                "num_groups": len(group_points),
                "time_ms": summarize_values(time_values),
                "bandwidth_GBs": summarize_values([float(pt["bandwidth_GBs"]) for pt in group_points]),
            },
        })

    return aggregated


def aggregate_p2p_section(
    payloads: List[Dict],
    section: str,
    policy: str,
) -> Optional[List[Dict]]:
    section_payloads = [p for p in payloads if p.get(section)]
    if not section_payloads:
        return None

    first_points = section_payloads[0][section]
    aggregated = []
    for idx, first in enumerate(first_points):
        group_points = [p[section][idx] for p in section_payloads]

        if section == "raw":
            time_values = [float(pt["time_ms"]) for pt in group_points]
            agg_time = select_stat(time_values, policy)
            total_bytes = float(first["total_bytes"])
            bandwidth_gbs = total_bytes / (agg_time / 1000.0) / 1e9 if agg_time > 0 else 0.0
            aggregated.append({
                **first,
                "time_ms": agg_time,
                "bandwidth_GBs": float(bandwidth_gbs),
                "group_stats": {
                    "num_groups": len(group_points),
                    "time_ms": summarize_values(time_values),
                    "bandwidth_GBs": summarize_values([float(pt["bandwidth_GBs"]) for pt in group_points]),
                },
            })
            continue

        per_step_values = [float(pt["per_step_time_ms"]) for pt in group_points]
        total_ring_values = [float(pt["total_ring_time_ms"]) for pt in group_points]
        agg_per_step = select_stat(per_step_values, policy)
        agg_total_ring = select_stat(total_ring_values, policy)
        kv_bytes_mb = float(first["kv_bytes_per_step_MB"])
        bandwidth_gbs = (kv_bytes_mb * 1024 * 1024) / (agg_per_step / 1000.0) / 1e9 if agg_per_step > 0 else 0.0

        aggregated.append({
            **first,
            "per_step_time_ms": agg_per_step,
            "total_ring_time_ms": agg_total_ring,
            "bandwidth_GBs": float(bandwidth_gbs),
            "group_stats": {
                "num_groups": len(group_points),
                "per_step_time_ms": summarize_values(per_step_values),
                "total_ring_time_ms": summarize_values(total_ring_values),
                "bandwidth_GBs": summarize_values([float(pt["bandwidth_GBs"]) for pt in group_points]),
            },
        })

    return aggregated


def build_fit_from_points(points: Optional[List[Dict]], x_key: str, y_key: str) -> Optional[Dict[str, float]]:
    if not points:
        return None
    xs = [float(p[x_key]) for p in points]
    ys = [float(p[y_key]) for p in points]
    if len(xs) < 2:
        return None
    return linear_fit(xs, ys)


def build_interp_points(points: Optional[List[Dict]], x_key: str, y_key: str) -> Optional[List[List[float]]]:
    if not points:
        return None
    return [[float(p[x_key]), float(p[y_key])] for p in points]


def aggregate_group_payloads(
    payloads: List[Dict],
    comm_type: str,
    group_size: int,
    topology: str,
    policy: str,
) -> Dict:
    if comm_type == "alltoall":
        model_points = aggregate_alltoall_section(payloads, "model", group_size, policy)
        raw_points = aggregate_alltoall_section(payloads, "raw", group_size, policy)
        fit_source = raw_points if raw_points else model_points
        fit_x_key = "msg_size_MB" if raw_points else "total_bytes_MB"
        fit_y_key = "time_ms"
        interp_x_key = "total_bytes_MB"
        interp_y_key = "time_ms"
    else:
        model_points = aggregate_p2p_section(payloads, "model", policy)
        raw_points = aggregate_p2p_section(payloads, "raw", policy)
        fit_source = raw_points if raw_points else model_points
        fit_x_key = "msg_size_MB" if raw_points else "kv_bytes_per_step_MB"
        fit_y_key = "time_ms" if raw_points else "per_step_time_ms"
        interp_x_key = "kv_bytes_per_step_MB"
        interp_y_key = "per_step_time_ms"

    summary_bandwidth = 0.0
    if raw_points:
        large_bw = [float(p["bandwidth_GBs"]) for p in raw_points if float(p["msg_size_MB"]) >= 16.0]
        summary_bandwidth = float(np.mean(large_bw)) if large_bw else 0.0
    elif model_points:
        summary_bandwidth = float(np.mean([float(p["bandwidth_GBs"]) for p in model_points]))

    return {
        "topology": topology,
        "num_groups_profiled": len(payloads),
        "group_ranks": [p["group_ranks"] for p in payloads],
        "aggregation_policy": {
            "within_group_rank_latency": "max",
            "across_groups": policy,
        },
        "model": model_points,
        "raw": raw_points,
        "summary_bandwidth_GBs": summary_bandwidth,
        "linear_fit": build_fit_from_points(fit_source, fit_x_key, fit_y_key),
        "interp_points": build_interp_points(model_points, interp_x_key, interp_y_key),
        "ring_step_fit": (
            build_fit_from_points(model_points, "kv_bytes_per_step_MB", "per_step_time_ms")
            if comm_type == "p2p_ring"
            else None
        ),
    }


def maybe_leader_payload(my_group_ranks: Optional[List[int]], model_results, raw_results):
    if my_group_ranks is None or dist.get_rank() != my_group_ranks[0]:
        return None
    return {
        "group_ranks": my_group_ranks,
        "model": model_results,
        "raw": raw_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Unified topology-aware communication profiling for AdaCPSP")
    parser.add_argument("--hidden_size", type=int, default=4096)
    parser.add_argument("--num_attention_heads", type=int, default=32)
    parser.add_argument("--num_kv_heads", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--save_dir", type=str, default="./configs")
    parser.add_argument("--model_name", type=str, default="llama-7b")
    parser.add_argument("--mode", type=str, default="both", choices=["model", "raw", "both"])
    parser.add_argument("--topology", type=str, default="both", choices=["consecutive", "strided", "both"])
    parser.add_argument("--gpus_per_node", type=int, default=8)
    parser.add_argument(
        "--across-group-agg",
        type=str,
        default="p90",
        choices=["mean", "median", "p90", "max"],
        help="How to aggregate multiple groups of the same size/topology",
    )
    args, _ = parser.parse_known_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    topologies_to_profile = ["consecutive", "strided"] if args.topology == "both" else [args.topology]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if rank == 0:
        print("=" * 80)
        print(" AdaCPSP Unified Communication Profiler")
        print("=" * 80)
        print(f" Model: {args.model_name}")
        print(f" World size: {world_size}")
        print(f" Topologies: {topologies_to_profile}")
        print(f" Across-group aggregation: {args.across_group_agg}")
        print("=" * 80)

    output = {
        "type": "comm_profile",
        "schema_version": 1,
        "timestamp": timestamp,
        "model_name": args.model_name,
        "world_size": world_size,
        "gpus_per_node": args.gpus_per_node,
        "hidden_size": args.hidden_size,
        "num_attention_heads": args.num_attention_heads,
        "num_kv_heads": args.num_kv_heads,
        "num_layers": args.num_layers,
        "topologies_profiled": topologies_to_profile,
        "aggregation_policy": {
            "within_group_rank_latency": "max",
            "across_groups": args.across_group_agg,
        },
        "alltoall": {
            "results": {},
            "linear_fits": {},
            "bandwidth_dict_GBs": {"1": 1e10},
            "bandwidth_dict_consec_GBs": {"1": 1e10},
            "bandwidth_dict_strided_GBs": {"1": 1e10},
            "interp_tables": {},
        },
        "p2p_ring": {
            "results": {},
            "linear_fits": {},
            "ring_step_fits": {},
            "bandwidth_dict_GBs": {"1": 1e10},
            "bandwidth_dict_consec_GBs": {"1": 1e10},
            "bandwidth_dict_strided_GBs": {"1": 1e10},
            "interp_tables": {},
        },
    }

    sp_size = 2
    while sp_size <= world_size:
        for topology in topologies_to_profile:
            group_ranks_list = build_group_ranks_list(world_size, sp_size, topology)
            my_group = None
            my_group_ranks = None
            for ranks in group_ranks_list:
                group = dist.new_group(ranks=ranks)
                if rank in ranks:
                    my_group = group
                    my_group_ranks = ranks

            model_results = None
            raw_results = None
            if args.mode in ("model", "both"):
                seq_lengths = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
                seq_lengths = [s for s in seq_lengths if s >= sp_size * 2]
                model_results = profile_alltoall(
                    my_group,
                    sp_size,
                    args.hidden_size,
                    args.num_attention_heads,
                    args.num_layers,
                    seq_lengths,
                    warmup_iters=args.warmup,
                    profile_iters=args.iters,
                )
            if args.mode in ("raw", "both"):
                msg_sizes_mb = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
                raw_results = profile_alltoall_single_tensor(
                    my_group,
                    sp_size,
                    msg_sizes_mb,
                    warmup_iters=args.warmup,
                    profile_iters=args.iters,
                )

            payload = maybe_leader_payload(my_group_ranks, model_results, raw_results)
            gathered = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, payload)

            if rank == 0:
                leader_payloads = [p for p in gathered if p is not None]
                tk = topo_key(sp_size, topology)
                aggregated = aggregate_group_payloads(
                    leader_payloads,
                    "alltoall",
                    sp_size,
                    topology,
                    args.across_group_agg,
                )
                output["alltoall"]["results"][tk] = aggregated
                if aggregated["linear_fit"] is not None:
                    output["alltoall"]["linear_fits"][tk] = aggregated["linear_fit"]
                if aggregated["interp_points"] is not None:
                    output["alltoall"]["interp_tables"][tk] = aggregated["interp_points"]
                output["alltoall"][f"bandwidth_dict_{'consec' if topology == 'consecutive' else 'strided'}_GBs"][str(sp_size)] = aggregated["summary_bandwidth_GBs"]
                if topology == "consecutive":
                    output["alltoall"]["bandwidth_dict_GBs"][str(sp_size)] = aggregated["summary_bandwidth_GBs"]
                elif str(sp_size) not in output["alltoall"]["bandwidth_dict_GBs"]:
                    output["alltoall"]["bandwidth_dict_GBs"][str(sp_size)] = aggregated["summary_bandwidth_GBs"]

            dist.barrier()
        sp_size *= 2

    cp_size = 2
    while cp_size <= world_size:
        for topology in topologies_to_profile:
            group_ranks_list = build_group_ranks_list(world_size, cp_size, topology)
            my_group = None
            my_group_ranks = None
            for ranks in group_ranks_list:
                group = dist.new_group(ranks=ranks)
                if rank in ranks:
                    my_group = group
                    my_group_ranks = ranks

            model_results = None
            raw_results = None
            if args.mode in ("model", "both"):
                seq_lengths = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
                seq_lengths = [s for s in seq_lengths if s >= cp_size * 2]
                model_results = profile_p2p_ring(
                    my_group,
                    cp_size,
                    args.hidden_size,
                    args.num_kv_heads,
                    seq_lengths,
                    warmup_iters=args.warmup,
                    profile_iters=args.iters,
                )
            if args.mode in ("raw", "both"):
                msg_sizes_mb = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
                raw_results = profile_p2p_raw(
                    my_group,
                    cp_size,
                    msg_sizes_mb,
                    warmup_iters=args.warmup,
                    profile_iters=args.iters,
                )

            payload = maybe_leader_payload(my_group_ranks, model_results, raw_results)
            gathered = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, payload)

            if rank == 0:
                leader_payloads = [p for p in gathered if p is not None]
                tk = topo_key(cp_size, topology)
                aggregated = aggregate_group_payloads(
                    leader_payloads,
                    "p2p_ring",
                    cp_size,
                    topology,
                    args.across_group_agg,
                )
                output["p2p_ring"]["results"][tk] = aggregated
                if aggregated["linear_fit"] is not None:
                    output["p2p_ring"]["linear_fits"][tk] = aggregated["linear_fit"]
                if aggregated["ring_step_fit"] is not None:
                    output["p2p_ring"]["ring_step_fits"][tk] = aggregated["ring_step_fit"]
                if aggregated["interp_points"] is not None:
                    output["p2p_ring"]["interp_tables"][tk] = aggregated["interp_points"]
                output["p2p_ring"][f"bandwidth_dict_{'consec' if topology == 'consecutive' else 'strided'}_GBs"][str(cp_size)] = aggregated["summary_bandwidth_GBs"]
                if topology == "consecutive":
                    output["p2p_ring"]["bandwidth_dict_GBs"][str(cp_size)] = aggregated["summary_bandwidth_GBs"]
                elif str(cp_size) not in output["p2p_ring"]["bandwidth_dict_GBs"]:
                    output["p2p_ring"]["bandwidth_dict_GBs"][str(cp_size)] = aggregated["summary_bandwidth_GBs"]

            dist.barrier()
        cp_size *= 2

    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(
            args.save_dir,
            f"comm_profile_{args.model_name}_{world_size}gpus_{timestamp}.json",
        )
        with open(save_path, "w") as f:
            json.dump(output, f, indent=2)

        print("\n" + "=" * 80)
        print(f"Communication profile saved to: {save_path}")
        print("AlltoAll summary:")
        for key, result in output["alltoall"]["results"].items():
            print(f"  {key:>20}: BW={result['summary_bandwidth_GBs']:.2f} GB/s, groups={result['num_groups_profiled']}")
        print("P2P Ring summary:")
        for key, result in output["p2p_ring"]["results"].items():
            print(f"  {key:>20}: BW={result['summary_bandwidth_GBs']:.2f} GB/s, groups={result['num_groups_profiled']}")
        print("=" * 80)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
