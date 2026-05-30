#!/usr/bin/env python3
import argparse
import importlib.util
import json
import math
import os
import statistics
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(MODEL_DIR, "../../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)
SITE_PACKAGE_DIR = os.path.join(REPO_ROOT, "galvatron", "site_package")
if os.path.isdir(SITE_PACKAGE_DIR) and SITE_PACKAGE_DIR not in sys.path:
    sys.path.insert(0, SITE_PACKAGE_DIR)

from profile_topo_utils import build_group_ranks_list  # noqa: E402


def _load_attention_impl():
    module_path = os.path.join(REPO_ROOT, "galvatron", "core", "runtime", "tensor_parallel", "attention_impl.py")
    spec = importlib.util.spec_from_file_location("align_attention_impl", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load attention_impl from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ATTN_IMPL = _load_attention_impl()
DistributedAttention = _ATTN_IMPL.DistributedAttention
FlashSelfAttentionVarlen = _ATTN_IMPL.FlashSelfAttentionVarlen
ZigzagRingFlashAttentionVarlen = _ATTN_IMPL.ZigzagRingFlashAttentionVarlen


def summarize(values: List[float]) -> Dict[str, float]:
    values = sorted(float(v) for v in values)
    if not values:
        return {"min": 0.0, "p50": 0.0, "mean": 0.0, "p90": 0.0, "max": 0.0, "std": 0.0}
    return {
        "min": values[0],
        "p50": float(values[len(values) // 2]),
        "mean": float(statistics.mean(values)),
        "p90": float(values[min(len(values) - 1, math.ceil(0.9 * len(values)) - 1)]),
        "max": values[-1],
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
    }


def balanced_counts(total_items: int, num_bins: int) -> List[int]:
    base = total_items // num_bins
    rem = total_items % num_bins
    return [base + (1 if i < rem else 0) for i in range(num_bins)]


def create_group_for_ranks(ranks: List[int]):
    group = dist.new_group(ranks=ranks)
    return {
        "group": group,
        "ranks": list(ranks),
        "size": len(ranks),
    }


def build_strategy_groups(
    world_size: int,
    rank: int,
    attn_type: str,
    parallel_size: int,
    sp_size: int,
    cp_size: int,
    group_topology: str,
    placement: str,
):
    outer_groups = build_group_ranks_list(world_size, parallel_size, group_topology)
    my_outer = None
    my_outer_idx = -1
    my_outer_group = None

    for idx, ranks in enumerate(outer_groups):
        group_obj = create_group_for_ranks(ranks)
        if rank in ranks:
            my_outer = ranks
            my_outer_idx = idx
            my_outer_group = group_obj

    my_sp_group = None
    my_cp_group = None
    my_cp_ranks = None

    if attn_type == "ulysses":
        my_sp_group = my_outer_group
    elif attn_type == "ring":
        my_cp_group = my_outer_group
        my_cp_ranks = my_outer
    elif attn_type == "usp":
        base_ranks = list(my_outer)
        if placement == "head_first":
            for cp_idx in range(cp_size):
                sp_ranks = [base_ranks[cp_idx * sp_size + j] for j in range(sp_size)]
                sp_group = create_group_for_ranks(sp_ranks)
                if rank in sp_ranks:
                    my_sp_group = sp_group
            for sp_idx in range(sp_size):
                cp_ranks = [base_ranks[cp_idx * sp_size + sp_idx] for cp_idx in range(cp_size)]
                cp_group = create_group_for_ranks(cp_ranks)
                if rank in cp_ranks:
                    my_cp_group = cp_group
                    my_cp_ranks = cp_ranks
        else:
            for sp_idx in range(sp_size):
                cp_ranks = [base_ranks[sp_idx * cp_size + j] for j in range(cp_size)]
                cp_group = create_group_for_ranks(cp_ranks)
                if rank in cp_ranks:
                    my_cp_group = cp_group
                    my_cp_ranks = cp_ranks
            for cp_idx in range(cp_size):
                sp_ranks = [base_ranks[sp_idx * cp_size + cp_idx] for sp_idx in range(sp_size)]
                sp_group = create_group_for_ranks(sp_ranks)
                if rank in sp_ranks:
                    my_sp_group = sp_group
    elif attn_type != "local":
        raise ValueError(f"Unknown attn_type: {attn_type}")

    return outer_groups, my_outer_idx, my_outer_group, my_sp_group, my_cp_group, my_cp_ranks


def make_cu_seqlens(num_seqs: int, seq_len: int, device: torch.device) -> torch.Tensor:
    if num_seqs <= 0:
        return torch.tensor([0], dtype=torch.int32, device=device)
    return torch.arange(0, (num_seqs + 1) * seq_len, step=seq_len, dtype=torch.int32, device=device)


def build_inputs(
    attn_type: str,
    seq_len: int,
    num_seqs: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    parallel_size: int,
    sp_size: int,
    cp_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if num_seqs <= 0:
        raise ValueError("num_seqs must be positive for benchmarking")

    if attn_type in ("ring", "usp"):
        if seq_len % (2 * cp_size) != 0:
            raise ValueError(f"seq_len={seq_len} must be divisible by 2*cp_size={2 * cp_size}")
        cp_local_seq = seq_len // cp_size
        cu = make_cu_seqlens(num_seqs, cp_local_seq, device)
        max_seqlen = cp_local_seq
        if attn_type == "ring":
            local_tokens = num_seqs * cp_local_seq
        else:
            total_cp_local_tokens = num_seqs * cp_local_seq
            if total_cp_local_tokens % sp_size != 0:
                raise ValueError(
                    f"num_seqs*seq_len/cp={total_cp_local_tokens} must be divisible by sp_size={sp_size}"
                )
            local_tokens = total_cp_local_tokens // sp_size
    else:
        total_tokens = num_seqs * seq_len
        cu = make_cu_seqlens(num_seqs, seq_len, device)
        max_seqlen = seq_len
        if attn_type == "ulysses":
            if total_tokens % parallel_size != 0:
                raise ValueError(f"total_tokens={total_tokens} must be divisible by parallel_size={parallel_size}")
            local_tokens = total_tokens // parallel_size
        else:
            local_tokens = total_tokens

    q = torch.randn(1, local_tokens, n_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(1, local_tokens, n_kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(1, local_tokens, n_kv_heads, head_dim, device=device, dtype=dtype)
    return q, k, v, cu, max_seqlen


def instantiate_module(
    attn_type: str,
    sp_group,
    cp_group,
    cp_ranks,
):
    if attn_type == "local":
        return FlashSelfAttentionVarlen(causal=True, attention_dropout=0.0)
    if attn_type == "ulysses":
        local_attention = FlashSelfAttentionVarlen(causal=True, attention_dropout=0.0)
        return DistributedAttention(local_attention=local_attention, sequence_process_group=sp_group["group"])
    if attn_type == "ring":
        return ZigzagRingFlashAttentionVarlen(
            attention_dropout=0.0,
            cp_group=cp_group["group"],
            cp_ranks=cp_ranks,
            causal=True,
        )
    if attn_type == "usp":
        local_attention = ZigzagRingFlashAttentionVarlen(
            attention_dropout=0.0,
            cp_group=cp_group["group"],
            cp_ranks=cp_ranks,
            causal=True,
        )
        return DistributedAttention(local_attention=local_attention, sequence_process_group=sp_group["group"])
    raise ValueError(f"Unknown attn_type: {attn_type}")


def run_once(module, attn_type, q, k, v, cu_seqlens, max_seqlen):
    q_ = q.detach().clone().requires_grad_(True)
    k_ = k.detach().clone().requires_grad_(True)
    v_ = v.detach().clone().requires_grad_(True)

    if attn_type == "local":
        out = module(q_, k_, v_, cu_seqlens, max_seqlen)
    elif attn_type == "ulysses":
        out = module(q_, k_, v_, 0, cu_seqlens, max_seqlen)
    elif attn_type == "ring":
        out = module(q_, k_, v_, cu_seqlens, max_seqlen)
    elif attn_type == "usp":
        out = module(q_, k_, v_, 0, cu_seqlens, max_seqlen)
    else:
        raise ValueError(f"Unknown attn_type: {attn_type}")

    loss = out.float().square().mean()
    loss.backward()
    return loss


def main():
    parser = argparse.ArgumentParser(description="Pure attention benchmark harness for Ulysses / Ring / USP")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--attn-type", choices=["local", "ulysses", "ring", "usp"], required=True)
    parser.add_argument("--parallel-size", type=int, required=True)
    parser.add_argument("--sp-size", type=int, default=1)
    parser.add_argument("--cp-size", type=int, default=1)
    parser.add_argument("--group-topology", choices=["consecutive", "strided"], default="consecutive")
    parser.add_argument("--placement", choices=["context_first", "head_first"], default="context_first")
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--num-seqs", type=int, required=True)
    parser.add_argument("--n-heads", type=int, required=True)
    parser.add_argument("--n-kv-heads", type=int, required=True)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    outer_groups, my_group_idx, my_outer_group, my_sp_group, my_cp_group, my_cp_ranks = build_strategy_groups(
        world_size=world_size,
        rank=rank,
        attn_type=args.attn_type,
        parallel_size=args.parallel_size,
        sp_size=args.sp_size,
        cp_size=args.cp_size,
        group_topology=args.group_topology,
        placement=args.placement,
    )

    num_groups = len(outer_groups)
    seq_counts = balanced_counts(args.num_seqs, num_groups)
    my_num_seqs = seq_counts[my_group_idx]

    result = {
        "case_name": args.case_name,
        "attn_type": args.attn_type,
        "parallel_size": args.parallel_size,
        "sp_size": args.sp_size,
        "cp_size": args.cp_size,
        "group_topology": args.group_topology,
        "placement": args.placement,
        "world_size": world_size,
        "seq_len": args.seq_len,
        "num_seqs": args.num_seqs,
        "n_heads": args.n_heads,
        "n_kv_heads": args.n_kv_heads,
        "head_dim": args.head_dim,
        "dtype": args.dtype,
        "num_groups": num_groups,
        "group_seq_counts": seq_counts,
        "status": "PASS",
    }

    try:
        q, k, v, cu, max_seqlen = build_inputs(
            attn_type=args.attn_type,
            seq_len=args.seq_len,
            num_seqs=my_num_seqs,
            n_heads=args.n_heads,
            n_kv_heads=args.n_kv_heads,
            head_dim=args.head_dim,
            parallel_size=args.parallel_size,
            sp_size=args.sp_size,
            cp_size=args.cp_size,
            device=device,
            dtype=dtype,
        )
        module = instantiate_module(args.attn_type, my_sp_group, my_cp_group, my_cp_ranks)
        module.train()

        for _ in range(args.warmup):
            run_once(module, args.attn_type, q, k, v, cu, max_seqlen)
        torch.cuda.synchronize()
        dist.barrier(group=my_outer_group["group"])

        per_iter_ms = []
        for _ in range(args.iters):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            run_once(module, args.attn_type, q, k, v, cu, max_seqlen)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
            elapsed_tensor = torch.tensor([elapsed_ms], device=device, dtype=torch.float64)
            dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX, group=my_outer_group["group"])
            per_iter_ms.append(float(elapsed_tensor.item()))

        payload = {
            "group_index": my_group_idx,
            "group_ranks": my_outer_group["ranks"],
            "group_num_seqs": my_num_seqs,
            "group_local_tokens": int(q.shape[1]),
            "group_cu_seqlens": [int(x) for x in cu.tolist()],
            "group_max_seqlen": int(max_seqlen),
            "per_iter_ms": per_iter_ms,
            "measured_per_layer_ms": float(statistics.median(per_iter_ms) if per_iter_ms else 0.0),
        }
    except Exception as exc:  # noqa: BLE001
        payload = {
            "group_index": my_group_idx,
            "group_ranks": my_outer_group["ranks"],
            "group_num_seqs": my_num_seqs,
            "error": repr(exc),
        }
        result["status"] = "FAIL"

    gathered = [None for _ in range(world_size)]
    if rank == my_outer_group["ranks"][0]:
        dist.all_gather_object(gathered, payload)
    else:
        dist.all_gather_object(gathered, None)

    if rank == 0:
        group_payloads = [p for p in gathered if p is not None]
        failures = [p for p in group_payloads if "error" in p]
        result["groups"] = group_payloads
        if failures:
            result["status"] = "FAIL"
            result["errors"] = failures
        else:
            group_times = [float(p["measured_per_layer_ms"]) for p in group_payloads]
            result["group_time_stats_ms"] = summarize(group_times)
            result["cluster_measured_per_layer_ms"] = max(group_times) if group_times else 0.0

        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[Benchmark] wrote {args.output_json}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
