#!/usr/bin/env python3
"""Run solver-only AdaCPSP ILP simulations for 16-GPU Qwen2.5-7B experiments.

This script intentionally uses the full per-sequence ILP (`method="ilp"`) and
does not set a SCIP time limit. It writes resumable JSONL/CSV outputs so long
overnight runs can be inspected while they are still running.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence as TypingSequence


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_SOLVER_PATH = THIS_DIR / "adacpsp_solver.py"
_SOLVER_SPEC = importlib.util.spec_from_file_location("adacpsp_solver_direct", _SOLVER_PATH)
if _SOLVER_SPEC is None or _SOLVER_SPEC.loader is None:
    raise ImportError(f"Could not load solver module from {_SOLVER_PATH}")
_SOLVER = importlib.util.module_from_spec(_SOLVER_SPEC)
sys.modules[_SOLVER_SPEC.name] = _SOLVER
_SOLVER_SPEC.loader.exec_module(_SOLVER)

AdaCPSPCostModel = _SOLVER.AdaCPSPCostModel
AdaCPSPOptimizer = _SOLVER.AdaCPSPOptimizer
Sequence = _SOLVER.Sequence
get_lens = _SOLVER.get_lens


MAX_SEQ_VALUES = [128 * 1024, 256 * 1024, 384 * 1024, 512 * 1024]
LONG_THRESHOLDS = [128 * 1024, 256 * 1024, 384 * 1024, 512 * 1024]
CASES = {
    "ulysses_only": ["ulysses"],
    "ring_only": ["ring"],
    "ulysses_ring": ["ulysses", "ring"],
    "usp_only": ["usp"],
    "ulysses_ring_usp": ["ulysses", "ring", "usp"],
}


def percentile(values: TypingSequence[int], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return float(xs[lo] * (1.0 - frac) + xs[hi] * frac)


def pad_len(length: int, world_size: int) -> int:
    unit = 2 * world_size
    return ((int(length) - 1) // unit + 1) * unit


def load_lengths(path: Path, max_supported_seq: int, world_size: int) -> List[int]:
    lengths: List[int] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            padded = pad_len(int(line), world_size)
            if padded <= max_supported_seq:
                lengths.append(padded)
    if not lengths:
        raise ValueError(f"No usable lengths loaded from {path}")
    return lengths


def stats_for_lengths(lengths: TypingSequence[int]) -> Dict[str, float]:
    total = sum(lengths)
    stats = {
        "num_seqs": len(lengths),
        "tokens": total,
        "min": min(lengths),
        "p50": percentile(lengths, 0.50),
        "p95": percentile(lengths, 0.95),
        "p99": percentile(lengths, 0.99),
        "max": max(lengths),
        "mean": total / len(lengths),
    }
    for threshold in LONG_THRESHOLDS:
        stats[f"ge_{threshold // 1024}k"] = sum(1 for x in lengths if x >= threshold)
    return stats


def sample_real_random(
    rng: random.Random,
    eligible: TypingSequence[int],
    batch_size: int,
) -> List[int]:
    return [rng.choice(eligible) for _ in range(batch_size)]


def sample_tail_conditioned(
    rng: random.Random,
    eligible: TypingSequence[int],
    max_seq: int,
    batch_size: int,
    long_count: int,
) -> List[int]:
    low = max(1, int(max_seq * 0.75))
    tail_pool = [x for x in eligible if low <= x <= max_seq]
    if len(tail_pool) < long_count:
        # Fall back to the top tail when the exact [0.75*max_seq, max_seq]
        # window is too sparse for a given dataset/max_seq.
        tail_pool = sorted(eligible)[-max(long_count, min(len(eligible), 256)) :]
    if len(tail_pool) < long_count:
        raise ValueError(f"Not enough tail samples for max_seq={max_seq}")
    batch = [rng.choice(tail_pool) for _ in range(long_count)]
    batch.extend(rng.choice(eligible) for _ in range(batch_size - long_count))
    rng.shuffle(batch)
    return batch


def summarize_groups(opt: AdaCPSPOptimizer, all_groups, all_results) -> Dict:
    microbatches = []
    strategy_summary: Dict[str, Dict] = defaultdict(
        lambda: {
            "groups": 0,
            "seqs": 0,
            "tokens": 0,
            "lengths": [],
            "time_ms": 0.0,
        }
    )

    for mb_idx, (groups, result) in enumerate(zip(all_groups, all_results)):
        group_rows = []
        for group_idx, (strat, group_seqs) in enumerate(groups):
            lens = get_lens(group_seqs)
            sp_for_mem = strat.sp_size if strat.attn_type in ("ulysses", "usp") else 1
            time_ms = opt.costmodel.total_time(lens, strat) if lens else 0.0
            activation_mb = (
                opt.costmodel.activation_size(lens, strat.parallel_size, sp_size=sp_for_mem)
                if lens else 0.0
            )
            model_states_mb = opt.costmodel.model_states_mb
            mem_mb = model_states_mb + activation_mb
            base_activation_mb = (
                opt.costmodel.act_per_token * sum(lens) / strat.parallel_size
                if lens else 0.0
            )
            head_padding_extra_mb = max(0.0, activation_mb - base_activation_mb)
            name = str(strat)
            strategy_summary[name]["groups"] += 1
            strategy_summary[name]["seqs"] += len(lens)
            strategy_summary[name]["tokens"] += sum(lens)
            strategy_summary[name]["lengths"].extend(lens)
            strategy_summary[name]["time_ms"] += time_ms
            group_rows.append(
                {
                    "group_idx": group_idx,
                    "strategy": name,
                    "attn_type": strat.attn_type,
                    "parallel_size": strat.parallel_size,
                    "sp_size": strat.sp_size,
                    "cp_size": strat.cp_size,
                    "placement": strat.placement,
                    "num_seqs": len(lens),
                    "tokens": sum(lens),
                    "local_tokens": sum(lens) / strat.parallel_size if strat.parallel_size else 0,
                    "time_ms": time_ms,
                    "time_breakdown": build_time_breakdown(opt.costmodel, lens, strat),
                    "memory_mb": mem_mb,
                    "memory_breakdown_mb": {
                        "model_states": model_states_mb,
                        "activation": activation_mb,
                        "activation_base": base_activation_mb,
                        "activation_head_padding_extra": head_padding_extra_mb,
                        "formula": "model_states + activation; activation = act_per_token * seq_sum / num_gpus + head_padding_extra",
                    },
                    "seq_ids": [s.id for s in group_seqs],
                    "seqlens": lens,
                    "length_stats": stats_for_lengths(lens) if lens else {},
                }
            )
        microbatches.append(
            {
                "microbatch_idx": mb_idx,
                "M_ms": float(result["M"]),
                "num_groups": len(groups),
                "groups": group_rows,
            }
        )

    strategy_rows = {}
    for name, row in strategy_summary.items():
        lens = row.pop("lengths")
        row["length_stats"] = stats_for_lengths(lens) if lens else {}
        strategy_rows[name] = row
    return {
        "microbatches": microbatches,
        "strategy_summary": strategy_rows,
    }


def build_time_breakdown(cm: AdaCPSPCostModel, seqlens: List[int], strat) -> Dict:
    total_tokens = sum(seqlens)
    a2a_topology = cm._get_topo(strat.placement, "alltoall", strat.sp_size, strat.cp_size)
    ring_topology = cm._get_topo(strat.placement, "ring", strat.sp_size, strat.cp_size)
    q_factor, kv_factor = cm.head_padding_overhead(strat.sp_size if strat.attn_type in ("ulysses", "usp") else 1)
    common = {
        "total_estimated_ms": round(cm.total_time(seqlens, strat), 6),
        "model": "overlap" if cm.enable_overlap_model and strat.attn_type in ("ring", "usp") else "additive",
        "topology": {
            "alltoall": a2a_topology,
            "ring": ring_topology,
        },
        "head_padding": {
            "q_factor": round(q_factor, 6),
            "kv_factor": round(kv_factor, 6),
        },
    }

    if strat.attn_type == "ulysses":
        compute_fwd = cm.compute_time(seqlens, strat)
        compute_bwd = compute_fwd * cm.bwd_fwd_ratio
        alltoall = cm.alltoall_time(seqlens, strat.sp_size, a2a_topology)
        common.update({
            "compute_fwd_ms": round(compute_fwd, 6),
            "compute_bwd_ms": round(compute_bwd, 6),
            "alltoall_comm_ms": round(alltoall, 6),
            "p2p_ring_comm_ms": 0.0,
            "overlap_leakage": None,
            "formula": "compute_fwd * (1 + bwd_fwd_ratio) + alltoall_comm",
        })
        return common

    if strat.attn_type == "ring":
        cp = strat.cp_size
        step_compute = cm._ring_step_compute_per_layer(seqlens, strat)
        fwd_comm_step = cm._p2p_fwd_comm_per_step(total_tokens, cp, topo=ring_topology) if cp > 1 else 0.0
        bwd_comm_step = cm._p2p_bwd_comm_per_step(total_tokens, cp, topo=ring_topology) if cp > 1 else 0.0
        if cp > 1 and cm.enable_overlap_model:
            fwd_per_layer = ((cp - 1) * cm._leaky_max(step_compute, fwd_comm_step) + step_compute)
            bwd_step_compute = step_compute * cm.bwd_fwd_ratio
            bwd_per_layer = ((cp - 1) * cm._leaky_max(bwd_step_compute, bwd_comm_step) + bwd_step_compute)
            total = (fwd_per_layer + bwd_per_layer) * cm.l
        else:
            fwd_per_layer = step_compute
            bwd_per_layer = step_compute * cm.bwd_fwd_ratio
            total = (fwd_per_layer + bwd_per_layer) * cm.l
        common.update({
            "ring_step_compute_per_layer_ms": round(step_compute, 6),
            "ring_fwd_comm_step_ms": round(fwd_comm_step, 6),
            "ring_bwd_comm_step_ms": round(bwd_comm_step, 6),
            "ring_fwd_per_layer_ms": round(fwd_per_layer, 6),
            "ring_bwd_per_layer_ms": round(bwd_per_layer, 6),
            "ring_layers": cm.l,
            "alltoall_comm_ms": 0.0,
            "overlap_leakage": cm.overlap_leakage,
            "formula": "sum_layers(fwd/bwd leaky overlap across ring steps)",
            "total_recomputed_ms": round(total, 6),
        })
        return common

    if strat.attn_type == "usp":
        sp, cp = strat.sp_size, strat.cp_size
        parallel_size = sp * cp
        compute_step = cm._ring_step_compute_per_layer(seqlens, strat)
        qo_msg_mb = cm.h * q_factor * total_tokens * 2 / 1024 / 1024 / parallel_size
        kv_msg_mb = cm.kv_hidden * kv_factor * total_tokens * 2 / 1024 / 1024 / parallel_size
        qo_a2a = cm._a2a_per_op_time(qo_msg_mb, sp, a2a_topology)
        kv_a2a = cm._a2a_per_op_time(kv_msg_mb, sp, a2a_topology)
        a2a_fwd_per_layer = 2 * qo_a2a + 2 * kv_a2a
        a2a_bwd_per_layer = 2 * qo_a2a + 2 * kv_a2a
        kv_hidden_after_ulysses = cm.kv_hidden * kv_factor / sp
        fwd_comm_step = cm._p2p_fwd_comm_per_step(total_tokens, cp, kv_hidden_after_ulysses, ring_topology)
        bwd_comm_step = cm._p2p_bwd_comm_per_step(total_tokens, cp, kv_hidden_after_ulysses, ring_topology)
        ring_fwd_per_layer = ((cp - 1) * cm._leaky_max(compute_step, fwd_comm_step) + compute_step)
        bwd_step_compute = compute_step * cm.bwd_fwd_ratio
        ring_bwd_per_layer = ((cp - 1) * cm._leaky_max(bwd_step_compute, bwd_comm_step) + bwd_step_compute)
        total = (a2a_fwd_per_layer + a2a_bwd_per_layer + ring_fwd_per_layer + ring_bwd_per_layer) * cm.l
        common.update({
            "usp_compute_step_per_layer_ms": round(compute_step, 6),
            "a2a_qo_msg_mb": round(qo_msg_mb, 6),
            "a2a_kv_msg_mb": round(kv_msg_mb, 6),
            "a2a_qo_per_op_ms": round(qo_a2a, 6),
            "a2a_kv_per_op_ms": round(kv_a2a, 6),
            "a2a_fwd_per_layer_ms": round(a2a_fwd_per_layer, 6),
            "a2a_bwd_per_layer_ms": round(a2a_bwd_per_layer, 6),
            "ring_fwd_comm_step_ms": round(fwd_comm_step, 6),
            "ring_bwd_comm_step_ms": round(bwd_comm_step, 6),
            "ring_fwd_per_layer_ms": round(ring_fwd_per_layer, 6),
            "ring_bwd_per_layer_ms": round(ring_bwd_per_layer, 6),
            "layers": cm.l,
            "overlap_leakage": cm.overlap_leakage,
            "formula": "layers * (a2a_fwd + a2a_bwd + ring_fwd_overlap + ring_bwd_overlap)",
            "total_recomputed_ms": round(total, 6),
        })
        return common

    return common


def yaml_scalar(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(round(value, 6)) if isinstance(value, float) else str(value)
    text = str(value)
    if not text:
        return '""'
    if any(ch in text for ch in [":", "#", "{", "}", "[", "]", ","]) or text.strip() != text:
        return json.dumps(text, ensure_ascii=False)
    return text


def yaml_inline_list(values: TypingSequence) -> str:
    return "[" + ", ".join(yaml_scalar(v) for v in values) + "]"


def yaml_dump(obj, indent: int = 0) -> List[str]:
    pad = " " * indent
    lines: List[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, (dict, list)) and not (
                isinstance(value, list) and all(not isinstance(x, (dict, list)) for x in value)
            ):
                lines.append(f"{pad}{key}:")
                lines.extend(yaml_dump(value, indent + 2))
            elif isinstance(value, list):
                lines.append(f"{pad}{key}: {yaml_inline_list(value)}")
            else:
                lines.append(f"{pad}{key}: {yaml_scalar(value)}")
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                lines.append(f"{pad}-")
                lines.extend(yaml_dump(item, indent + 2))
            elif isinstance(item, list):
                lines.append(f"{pad}- {yaml_inline_list(item)}")
            else:
                lines.append(f"{pad}- {yaml_scalar(item)}")
    else:
        lines.append(f"{pad}{yaml_scalar(obj)}")
    return lines


def case_to_yaml_case(case: Dict) -> Dict:
    if case.get("status") != "ok":
        return {
            "allowed_attn_types": case.get("attn_types", []),
            "status": case.get("status"),
            "solver_wall_s": case.get("solver_wall_s"),
        }
    microbatches = []
    for mb in case["details"]["microbatches"]:
        groups = []
        mb_seq_ids = []
        mb_seqlens = []
        for group in mb["groups"]:
            mb_seq_ids.extend(group["seq_ids"])
            mb_seqlens.extend(group["seqlens"])
            mem = group["memory_breakdown_mb"]
            groups.append({
                "group_id": group["group_idx"],
                "num_gpus": group["parallel_size"],
                "strategy": group["attn_type"],
                "strategy_label": group["strategy"],
                "sp_size": group["sp_size"],
                "cp_size": group["cp_size"],
                "placement": group["placement"],
                "estimated_time_ms": round(group["time_ms"], 6),
                "time_breakdown": group["time_breakdown"],
                "estimated_memory_gb": round(group["memory_mb"] / 1024, 6),
                "memory_breakdown_gb": {
                    "model_states": round(mem["model_states"] / 1024, 6),
                    "activation": round(mem["activation"] / 1024, 6),
                    "activation_base": round(mem["activation_base"] / 1024, 6),
                    "activation_head_padding_extra": round(mem["activation_head_padding_extra"] / 1024, 6),
                    "formula": mem["formula"],
                },
                "seq_sum": group["tokens"],
                "local_tokens": round(group["local_tokens"], 6),
                "sequence_ids": group["seq_ids"],
                "sequence_lengths": group["seqlens"],
            })
        order = sorted(range(len(mb_seq_ids)), key=lambda i: mb_seq_ids[i])
        microbatches.append({
            "microbatch_id": mb["microbatch_idx"],
            "estimated_time_ms": round(mb["M_ms"], 6),
            "seq_sum": sum(mb_seqlens),
            "num_sequences": len(mb_seqlens),
            "sequence_ids": [mb_seq_ids[i] for i in order],
            "sequence_lengths": [mb_seqlens[i] for i in order],
            "groups": groups,
        })
    return {
        "allowed_attn_types": case.get("attn_types", []),
        "implicit_local_p1": True,
        "status": "ok",
        "total_estimated_time_ms": round(case["total_time_ms"], 6),
        "solver_wall_s": round(case["solver_wall_s"], 6),
        "selected_microbatch_count": case["microbatch_count"],
        "microbatches": microbatches,
    }


def write_batch_yaml(output_dir: Path, record: Dict) -> None:
    batch = record["batch"]
    root = output_dir / "batches" / batch["sampling_mode"] / f"max_seq_{batch['max_seq']}"
    root.mkdir(parents=True, exist_ok=True)
    cases = {name: case_to_yaml_case(case) for name, case in record["cases"].items()}
    baseline = record["cases"].get("ulysses_only", {}).get("total_time_ms")
    speedups = {}
    for name, case in record["cases"].items():
        t = case.get("total_time_ms")
        speedups[name] = round(baseline / t, 6) if baseline and t else None
    doc = {
        "experiment": {
            "model": "qwen2.5-7b",
            "cluster": {
                "num_gpus": 16,
                "gpus_per_node": 8,
                "gpu_type": "A100-80G",
            },
            "memory_limit_gb": 72,
            "method": "full_ilp",
            "time_limit": None,
            "note": "Attention compute beyond profiled range is extrapolated from the last quadratic segment.",
        },
        "batch": {
            "batch_key": batch["batch_key"],
            "sampling_mode": batch["sampling_mode"],
            "max_seq": batch["max_seq"],
            "batch_idx": batch["batch_idx"],
            "global_batch_size": batch["stats"]["num_seqs"],
            "seq_sum": batch["stats"]["tokens"],
            "stats": batch["stats"],
            "sequence_ids": list(range(len(batch["seqlens"]))),
            "sequence_lengths": batch["seqlens"],
        },
        "cases": cases,
        "comparison": {
            "baseline": "ulysses_only",
            "speedup_vs_ulysses_only": speedups,
        },
    }
    path = root / f"batch_{batch['batch_idx']:03d}.yaml"
    path.write_text("\n".join(yaml_dump(doc)) + "\n", encoding="utf-8")


def run_case(
    cm: AdaCPSPCostModel,
    lengths: TypingSequence[int],
    case_name: str,
    attn_types: List[str],
    memory_limit_gb: int,
    cluster_size: int,
    hide_solver_output: bool,
) -> Dict:
    opt = AdaCPSPOptimizer(
        cluster_size=cluster_size,
        memory_limit_gb=memory_limit_gb,
        costmodel=cm,
        hide_output=hide_solver_output,
        # No "limits/time": this intentionally leaves SCIP without a time limit.
        scip_param_dict={"display/verblevel": 0},
        allowed_attn_types=attn_types,
        max_parallel_size=cluster_size,
    )
    seqs = [Sequence(seq=int(seq), id=i) for i, seq in enumerate(lengths)]
    start = time.time()
    all_groups, all_results = opt.solve_globalbatch(
        seqs,
        method="ilp",
        bucket_num=0,
    )
    wall_s = time.time() - start
    if not all_results:
        return {
            "case": case_name,
            "attn_types": attn_types,
            "status": "infeasible",
            "solver_wall_s": wall_s,
            "total_time_ms": None,
            "microbatch_count": 0,
            "details": {},
        }
    total_time_ms = float(sum(r["M"] for r in all_results))
    details = summarize_groups(opt, all_groups, all_results)
    return {
        "case": case_name,
        "attn_types": attn_types,
        "status": "ok",
        "solver_wall_s": wall_s,
        "total_time_ms": total_time_ms,
        "microbatch_count": len(all_groups),
        "details": details,
    }


def append_jsonl(path: Path, row: Dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv_rows(path: Path, rows: Iterable[Dict], fieldnames: List[str]) -> None:
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def flatten_summary_rows(record: Dict) -> List[Dict]:
    batch = record["batch"]
    rows = []
    baseline = record["cases"].get("ulysses_only", {})
    baseline_time = baseline.get("total_time_ms")
    for case_name, case in record["cases"].items():
        t = case.get("total_time_ms")
        speedup = baseline_time / t if baseline_time and t else None
        rows.append(
            {
                "batch_key": batch["batch_key"],
                "sampling_mode": batch["sampling_mode"],
                "max_seq": batch["max_seq"],
                "batch_idx": batch["batch_idx"],
                "case": case_name,
                "status": case.get("status"),
                "total_time_ms": t,
                "speedup_vs_ulysses": speedup,
                "solver_wall_s": case.get("solver_wall_s"),
                "microbatch_count": case.get("microbatch_count"),
                "tokens": batch["stats"]["tokens"],
                "max": batch["stats"]["max"],
                "p95": batch["stats"]["p95"],
                "p99": batch["stats"]["p99"],
                "ge_128k": batch["stats"]["ge_128k"],
                "ge_256k": batch["stats"]["ge_256k"],
                "ge_384k": batch["stats"]["ge_384k"],
                "ge_512k": batch["stats"]["ge_512k"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "varlen_datasets" / "github.txt")
    parser.add_argument("--attention-profile", type=Path, default=THIS_DIR / "configs" / "profile_validate_qwen2.5-7b_20260406_225144.json")
    parser.add_argument("--comm-profile", type=Path, default=THIS_DIR / "configs" / "comm_profile_qwen2.5-7b_16gpus_20260512_003327.json")
    parser.add_argument("--output-dir", type=Path, default=THIS_DIR / "configs" / "ilp_sim_qwen25_7b_16gpu")
    parser.add_argument("--cluster-size", type=int, default=16)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--memory-limit-gb", type=int, default=72)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--random-batches", type=int, default=5)
    parser.add_argument("--tail-batches", type=int, default=5)
    parser.add_argument("--tail-long-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument("--hide-solver-output", action="store_true", default=True)
    parser.add_argument("--max-seq", type=int, nargs="+", default=MAX_SEQ_VALUES)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_jsonl = args.output_dir / "results.jsonl"
    summary_csv = args.output_dir / "summary.csv"
    manifest_path = args.output_dir / "manifest.json"

    manifest = {
        "dataset": str(args.dataset),
        "attention_profile": str(args.attention_profile),
        "comm_profile": str(args.comm_profile),
        "cluster_size": args.cluster_size,
        "gpus_per_node": args.gpus_per_node,
        "memory_limit_gb": args.memory_limit_gb,
        "global_batch_size": args.global_batch_size,
        "max_seq": args.max_seq,
        "random_batches": args.random_batches,
        "tail_batches": args.tail_batches,
        "tail_long_count": args.tail_long_count,
        "seed": args.seed,
        "cases": CASES,
        "method": "full_ilp_no_time_limit",
        "note": "Attention compute beyond profiled range is extrapolated from the last quadratic segment.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    max_supported_seq = max(args.max_seq)
    lengths_all = load_lengths(args.dataset, max_supported_seq, args.cluster_size)
    print(f"Loaded {len(lengths_all)} lengths from {args.dataset}")
    print(f"Dataset stats up to {max_supported_seq}: {stats_for_lengths(lengths_all)}")

    cm = AdaCPSPCostModel.from_attention_and_comm_profiles(
        attention_json=str(args.attention_profile),
        comm_profile_json=str(args.comm_profile),
        cluster_size=args.cluster_size,
        param_size_B=7.0,
        gpus_per_node=args.gpus_per_node,
    )

    rng = random.Random(args.seed)
    csv_fields = [
        "batch_key",
        "sampling_mode",
        "max_seq",
        "batch_idx",
        "case",
        "status",
        "total_time_ms",
        "speedup_vs_ulysses",
        "solver_wall_s",
        "microbatch_count",
        "tokens",
        "max",
        "p95",
        "p99",
        "ge_128k",
        "ge_256k",
        "ge_384k",
        "ge_512k",
    ]

    for max_seq in args.max_seq:
        eligible = [x for x in lengths_all if x <= max_seq]
        if not eligible:
            raise ValueError(f"No samples eligible for max_seq={max_seq}")
        batches = []
        for batch_idx in range(args.random_batches):
            batches.append(("real_random", batch_idx, sample_real_random(rng, eligible, args.global_batch_size)))
        for batch_idx in range(args.tail_batches):
            batches.append((
                "tail_conditioned",
                batch_idx,
                sample_tail_conditioned(rng, eligible, max_seq, args.global_batch_size, args.tail_long_count),
            ))

        for sampling_mode, batch_idx, lengths in batches:
            batch_key = f"{sampling_mode}_max{max_seq}_b{batch_idx}"
            batch = {
                "batch_key": batch_key,
                "sampling_mode": sampling_mode,
                "max_seq": max_seq,
                "batch_idx": batch_idx,
                "stats": stats_for_lengths(lengths),
                "seqlens": lengths,
            }
            print("\n" + "=" * 100)
            print(f"Batch {batch_key}: {batch['stats']}")
            cases: Dict[str, Dict] = {}
            for case_name, attn_types in CASES.items():
                print(f"[RUN] {batch_key} case={case_name} attn_types={attn_types}", flush=True)
                case = run_case(
                    cm=cm,
                    lengths=lengths,
                    case_name=case_name,
                    attn_types=attn_types,
                    memory_limit_gb=args.memory_limit_gb,
                    cluster_size=args.cluster_size,
                    hide_solver_output=args.hide_solver_output,
                )
                cases[case_name] = case
                print(
                    f"[DONE] {batch_key} case={case_name} status={case['status']} "
                    f"time_ms={case.get('total_time_ms')} wall_s={case['solver_wall_s']:.2f} "
                    f"microbatches={case.get('microbatch_count')}",
                    flush=True,
                )
            record = {
                "batch": batch,
                "cases": cases,
            }
            append_jsonl(results_jsonl, record)
            write_batch_yaml(args.output_dir, record)
            write_csv_rows(summary_csv, flatten_summary_rows(record), csv_fields)
            print(f"[WRITE] appended {batch_key} to {results_jsonl} and {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
