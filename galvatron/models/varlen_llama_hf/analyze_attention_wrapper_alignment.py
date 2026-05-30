#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare attention-wrapper benchmark JSONs with AdaCPSP CostModel.

The input JSONs are produced by align_costmodel/03_benchmark_attention.py.
Those benchmarks time the real attention wrappers (local/Ulysses/Ring/USP)
for one forward+backward call. They do not include QKV/O projections, MLP,
optimizer, or FSDP overhead.
"""

import argparse
import csv
import glob
import importlib.util
import json
import os
import sys
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


def load_adacpsp_solver():
    module_path = os.path.join(SCRIPT_DIR, "adacpsp_solver.py")
    spec = importlib.util.spec_from_file_location("alignment_adacpsp_solver", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load adacpsp_solver from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_SOLVER = load_adacpsp_solver()
AdaCPSPCostModel = _SOLVER.AdaCPSPCostModel
ParallelStrategy = _SOLVER.ParallelStrategy


def load_json(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def latest_matching(pattern: str, predicate: Callable[[Dict[str, Any]], bool]) -> Optional[str]:
    for path in sorted(glob.glob(pattern), reverse=True):
        try:
            data = load_json(path)
        except Exception:
            continue
        if predicate(data):
            return path
    return None


def discover_profiles(configs_dir: str) -> Tuple[str, str, Optional[str]]:
    attention_json = latest_matching(
        os.path.join(configs_dir, "profile_validate_*.json"),
        lambda d: "attention" in d and "segments" in d.get("attention", {}),
    )
    comm_json = latest_matching(
        os.path.join(configs_dir, "comm_profile_*.json"),
        lambda d: "alltoall" in d and "p2p_ring" in d,
    )
    if attention_json is None:
        raise FileNotFoundError(f"No attention profile_validate_*.json found in {configs_dir}")
    if comm_json is None:
        raise FileNotFoundError(f"No unified comm_profile_*.json found in {configs_dir}")
    attention_model = load_json(attention_json).get("model_name")
    validation_json = latest_matching(
        os.path.join(configs_dir, "profile_validate_*.json"),
        lambda d: "comm_validation" in d and d.get("model_name") == attention_model,
    )
    return attention_json, comm_json, validation_json


def iter_benchmark_jsons(path: str) -> Iterable[str]:
    if os.path.isdir(path):
        yield from sorted(glob.glob(os.path.join(path, "**", "*.json"), recursive=True))
    else:
        yield path


def strategy_from_case(case: Dict[str, Any]) -> ParallelStrategy:
    attn_type = case["attn_type"]
    parallel_size = int(case.get("parallel_size", 1))
    if attn_type == "local":
        return ParallelStrategy("ulysses", 1)
    if attn_type == "ulysses":
        return ParallelStrategy("ulysses", parallel_size)
    if attn_type == "ring":
        return ParallelStrategy("ring", parallel_size)
    if attn_type == "usp":
        return ParallelStrategy(
            "usp",
            parallel_size,
            sp_size=int(case["sp_size"]),
            cp_size=int(case["cp_size"]),
            placement=case.get("placement", "context_first"),
        )
    raise ValueError(f"Unknown attn_type: {attn_type}")


def group_seqlens(case: Dict[str, Any], group: Dict[str, Any]) -> List[int]:
    return [int(case["seq_len"])] * int(group.get("group_num_seqs", 0))


def predict_group(cm: AdaCPSPCostModel, case: Dict[str, Any], group: Dict[str, Any]) -> Dict[str, float]:
    seqlens = group_seqlens(case, group)
    if not seqlens:
        return {
            "predicted_total_per_layer_ms": 0.0,
            "predicted_fwd_compute_per_layer_ms": 0.0,
            "predicted_comm_per_layer_ms": 0.0,
        }
    strategy = strategy_from_case(case)
    total_ms = cm.total_time(seqlens, strategy)
    fwd_compute_ms = cm.compute_time(seqlens, strategy)
    comm_ms = cm.comm_time(seqlens, strategy)
    return {
        "predicted_total_per_layer_ms": float(total_ms / cm.l),
        # Diagnostic only: forward attention compute component. total_time()
        # includes backward ratio and may overlap Ring/USP compute and comm.
        "predicted_fwd_compute_per_layer_ms": float(fwd_compute_ms / cm.l),
        # Diagnostic only under overlap models; not necessarily additive.
        "predicted_comm_per_layer_ms": float(comm_ms / cm.l),
    }


def summarize_case(cm: AdaCPSPCostModel, path: str, case: Dict[str, Any]) -> Dict[str, Any]:
    groups = case.get("groups", [])
    group_rows = []
    for group in groups:
        pred = predict_group(cm, case, group)
        measured = float(group.get("measured_per_layer_ms", 0.0))
        predicted = pred["predicted_total_per_layer_ms"]
        error_pct = abs(predicted - measured) / measured * 100.0 if measured > 0 else 0.0
        group_rows.append({
            "group_index": group.get("group_index"),
            "group_ranks": group.get("group_ranks"),
            "group_num_seqs": group.get("group_num_seqs", 0),
            "group_local_tokens": group.get("group_local_tokens", 0),
            "measured_per_layer_ms": measured,
            "error_pct": error_pct,
            **pred,
        })

    measured_cluster = float(case.get("cluster_measured_per_layer_ms", 0.0))
    predicted_cluster = max((r["predicted_total_per_layer_ms"] for r in group_rows), default=0.0)
    cluster_error_pct = (
        abs(predicted_cluster - measured_cluster) / measured_cluster * 100.0
        if measured_cluster > 0 else 0.0
    )
    return {
        "source_json": path,
        "case_name": case.get("case_name", os.path.basename(path)),
        "status": case.get("status", "UNKNOWN"),
        "attn_type": case.get("attn_type"),
        "parallel_size": int(case.get("parallel_size", 1)),
        "sp_size": int(case.get("sp_size", 1)),
        "cp_size": int(case.get("cp_size", 1)),
        "placement": case.get("placement", "context_first"),
        "group_topology": case.get("group_topology", "consecutive"),
        "world_size": int(case.get("world_size", 1)),
        "seq_len": int(case.get("seq_len", 0)),
        "num_seqs": int(case.get("num_seqs", 0)),
        "num_groups": int(case.get("num_groups", len(groups))),
        "measured_cluster_per_layer_ms": measured_cluster,
        "predicted_cluster_per_layer_ms": predicted_cluster,
        "cluster_error_pct": cluster_error_pct,
        "predicted_group_max_fwd_compute_per_layer_ms": max(
            (r.get("predicted_fwd_compute_per_layer_ms", 0.0) for r in group_rows),
            default=0.0,
        ),
        "predicted_group_max_comm_per_layer_ms": max(
            (r.get("predicted_comm_per_layer_ms", 0.0) for r in group_rows),
            default=0.0,
        ),
        "groups": group_rows,
    }


def flatten_case(row: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in row.items() if k != "groups"}


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze attention wrapper vs AdaCPSP CostModel alignment")
    parser.add_argument("--benchmark-json", required=True, help="Benchmark JSON file or directory from 03_benchmark_attention.py")
    parser.add_argument("--configs-dir", default=os.path.join(SCRIPT_DIR, "configs"))
    parser.add_argument("--attention-json", default=None)
    parser.add_argument("--comm-json", default=None)
    parser.add_argument("--validation-json", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--param-size-b", type=float, default=7.0)
    parser.add_argument("--zero-stage", type=int, default=3)
    parser.add_argument("--act-per-token", type=float, default=3.96)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    args = parser.parse_args()

    discovered_attention, discovered_comm, discovered_validation = discover_profiles(args.configs_dir)
    attention_json = args.attention_json or discovered_attention
    comm_json = args.comm_json or discovered_comm
    validation_json = args.validation_json if args.validation_json is not None else discovered_validation

    cm = AdaCPSPCostModel.from_attention_and_comm_profiles(
        attention_json=attention_json,
        comm_profile_json=comm_json,
        cluster_size=1,
        param_size_B=args.param_size_b,
        zero_stage=args.zero_stage,
        act_per_token=args.act_per_token,
        validation_json=validation_json,
        gpus_per_node=args.gpus_per_node,
    )

    rows = []
    failures = []
    for path in iter_benchmark_jsons(args.benchmark_json):
        try:
            case = load_json(path)
        except Exception:
            continue
        if "attn_type" not in case or "seq_len" not in case:
            continue
        if case.get("status") != "PASS":
            failures.append({"source_json": path, "status": case.get("status"), "errors": case.get("errors")})
        rows.append(summarize_case(cm, path, case))

    if not rows:
        raise SystemExit(f"No benchmark JSON cases found under {args.benchmark_json}")

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(args.benchmark_json) if not os.path.isdir(args.benchmark_json) else args.benchmark_json,
        "alignment",
    )
    os.makedirs(output_dir, exist_ok=True)

    write_csv(os.path.join(output_dir, "attention_wrapper_alignment.csv"), [flatten_case(r) for r in rows])
    with open(os.path.join(output_dir, "attention_wrapper_alignment.json"), "w") as f:
        json.dump({
            "attention_json": attention_json,
            "comm_json": comm_json,
            "validation_json": validation_json,
            "num_cases": len(rows),
            "num_failures": len(failures),
            "failures": failures,
            "cases": rows,
        }, f, indent=2)

    by_type: Dict[str, List[float]] = {}
    for row in rows:
        by_type.setdefault(str(row["attn_type"]), []).append(float(row["cluster_error_pct"]))
    summary = {
        "attention_json": attention_json,
        "comm_json": comm_json,
        "validation_json": validation_json,
        "num_cases": len(rows),
        "mean_error_pct_by_attn_type": {k: sum(v) / len(v) for k, v in sorted(by_type.items())},
        "max_error_pct": max(float(r["cluster_error_pct"]) for r in rows),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
