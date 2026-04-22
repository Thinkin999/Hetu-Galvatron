#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
import statistics
import sys
from typing import Dict, Tuple


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(MODEL_DIR, "../../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from adacpsp_solver import AdaCPSPCostModel, ParallelStrategy  # noqa: E402


def load_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def latest_matching(path_glob: str, predicate=None) -> str:
    for path in sorted(glob.glob(path_glob), reverse=True):
        try:
            data = load_json(path)
        except Exception:
            continue
        if predicate is None or predicate(data):
            return path
    return ""


def build_costmodel(args) -> Tuple[AdaCPSPCostModel, str, str, str]:
    attention_json = args.attention_json or latest_matching(
        os.path.join(args.configs_dir, "profile_validate_*.json"),
        predicate=lambda d: "attention" in d and "segments" in d.get("attention", {}),
    )
    comm_json = args.comm_json or latest_matching(
        os.path.join(args.configs_dir, "comm_profile_*.json"),
        predicate=lambda d: "alltoall" in d and "p2p_ring" in d,
    )
    validation_json = args.validation_json or latest_matching(
        os.path.join(args.configs_dir, "profile_validate_*.json"),
        predicate=lambda d: "comm_validation" in d,
    )

    if not attention_json:
        raise FileNotFoundError("Could not find attention profile JSON")
    if not comm_json:
        raise FileNotFoundError("Could not find unified communication profile JSON")

    cm = AdaCPSPCostModel.from_attention_and_comm_profiles(
        attention_json=attention_json,
        comm_profile_json=comm_json,
        cluster_size=args.world_size,
        gpus_per_node=args.gpus_per_node,
        validation_json=validation_json if validation_json else None,
    )
    return cm, attention_json, comm_json, validation_json


def make_strategy(data: Dict) -> ParallelStrategy:
    attn_type = data["attn_type"]
    if attn_type == "local":
        return ParallelStrategy("ulysses", 1)
    placement = data["placement"] if attn_type == "usp" else "context_first"
    return ParallelStrategy(
        attn_type=attn_type,
        parallel_size=int(data["parallel_size"]),
        sp_size=int(data["sp_size"]),
        cp_size=int(data["cp_size"]),
        placement=placement,
    )


def main():
    parser = argparse.ArgumentParser(description="Align real forced-strategy runs with AdaCPSP cost model")
    parser.add_argument("--result-dir", required=True, help="align_costmodel/results/<run_id>")
    parser.add_argument("--configs-dir", default=os.path.join(MODEL_DIR, "configs"))
    parser.add_argument("--attention-json", default=None)
    parser.add_argument("--comm-json", default=None)
    parser.add_argument("--validation-json", default=None)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--gpus-per-node", type=int, required=True)
    args = parser.parse_args()

    summary_dir = os.path.join(args.result_dir, "summary")
    bench_paths = sorted(glob.glob(os.path.join(summary_dir, "bench_*.json")))
    if not bench_paths:
        raise FileNotFoundError(f"Missing benchmark JSONs under {summary_dir}")

    cm, attention_json, comm_json, validation_json = build_costmodel(args)

    output_csv = os.path.join(summary_dir, "costmodel_alignment.csv")
    output_md = os.path.join(summary_dir, "costmodel_alignment.md")

    rows = []
    for path in bench_paths:
        data = load_json(path)
        if data.get("status") != "PASS":
            continue
        strategy = make_strategy(data)
        seq_len = int(data["seq_len"])

        group_total_ms = []
        group_comm_ms = []
        for group in data["groups"]:
            seqlens = [seq_len] * int(group["group_num_seqs"])
            group_total_ms.append(cm.total_time(seqlens, strategy))
            group_comm_ms.append(cm.comm_time(seqlens, strategy))

        predicted_total_ms = max(group_total_ms) if group_total_ms else 0.0
        predicted_comm_ms = max(group_comm_ms) if group_comm_ms else 0.0
        predicted_total_per_layer_ms = predicted_total_ms / cm.l if cm.l > 0 else 0.0
        predicted_comm_per_layer_ms = predicted_comm_ms / cm.l if cm.l > 0 else 0.0

        measured_per_layer_ms = float(data.get("cluster_measured_per_layer_ms", 0.0))
        measured_total_ms = measured_per_layer_ms * cm.l
        error_pct = (
            abs(predicted_total_per_layer_ms - measured_per_layer_ms) / measured_per_layer_ms * 100.0
            if measured_per_layer_ms > 0
            else 0.0
        )

        if data["attn_type"] == "usp":
            strategy_label = f"usp:{data['sp_size']}x{data['cp_size']}"
        elif data["attn_type"] == "local":
            strategy_label = "local:1"
        else:
            strategy_label = f"{data['attn_type']}:{data['parallel_size']}"

        rows.append({
            "case_name": data["case_name"],
            "seq_len": seq_len,
            "gbs": int(data["num_seqs"]),
            "attn_type": data["attn_type"],
            "parallel_size": int(data["parallel_size"]),
            "sp_size": int(data["sp_size"]),
            "cp_size": int(data["cp_size"]),
            "group_topology": data["group_topology"],
            "placement": data["placement"],
            "forced_strategy": strategy_label,
            "predicted_total_ms": predicted_total_ms,
            "predicted_total_per_layer_ms": predicted_total_per_layer_ms,
            "predicted_comm_ms": predicted_comm_ms,
            "predicted_comm_per_layer_ms": predicted_comm_per_layer_ms,
            "measured_total_ms": measured_total_ms,
            "measured_per_layer_ms": measured_per_layer_ms,
            "error_pct": error_pct,
            "log_file": path,
        })

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_name",
                "seq_len",
                "gbs",
                "attn_type",
                "parallel_size",
                "sp_size",
                "cp_size",
                "group_topology",
                "placement",
                "forced_strategy",
                "predicted_total_ms",
                "predicted_total_per_layer_ms",
                "predicted_comm_ms",
                "predicted_comm_per_layer_ms",
                "measured_total_ms",
                "measured_per_layer_ms",
                "error_pct",
                "log_file",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    errors = [row["error_pct"] for row in rows]
    mean_error = statistics.mean(errors) if errors else 0.0
    max_error = max(errors) if errors else 0.0

    with open(output_md, "w") as f:
        f.write("# Cost Model Alignment\n\n")
        f.write(f"- Attention profile: `{attention_json}`\n")
        f.write(f"- Communication profile: `{comm_json}`\n")
        f.write(f"- Validation profile: `{validation_json}`\n")
        f.write(f"- Benchmark results dir: `{summary_dir}`\n")
        f.write(f"- Mean per-layer error: `{mean_error:.2f}%`\n")
        f.write(f"- Max per-layer error: `{max_error:.2f}%`\n\n")
        f.write("| case | seq_len | gbs | strategy | topology | placement | pred_layer_ms | pred_comm_layer_ms | meas_layer_ms | error_pct |\n")
        f.write("| --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            f.write(
                f"| {row['case_name']} | {row['seq_len']} | {row['gbs']} | "
                f"`{row['forced_strategy']}` | {row['group_topology']} | {row['placement']} | "
                f"{row['predicted_total_per_layer_ms']:.3f} | {row['predicted_comm_per_layer_ms']:.3f} | "
                f"{row['measured_per_layer_ms']:.3f} | {row['error_pct']:.2f}% |\n"
            )

    print(f"Attention profile: {attention_json}")
    print(f"Communication profile: {comm_json}")
    print(f"Validation profile: {validation_json}")
    print(f"Wrote alignment CSV: {output_csv}")
    print(f"Wrote alignment summary: {output_md}")
    print(f"Mean per-layer error: {mean_error:.2f}%")
    print(f"Max per-layer error: {max_error:.2f}%")


if __name__ == "__main__":
    main()
