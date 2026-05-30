#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze AdaCPSP end-to-end timing JSONL records.

The training script writes one JSONL file per profiled rank.  This script
aggregates those records, fits the residual between measured step time and the
current AdaCPSP attention+communication prediction, and reports calibrated
end-to-end speedups.
"""

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Dict, Iterable, List


def iter_records(profile_dir: str) -> Iterable[Dict]:
    for name in sorted(os.listdir(profile_dir)):
        if not name.endswith(".jsonl"):
            continue
        path = os.path.join(profile_dir, name)
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("phase") == "train_step":
                    yield record


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    idx = (len(xs) - 1) * pct / 100.0
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (idx - lo)


def linear_fit(xs: List[float], ys: List[float]) -> Dict[str, float]:
    n = len(xs)
    if n < 2:
        beta = ys[0] if ys else 0.0
        return {"alpha_ms_per_token": 0.0, "beta_ms": beta, "r_squared": 0.0}

    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        alpha = 0.0
        beta = sy / n
    else:
        alpha = (n * sxy - sx * sy) / denom
        beta = (sy - alpha * sx) / n

    y_mean = sy / n
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum((y - (alpha * x + beta)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return {
        "alpha_ms_per_token": alpha,
        "beta_ms": beta,
        "r_squared": r2,
    }


def extract_row(record: Dict) -> Dict:
    timings = record.get("timings_ms", {})
    global_batch = record.get("global_batch", {})
    local_batch = record.get("local_batch", {})
    predicted = record.get("predicted_adacpsp") or {}
    measured_ms = float(timings.get("wall_step_total", 0.0))
    predicted_ms = predicted.get("total_ms")
    tokens_per_device = global_batch.get("tokens_per_device")
    if tokens_per_device is None:
        tokens_per_device = local_batch.get("local_tokens", 0)
    global_tokens = global_batch.get("global_tokens", local_batch.get("local_tokens", 0))

    return {
        "strategy": record.get("strategy_label", "auto"),
        "rank": record.get("rank"),
        "loader_iter": record.get("loader_iter"),
        "measured_ms": measured_ms,
        "predicted_ms": float(predicted_ms) if predicted_ms is not None else None,
        "residual_ms": measured_ms - float(predicted_ms) if predicted_ms is not None else None,
        "tokens_per_device": float(tokens_per_device or 0),
        "global_tokens": int(global_tokens or 0),
        "forward_backward_ms": float(timings.get("forward_backward", 0.0)),
        "optimizer_step_ms": float(timings.get("optimizer_step", 0.0)),
        "grad_clip_ms": float(timings.get("grad_clip", 0.0)),
        "zero_grad_ms": float(timings.get("zero_grad", 0.0)),
        "solve_and_dispatch_ms": float(timings.get("solve_and_dispatch", 0.0)),
        "tokens_per_second": float(record.get("tokens_per_second", 0.0)),
    }


def align_step_rows(rank_rows: List[Dict]) -> List[Dict]:
    """Align rank-local records into global train-step records.

    A distributed training step is gated by the slowest profiled rank, while the
    cost model prediction is a single global schedule estimate.  Residuals must
    therefore be computed after grouping by (strategy, loader_iter), not from
    each rank-local timing independently.
    """
    grouped: Dict[tuple, List[Dict]] = defaultdict(list)
    for row in rank_rows:
        grouped[(row["strategy"], row["loader_iter"])].append(row)

    step_rows = []
    for (strategy, loader_iter), items in sorted(grouped.items()):
        slowest = max(items, key=lambda r: r["measured_ms"])
        predicted_values = [r["predicted_ms"] for r in items if r["predicted_ms"] is not None]
        predicted_ms = predicted_values[0] if predicted_values else None
        if len(predicted_values) > 1:
            spread = max(predicted_values) - min(predicted_values)
            if spread > 1e-6:
                predicted_ms = mean(predicted_values)

        measured_ms = slowest["measured_ms"]
        global_tokens = max(r["global_tokens"] for r in items)
        tokens_per_device = max(r["tokens_per_device"] for r in items)
        row = {
            "strategy": strategy,
            "loader_iter": loader_iter,
            "measured_ms": measured_ms,
            "measured_rank": slowest["rank"],
            "profiled_ranks": ";".join(str(r["rank"]) for r in sorted(items, key=lambda x: x["rank"])),
            "predicted_ms": predicted_ms,
            "residual_ms": measured_ms - predicted_ms if predicted_ms is not None else None,
            "tokens_per_device": tokens_per_device,
            "global_tokens": global_tokens,
            "forward_backward_ms": max(r["forward_backward_ms"] for r in items),
            "optimizer_step_ms": max(r["optimizer_step_ms"] for r in items),
            "grad_clip_ms": max(r["grad_clip_ms"] for r in items),
            "zero_grad_ms": max(r["zero_grad_ms"] for r in items),
            "solve_and_dispatch_ms": max(r["solve_and_dispatch_ms"] for r in items),
            "tokens_per_second": (
                global_tokens / (measured_ms / 1000.0) if measured_ms > 0 else 0.0
            ),
            "num_rank_records": len(items),
        }
        step_rows.append(row)
    return step_rows


def aggregate_by_strategy(rows: List[Dict]) -> List[Dict]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        grouped[row["strategy"]].append(row)

    aggregates = []
    for strategy, items in sorted(grouped.items()):
        measured = [r["measured_ms"] for r in items]
        predicted = [r["predicted_ms"] for r in items if r["predicted_ms"] is not None]
        residual = [r["residual_ms"] for r in items if r["residual_ms"] is not None]
        calibrated = [r["calibrated_predicted_ms"] for r in items if "calibrated_predicted_ms" in r]
        calibrated_err = [r["calibrated_error_pct"] for r in items if "calibrated_error_pct" in r]
        aggregates.append({
            "strategy": strategy,
            "n_steps": len(items),
            "measured_mean_ms": mean(measured),
            "measured_p50_ms": percentile(measured, 50),
            "measured_p90_ms": percentile(measured, 90),
            "predicted_mean_ms": mean(predicted),
            "residual_mean_ms": mean(residual),
            "calibrated_predicted_mean_ms": mean(calibrated),
            "calibrated_error_mean_pct": mean(calibrated_err),
            "calibrated_error_p90_pct": percentile(calibrated_err, 90),
            "tokens_per_device_mean": mean([r["tokens_per_device"] for r in items]),
            "tokens_per_second_mean": mean([r["tokens_per_second"] for r in items]),
            "forward_backward_mean_ms": mean([r["forward_backward_ms"] for r in items]),
            "optimizer_step_mean_ms": mean([r["optimizer_step_ms"] for r in items]),
            "solve_and_dispatch_mean_ms": mean([r["solve_and_dispatch_ms"] for r in items]),
        })
    return aggregates


def write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze AdaCPSP end-to-end timing records")
    parser.add_argument("--profile-dir", required=True, help="Directory containing rank*.jsonl timing files")
    parser.add_argument("--output-dir", default=None, help="Directory for aggregate CSV/JSON outputs")
    parser.add_argument("--baseline-strategy", default=None, help="Strategy label used as speedup baseline")
    args = parser.parse_args()

    rows = [extract_row(record) for record in iter_records(args.profile_dir)]
    rows = [r for r in rows if r["measured_ms"] > 0]
    if not rows:
        raise SystemExit(f"No train_step timing records found in {args.profile_dir}")
    step_rows = align_step_rows(rows)

    output_dir = args.output_dir or os.path.join(args.profile_dir, "analysis")
    os.makedirs(output_dir, exist_ok=True)

    fit_rows = [r for r in step_rows if r["residual_ms"] is not None]
    fit = linear_fit(
        [r["tokens_per_device"] for r in fit_rows],
        [r["residual_ms"] for r in fit_rows],
    )
    for row in step_rows:
        residual_pred = fit["alpha_ms_per_token"] * row["tokens_per_device"] + fit["beta_ms"]
        row["calibrated_predicted_ms"] = (
            row["predicted_ms"] + residual_pred if row["predicted_ms"] is not None else residual_pred
        )
        row["calibrated_error_pct"] = (
            abs(row["calibrated_predicted_ms"] - row["measured_ms"]) / row["measured_ms"] * 100.0
            if row["measured_ms"] > 0 else float("nan")
        )

    aggregates = aggregate_by_strategy(step_rows)
    baseline = args.baseline_strategy
    if baseline is None:
        baseline = "ulysses:16" if any(a["strategy"] == "ulysses:16" for a in aggregates) else aggregates[0]["strategy"]
    baseline_measured = next((a["measured_mean_ms"] for a in aggregates if a["strategy"] == baseline), None)
    baseline_calibrated = next((a["calibrated_predicted_mean_ms"] for a in aggregates if a["strategy"] == baseline), None)
    if baseline_measured:
        for agg in aggregates:
            agg["measured_speedup_vs_baseline"] = baseline_measured / agg["measured_mean_ms"]
    else:
        for agg in aggregates:
            agg["measured_speedup_vs_baseline"] = float("nan")
    if baseline_calibrated:
        for agg in aggregates:
            agg["calibrated_speedup_vs_baseline"] = baseline_calibrated / agg["calibrated_predicted_mean_ms"]
    else:
        for agg in aggregates:
            agg["calibrated_speedup_vs_baseline"] = float("nan")

    write_csv(os.path.join(output_dir, "rank_records_raw.csv"), rows)
    write_csv(os.path.join(output_dir, "step_records_calibrated.csv"), step_rows)
    write_csv(os.path.join(output_dir, "aggregate_by_strategy.csv"), aggregates)

    summary = {
        "profile_dir": args.profile_dir,
        "num_rank_records": len(rows),
        "num_aligned_steps": len(step_rows),
        "baseline_strategy": baseline,
        "alignment": {
            "unit": "global_train_step",
            "group_key": ["strategy", "loader_iter"],
            "measured_ms": "max wall_step_total across profiled ranks",
            "predicted_ms": "AdaCPSP global attention+communication schedule estimate for the same step",
            "residual_ms": "measured_ms - predicted_ms after step alignment",
        },
        "residual_fit": fit,
        "mean_calibrated_error_pct": mean([r["calibrated_error_pct"] for r in step_rows]),
        "p90_calibrated_error_pct": percentile([r["calibrated_error_pct"] for r in step_rows], 90),
        "strategies": aggregates,
    }
    with open(os.path.join(output_dir, "end2end_fit_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
