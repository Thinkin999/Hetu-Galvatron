#!/usr/bin/env python3
"""Summarize AdaCPSP ILP simulation JSONL results."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def percentile(values: Sequence[float], q: float) -> float:
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


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def mean(values: Iterable[float]) -> float:
    xs = [float(x) for x in values if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def summarize(records: List[Dict]) -> Dict:
    by_setting: Dict[tuple, List[Dict]] = defaultdict(list)
    strategy_lengths: Dict[tuple, List[int]] = defaultdict(list)
    strategy_tokens: Dict[tuple, int] = defaultdict(int)
    strategy_groups: Dict[tuple, int] = defaultdict(int)

    for record in records:
        batch = record["batch"]
        setting = (batch["sampling_mode"], batch["max_seq"])
        by_setting[setting].append(record)
        for case_name, case in record["cases"].items():
            for strat, row in case.get("details", {}).get("strategy_summary", {}).items():
                key = (batch["sampling_mode"], batch["max_seq"], case_name, strat)
                ls = row.get("length_stats", {})
                # Reconstruct a useful distribution approximately from group details.
                for mb in case.get("details", {}).get("microbatches", []):
                    for group in mb.get("groups", []):
                        if group.get("strategy") == strat:
                            strategy_lengths[key].extend(group.get("seqlens", []))
                strategy_tokens[key] += int(row.get("tokens", 0))
                strategy_groups[key] += int(row.get("groups", 0))

    setting_rows = []
    for (sampling_mode, max_seq), recs in sorted(by_setting.items()):
        case_names = sorted({name for r in recs for name in r["cases"]})
        baseline_times = [r["cases"].get("ulysses_only", {}).get("total_time_ms") for r in recs]
        for case_name in case_names:
            times = [r["cases"].get(case_name, {}).get("total_time_ms") for r in recs]
            walls = [r["cases"].get(case_name, {}).get("solver_wall_s") for r in recs]
            mbs = [r["cases"].get(case_name, {}).get("microbatch_count") for r in recs]
            speedups = [
                b / t for b, t in zip(baseline_times, times)
                if b is not None and t is not None and t > 0
            ]
            setting_rows.append({
                "sampling_mode": sampling_mode,
                "max_seq": max_seq,
                "case": case_name,
                "num_batches": len(recs),
                "mean_time_ms": mean(times),
                "p50_time_ms": percentile([t for t in times if t is not None], 0.50),
                "mean_speedup_vs_ulysses": mean(speedups),
                "p50_speedup_vs_ulysses": percentile(speedups, 0.50),
                "mean_solver_wall_s": mean(walls),
                "mean_microbatch_count": mean(mbs),
            })

    strategy_rows = []
    for key, lengths in sorted(strategy_lengths.items()):
        sampling_mode, max_seq, case_name, strat = key
        strategy_rows.append({
            "sampling_mode": sampling_mode,
            "max_seq": max_seq,
            "case": case_name,
            "strategy": strat,
            "groups": strategy_groups[key],
            "tokens": strategy_tokens[key],
            "seqs": len(lengths),
            "min_len": min(lengths) if lengths else 0,
            "p50_len": percentile(lengths, 0.50),
            "p95_len": percentile(lengths, 0.95),
            "p99_len": percentile(lengths, 0.99),
            "max_len": max(lengths) if lengths else 0,
            "ge_128k": sum(1 for x in lengths if x >= 128 * 1024),
            "ge_256k": sum(1 for x in lengths if x >= 256 * 1024),
            "ge_384k": sum(1 for x in lengths if x >= 384 * 1024),
            "ge_512k": sum(1 for x in lengths if x >= 512 * 1024),
        })

    return {
        "setting_rows": setting_rows,
        "strategy_rows": strategy_rows,
    }


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()

    records = load_jsonl(args.result_dir / "results.jsonl")
    summary = summarize(records)
    write_csv(args.result_dir / "aggregate_by_setting.csv", summary["setting_rows"])
    write_csv(args.result_dir / "aggregate_by_strategy.csv", summary["strategy_rows"])
    (args.result_dir / "aggregate_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Loaded {len(records)} completed batches from {args.result_dir}")
    print(f"Wrote {args.result_dir / 'aggregate_by_setting.csv'}")
    print(f"Wrote {args.result_dir / 'aggregate_by_strategy.csv'}")


if __name__ == "__main__":
    main()
