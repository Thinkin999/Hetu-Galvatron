#!/usr/bin/env python3
"""
Analyze experiment logs for FlexSP vs AdaCPSP comparison.

Usage:
  python analyze_experiment_logs.py --log_dir logs/
  python analyze_experiment_logs.py --log_dir logs/ --output results.md
"""

import os
import re
import sys
import argparse
import glob
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════
# LLaMA-7B
PARAM_B = 7.0
FLOPS_PER_TOKEN = 6 * PARAM_B * 1e9  # forward + backward approximation
PEAK_TFLOPS_PER_GPU = 312  # A100-SXM4 bf16
N_GPUS = 8

WARMUP_ITERS = 5  # Skip first N iters for performance statistics


def parse_log(log_path: str) -> Dict:
    """Parse a single experiment log file."""
    result = {
        "path": log_path,
        "name": os.path.basename(log_path).replace(".log", ""),
        "iter_times_ms": [],
        "losses": [],
        "avg_iter_time_s": None,
        "strategies": [],
        "errors": [],
        "max_seq": None,
        "gbs": None,
        "layers": None,
        "attn_types": None,
        "memory_snapshots": [],
    }

    if not os.path.exists(log_path):
        result["errors"].append(f"File not found: {log_path}")
        return result

    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()

            # ── Parse iteration time ──
            # "| Iteration:      1 | Consumed samples:           32 | Elapsed time per iteration (ms): 1234.5 | ..."
            m = re.search(r'Elapsed time per iteration \(ms\):\s*([\d.]+)', line)
            if m:
                result["iter_times_ms"].append(float(m.group(1)))

            # ── Parse average iteration time (from profiler summary) ──
            # "Average iteration time is: 1.2345 s"
            m = re.search(r'Average iteration time is:\s*([\d.]+)\s*s', line)
            if m:
                result["avg_iter_time_s"] = float(m.group(1))

            # ── Parse loss ──
            # "Loss: 1.234567e+01" or "[Epoch 0] (Iteration 0): Loss = 11.234"
            m = re.search(r'Loss[:=]\s*([\d.eE+-]+)', line)
            if m:
                try:
                    result["losses"].append(float(m.group(1)))
                except ValueError:
                    pass

            # ── Parse AdaCPSP strategy ──
            # "[AdaCPSP] MB0: type=ulysses, sp=8, cp=1"
            m = re.search(r'\[AdaCPSP\] MB(\d+): type=(\w+), sp=(\d+), cp=(\d+)', line)
            if m:
                result["strategies"].append({
                    "mb": int(m.group(1)),
                    "attn_type": m.group(2),
                    "sp": int(m.group(3)),
                    "cp": int(m.group(4)),
                })

            # ── Parse config info ──
            m = re.search(r'seq_length[=:]\s*(\d+)', line)
            if m:
                result["max_seq"] = int(m.group(1))

            m = re.search(r'global_train_batch_size[=:]\s*(\d+)', line)
            if m:
                result["gbs"] = int(m.group(1))

            m = re.search(r'num_hidden_layers[=:]\s*(\d+)', line)
            if m:
                result["layers"] = int(m.group(1))

            m = re.search(r'attn_types?[=:]\s*\[([^\]]+)\]', line)
            if m:
                result["attn_types"] = m.group(1)

            # ── Parse memory ──
            m = re.search(r'After Backward.*?(\d+\.\d+)\s*MB', line)
            if m:
                result["memory_snapshots"].append(float(m.group(1)))

            # ── Parse errors ──
            if "CUDA out of memory" in line or "RuntimeError" in line or "NCCL" in line.upper() and "error" in line.lower():
                result["errors"].append(line[:200])

    return result


def compute_metrics(result: Dict) -> Dict:
    """Compute derived metrics from parsed log."""
    metrics = {}

    iter_times = result["iter_times_ms"]
    if len(iter_times) > WARMUP_ITERS:
        # Skip warmup
        stable_times = iter_times[WARMUP_ITERS:]
        metrics["avg_iter_ms"] = sum(stable_times) / len(stable_times)
        metrics["min_iter_ms"] = min(stable_times)
        metrics["max_iter_ms"] = max(stable_times)
        metrics["std_iter_ms"] = (sum((t - metrics["avg_iter_ms"])**2 for t in stable_times) / len(stable_times)) ** 0.5
        metrics["n_stable_iters"] = len(stable_times)
    elif len(iter_times) > 0:
        metrics["avg_iter_ms"] = sum(iter_times) / len(iter_times)
        metrics["n_stable_iters"] = len(iter_times)
    else:
        metrics["avg_iter_ms"] = None
        metrics["n_stable_iters"] = 0

    # Use profiler's average if available (more accurate, uses CUDA events)
    if result["avg_iter_time_s"] is not None:
        metrics["profiler_avg_iter_ms"] = result["avg_iter_time_s"] * 1000

    # MFU calculation (needs GBS to estimate tokens per iter)
    # This is a rough estimate; actual tokens vary per iter in varlen
    gbs = result.get("gbs")
    avg_ms = metrics.get("avg_iter_ms")
    if gbs and avg_ms and avg_ms > 0:
        # For varlen, we don't know exact tokens per iter from log
        # Use GBS * avg_seq_len estimate — this will be refined by user
        # Here we just record the raw iter time for comparison
        pass

    # Strategy distribution
    strat_counts = defaultdict(int)
    for s in result["strategies"]:
        key = f"{s['attn_type']}×{s['sp']}×{s['cp']}"
        strat_counts[key] += 1
    metrics["strategy_distribution"] = dict(strat_counts)

    # Memory
    if result["memory_snapshots"]:
        metrics["peak_memory_mb"] = max(result["memory_snapshots"])

    # Loss
    if result["losses"]:
        metrics["first_loss"] = result["losses"][0]
        metrics["last_loss"] = result["losses"][-1]
        metrics["loss_decreased"] = result["losses"][-1] < result["losses"][0] if len(result["losses"]) > 1 else None

    return metrics


def pair_experiments(log_dir: str) -> List[Tuple[Dict, Dict]]:
    """Find and pair flexsp/adacpsp logs for the same experiment ID."""
    logs = sorted(glob.glob(os.path.join(log_dir, "exp_*_*.log")))
    
    # Group by experiment ID
    exp_groups = defaultdict(dict)
    for log_path in logs:
        name = os.path.basename(log_path).replace(".log", "")
        # exp_E1_short_flexsp -> id=E1_short, mode=flexsp
        parts = name.split("_")
        # Find the mode (last part)
        if parts[-1] in ("flexsp", "adacpsp"):
            mode = parts[-1]
            exp_id = "_".join(parts[1:-1])  # E1_short, E2_mixed, etc.
            exp_groups[exp_id][mode] = log_path

    pairs = []
    for exp_id in sorted(exp_groups.keys()):
        group = exp_groups[exp_id]
        flex_path = group.get("flexsp")
        ada_path = group.get("adacpsp")
        
        flex_result = parse_log(flex_path) if flex_path else None
        ada_result = parse_log(ada_path) if ada_path else None

        pairs.append((exp_id, flex_result, ada_result))

    return pairs


def format_report(pairs: List, output_path: Optional[str] = None):
    """Format and print/save the comparison report."""
    lines = []
    
    def p(s=""):
        lines.append(s)
        print(s)

    p("=" * 80)
    p("  FlexSP vs AdaCPSP — Experiment Results")
    p("=" * 80)
    p()

    # ── Summary Table ──
    p("┌────────────────┬──────────────────┬──────────────────┬──────────┐")
    p("│ Experiment     │ FlexSP (ms/iter) │ AdaCPSP(ms/iter) │ Speedup  │")
    p("├────────────────┼──────────────────┼──────────────────┼──────────┤")

    summary_rows = []
    for exp_id, flex, ada in pairs:
        flex_m = compute_metrics(flex) if flex else {}
        ada_m = compute_metrics(ada) if ada else {}

        flex_avg = flex_m.get("profiler_avg_iter_ms") or flex_m.get("avg_iter_ms")
        ada_avg = ada_m.get("profiler_avg_iter_ms") or ada_m.get("avg_iter_ms")

        flex_str = f"{flex_avg:10.1f} ms" if flex_avg else "      N/A"
        ada_str = f"{ada_avg:10.1f} ms" if ada_avg else "      N/A"

        if flex_avg and ada_avg and ada_avg > 0:
            speedup = flex_avg / ada_avg
            speedup_str = f"{speedup:6.3f}×"
        else:
            speedup = None
            speedup_str = "   N/A"

        p(f"│ {exp_id:14s} │ {flex_str:>16s} │ {ada_str:>16s} │ {speedup_str:>8s} │")
        summary_rows.append((exp_id, flex_avg, ada_avg, speedup))

    p("└────────────────┴──────────────────┴──────────────────┴──────────┘")
    p()

    # ── Detailed per-experiment ──
    for exp_id, flex, ada in pairs:
        p(f"\n{'─'*60}")
        p(f"  Experiment: {exp_id}")
        p(f"{'─'*60}")

        for label, result in [("FlexSP", flex), ("AdaCPSP", ada)]:
            if result is None:
                p(f"\n  [{label}] — NO LOG FOUND")
                continue

            m = compute_metrics(result)
            p(f"\n  [{label}] {result['name']}")

            # Config
            cfg_parts = []
            if result.get("max_seq"): cfg_parts.append(f"max_seq={result['max_seq']}")
            if result.get("gbs"): cfg_parts.append(f"GBS={result['gbs']}")
            if result.get("layers"): cfg_parts.append(f"layers={result['layers']}")
            if cfg_parts:
                p(f"    Config: {', '.join(cfg_parts)}")

            # Iteration times
            avg = m.get("avg_iter_ms")
            if avg:
                p(f"    Avg iter time: {avg:.1f} ms "
                  f"(min={m.get('min_iter_ms', 0):.1f}, max={m.get('max_iter_ms', 0):.1f}, "
                  f"std={m.get('std_iter_ms', 0):.1f}, "
                  f"n={m.get('n_stable_iters', 0)} stable iters)")

            prof_avg = m.get("profiler_avg_iter_ms")
            if prof_avg:
                p(f"    Profiler avg:  {prof_avg:.1f} ms")

            # Strategy distribution
            strats = m.get("strategy_distribution", {})
            if strats:
                p(f"    Strategies:")
                total = sum(strats.values())
                for k, v in sorted(strats.items(), key=lambda x: -x[1]):
                    p(f"      {k:20s}: {v:4d} ({v/total*100:5.1f}%)")

            # Memory
            peak = m.get("peak_memory_mb")
            if peak:
                p(f"    Peak memory: {peak:.0f} MB ({peak/1024:.1f} GB)")

            # Loss
            if m.get("first_loss") is not None:
                arrow = "↓" if m.get("loss_decreased") else "↑" if m.get("loss_decreased") is False else "?"
                p(f"    Loss: {m['first_loss']:.4f} → {m['last_loss']:.4f} {arrow}")

            # Errors
            if result["errors"]:
                p(f"    ⚠ ERRORS ({len(result['errors'])}):")
                for err in result["errors"][:5]:
                    p(f"      {err}")

    # ── MFU Estimates ──
    p(f"\n{'='*60}")
    p("  MFU Estimation (approximate)")
    p(f"{'='*60}")
    p("  Note: For varlen datasets, tokens/iter varies.")
    p("  MFU below uses GBS × max_seq as upper bound (actual MFU is higher).")
    p()

    peak_tflops = PEAK_TFLOPS_PER_GPU * N_GPUS
    for exp_id, flex, ada in pairs:
        flex_m = compute_metrics(flex) if flex else {}
        ada_m = compute_metrics(ada) if ada else {}

        gbs = (flex or ada or {}).get("gbs")
        max_seq = (flex or ada or {}).get("max_seq")
        if not gbs or not max_seq:
            continue

        # Upper bound tokens (actual is less due to padding/shorter seqs)
        max_tokens = gbs * max_seq

        for label, m_dict in [("FlexSP", flex_m), ("AdaCPSP", ada_m)]:
            avg = m_dict.get("profiler_avg_iter_ms") or m_dict.get("avg_iter_ms")
            if avg and avg > 0:
                mfu_upper = max_tokens * FLOPS_PER_TOKEN / (avg * 1e-3) / (peak_tflops * 1e12) * 100
                throughput = max_tokens / (avg * 1e-3)
                p(f"  {exp_id:14s} {label:8s}: {throughput/1e3:.1f}K tok/s, MFU≤{mfu_upper:.1f}%")

    p()

    # ── Recommendations ──
    p(f"\n{'='*60}")
    p("  Recommendations")
    p(f"{'='*60}")

    any_ring = False
    for exp_id, flex, ada in pairs:
        if ada:
            m = compute_metrics(ada)
            for k in m.get("strategy_distribution", {}):
                if "ring" in k or "usp" in k:
                    any_ring = True
                    break

    if any_ring:
        p("  ★ AdaCPSP selected Ring/USP strategies in some experiments!")
        p("    This indicates potential benefit from mixed strategies.")
    else:
        p("  ★ AdaCPSP converged to Ulysses-only (same as FlexSP).")
        p("    On NVLink-connected GPUs, Ulysses is consistently optimal.")
        p("    Ring/USP advantages expected in cross-machine or GQA scenarios.")

    p()

    if output_path:
        with open(output_path, 'w') as f:
            f.write("\n".join(lines))
        print(f"\nReport saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze FlexSP vs AdaCPSP experiment logs")
    parser.add_argument("--log_dir", type=str, default="logs/",
                       help="Directory containing experiment logs")
    parser.add_argument("--output", type=str, default=None,
                       help="Save report to file (optional)")
    args = parser.parse_args()

    if not os.path.isdir(args.log_dir):
        print(f"Error: Log directory '{args.log_dir}' does not exist.")
        print(f"Run experiments first: cd llama_scripts && bash exp_run_all.sh")
        sys.exit(1)

    log_files = glob.glob(os.path.join(args.log_dir, "exp_*.log"))
    if not log_files:
        print(f"No experiment logs found in {args.log_dir}")
        print(f"Run experiments first: cd llama_scripts && bash exp_run_all.sh")
        sys.exit(1)

    print(f"Found {len(log_files)} log files in {args.log_dir}")
    
    pairs = pair_experiments(args.log_dir)
    format_report(pairs, args.output)


if __name__ == "__main__":
    main()

