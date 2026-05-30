#!/usr/bin/env python3
"""Parse Chrome traces from 15_trace_ulysses.py and extract per-op timings.

For each (case, rank, iter, direction, op-name) record_function window
(uly_{fwd,bwd}_{q,k,v,o}_a2a, uly_fwd_local_attn), we compute:

  cpu_window_us  : record_function CPU duration  (= our visible time)
  gpu_nccl_us    : sum of NCCL kernels within the window  (= actual on-GPU comm)
  gpu_flash_us   : sum of FlashAttention kernels within the window
                   (only meaningful for uly_fwd_local_attn / bwd flash steps)
  gpu_cmp_us     : sum of non-nccl, non-flash CUDA kernels (reshapes etc.)
  exposed_us     : cpu_window_us - max(gpu_nccl, gpu_flash) - launch_us
                   (CPU work not hidden by GPU work)
  launch_us      : sum of all "*kernelLaunch*" kernels  (or 0)

Outputs per (case, direction, op, rank) aggregated stats (mean over iters)
as a Markdown table and CSV.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics
import sys
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

Interval = Tuple[float, float]


def union(intervals: Sequence[Interval]) -> List[Interval]:
    if not intervals:
        return []
    ivs = sorted(intervals)
    out = [ivs[0]]
    for a, b in ivs[1:]:
        la, lb = out[-1]
        if a <= lb:
            out[-1] = (la, max(lb, b))
        else:
            out.append((a, b))
    return out


def union_length(intervals: Sequence[Interval]) -> float:
    return sum(b - a for a, b in union(intervals))


def clip(intervals: Sequence[Interval], lo: float, hi: float) -> List[Interval]:
    out = []
    for a, b in intervals:
        a2, b2 = max(a, lo), min(b, hi)
        if a2 < b2:
            out.append((a2, b2))
    return out


NCCL_RE = re.compile(r"nccl|send_recv|Memcpy.*P2P", re.IGNORECASE)
FLASH_RE = re.compile(r"flash[_-]?attn|fmha|flash_(fwd|bwd)", re.IGNORECASE)


def classify(name: str) -> str:
    if NCCL_RE.search(name):
        return "nccl"
    if FLASH_RE.search(name):
        return "flash"
    return "other"


_NAMED_OPS = [
    "uly_fwd_q_a2a", "uly_fwd_k_a2a", "uly_fwd_v_a2a",
    "uly_fwd_local_attn", "uly_fwd_o_a2a",
    "uly_bwd_q_a2a", "uly_bwd_k_a2a", "uly_bwd_v_a2a", "uly_bwd_o_a2a",
    # full fwd / bwd windows (for sanity)
    "uly_fwd", "uly_bwd",
]


def parse_trace(path: str):
    with open(path) as fh:
        data = json.load(fh)
    events = data.get("traceEvents", [])
    cpu_rf: List[Tuple[str, float, float]] = []
    gpu_k: List[Tuple[str, float, float]] = []
    for ev in events:
        if ev.get("ph") != "X":
            continue
        dur = float(ev.get("dur", 0) or 0)
        if dur <= 0:
            continue
        ts = float(ev.get("ts", 0) or 0)
        name = ev.get("name") or ""
        cat = (ev.get("cat") or "").lower()
        if "user_annotation" in cat or "annotation" in cat:
            if "kernel" in cat:  # gpu_user_annotation
                continue
            cpu_rf.append((name, ts, dur))
        elif "kernel" in cat:
            gpu_k.append((name, ts, dur))
    return cpu_rf, gpu_k


def find_op_windows(cpu_rf, op_name: str) -> List[Interval]:
    """Return ALL (lo, hi) windows for an op marker, sorted by ts."""
    out = []
    for n, ts, dur in cpu_rf:
        if n == op_name:
            out.append((ts, ts + dur))
    out.sort()
    return out


def analyze_trace(path: str, meta: dict) -> List[Dict]:
    cpu_rf, gpu_k = parse_trace(path)
    nccl_iv = [(ts, ts + dur) for n, ts, dur in gpu_k if classify(n) == "nccl"]
    flash_iv = [(ts, ts + dur) for n, ts, dur in gpu_k if classify(n) == "flash"]
    other_iv = [(ts, ts + dur) for n, ts, dur in gpu_k if classify(n) == "other"]

    rows = []
    for op in _NAMED_OPS:
        windows = find_op_windows(cpu_rf, op)
        for iter_idx, (lo, hi) in enumerate(windows):
            nccl = union_length(clip(nccl_iv, lo, hi))
            flash = union_length(clip(flash_iv, lo, hi))
            other = union_length(clip(other_iv, lo, hi))
            cpu = hi - lo
            # Exposed CPU time = cpu - max(gpu work)
            gpu_busy = max(nccl, flash, other)
            exposed = max(0.0, cpu - gpu_busy)
            rows.append({
                **meta,
                "op": op,
                "iter": iter_idx,
                "cpu_us": cpu,
                "nccl_us": nccl,
                "flash_us": flash,
                "other_us": other,
                "exposed_us": exposed,
            })
    return rows


def parse_case_dir(case_dir: str) -> Tuple[int, int]:
    """Parse 'seq8192_sp16' -> (8192, 16)."""
    m = re.match(r"seq(\d+)_sp(\d+)", os.path.basename(case_dir))
    if not m:
        return (0, 0)
    return int(m.group(1)), int(m.group(2))


def parse_rank(trace_path: str) -> int:
    base = os.path.basename(trace_path)
    m = re.search(r"rank(\d+)", base)
    return int(m.group(1)) if m else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-root", required=True,
                    help="Path containing seq*_sp*/ directories of *.pt.trace.json")
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args()

    case_dirs = sorted(glob.glob(os.path.join(args.trace_root, "seq*_sp*")))
    if not case_dirs:
        sys.exit(f"no case dirs in {args.trace_root}")

    all_rows = []
    for cd in case_dirs:
        seq, sp = parse_case_dir(cd)
        trace_paths = sorted(glob.glob(os.path.join(cd, "*.pt.trace.json")))
        for tp in trace_paths:
            rank = parse_rank(tp)
            meta = {"seq": seq, "sp": sp, "rank": rank,
                    "trace": os.path.basename(tp)}
            try:
                rows = analyze_trace(tp, meta)
                all_rows.extend(rows)
            except Exception as e:
                print(f"[warn] failed {tp}: {e}", file=sys.stderr)

    if args.out_csv:
        with open(args.out_csv, "w") as fh:
            if all_rows:
                fields = list(all_rows[0].keys())
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader()
                w.writerows(all_rows)
        print(f"[wrote] {args.out_csv}")

    # ---------- Aggregated summary per (case, op): mean across iters, rank 0 only ----------
    agg: Dict[Tuple[int, int, str], Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list))
    for r in all_rows:
        if r["rank"] != 0:
            continue
        # Skip the first iteration -- often has kernel-load and warmup noise
        if r["iter"] == 0:
            continue
        key = (r["seq"], r["sp"], r["op"])
        for f in ("cpu_us", "nccl_us", "flash_us", "exposed_us"):
            agg[key][f].append(r[f])

    lines = []
    lines.append("# Ulysses trace analysis (rank 0, iter>=1)")
    lines.append("")
    # Compose grouped table per case
    cases = sorted({(s, sp) for (s, sp, _) in agg.keys()})
    for (seq, sp) in cases:
        lines.append(f"## seq={seq} sp={sp}")
        lines.append("")
        lines.append("| op | cpu_us | nccl_us | flash_us | exposed_cpu_us |")
        lines.append("|----|-------:|--------:|---------:|---------------:|")
        for op in _NAMED_OPS:
            key = (seq, sp, op)
            if key not in agg:
                continue
            stats = agg[key]
            row = (f"| {op} "
                   f"| {statistics.mean(stats['cpu_us']):>8.1f} "
                   f"| {statistics.mean(stats['nccl_us']):>8.1f} "
                   f"| {statistics.mean(stats['flash_us']):>8.1f} "
                   f"| {statistics.mean(stats['exposed_us']):>13.1f} |")
            lines.append(row)
        lines.append("")
        # Also summary totals
        total_cpu = sum(
            statistics.mean(agg[(seq, sp, op)]["cpu_us"])
            for op in _NAMED_OPS if (seq, sp, op) in agg
            and op not in ("uly_fwd", "uly_bwd")
        )
        total_nccl = sum(
            statistics.mean(agg[(seq, sp, op)]["nccl_us"])
            for op in _NAMED_OPS if (seq, sp, op) in agg
            and op not in ("uly_fwd", "uly_bwd")
        )
        total_exposed = sum(
            statistics.mean(agg[(seq, sp, op)]["exposed_us"])
            for op in _NAMED_OPS if (seq, sp, op) in agg
            and op not in ("uly_fwd", "uly_bwd")
        )
        if (seq, sp, "uly_fwd") in agg:
            full_fwd = statistics.mean(agg[(seq, sp, "uly_fwd")]["cpu_us"])
        else:
            full_fwd = 0
        if (seq, sp, "uly_bwd") in agg:
            full_bwd = statistics.mean(agg[(seq, sp, "uly_bwd")]["cpu_us"])
        else:
            full_bwd = 0
        lines.append(f"**totals (μs)**: sum-of-op-cpu={total_cpu:.0f}  "
                     f"sum-of-op-nccl={total_nccl:.0f}  "
                     f"sum-of-exposed-cpu={total_exposed:.0f}  "
                     f"full-fwd={full_fwd:.0f}  full-bwd={full_bwd:.0f}")
        lines.append("")

    out_md = args.out_md or os.path.join(args.trace_root, "summary.md")
    with open(out_md, "w") as fh:
        fh.write("\n".join(lines))
    print(f"[wrote] {out_md}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
