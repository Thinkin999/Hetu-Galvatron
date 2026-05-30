#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parse Chrome traces produced by 08_trace_ring_attention.py and quantify
per-step compute / comm / overlap behaviour for Ring attention.

Hypothesis dashboard:
  H1  overlap_slowdown is not 1.1; it grows toward ~2.0 as min(c,m)/max(c,m) -> 0
      (short-sequence GPU resource contention).
  H2  bwd ring per-step is not fwd × 3.2 (ratio actually depends on cp_size / local_seq).
  H3  per-step flash compute is not f(S/cp): zigzag step has 3 shapes
      (diagonal causal / half-rectangle / transposed half-rect).
  H4  there is a real per-step wrapper overhead (CPU launch latency, sync, buffer alloc).

The script extracts the following per (case, rank, direction, step) row:

  seq_len, cp_size, topology, rank, direction, step
  step_total_us         CPU record_function window for the whole step
  p2p_launch_us         CPU record_function for "p2p_launch" sub-phase
  compute_phase_us      CPU record_function for "compute" sub-phase
  wait_phase_us         CPU record_function for "wait" sub-phase
  gpu_flash_us          Union of FlashAttention CUDA kernels in [step_lo, step_hi]
  gpu_nccl_us           Union of NCCL CUDA kernels   in [step_lo, step_hi]
  gpu_overlap_us        |flash ∩ nccl| on the GPU timeline
  overlap_ratio_min     gpu_overlap_us / min(flash, nccl)         (0=serial, 1=perfect overlap)
  eff_slowdown          (step_total_us - |flash - nccl|) / min(flash, nccl)
                        - matches the cost-model `overlap_slowdown` semantics:
                          step ≈ tail + slowdown * min(c, m)
                        - >1 ⇒ overlap is worse than perfect; 2 ⇒ fully serial.

Outputs:
  --output-csv: one row per (case, rank, direction, step)
  --output-summary-md: aggregated table grouped by (seq_len, cp_size, direction)
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple


Interval = Tuple[float, float]


# ---------------------------------------------------------------------------
# Interval helpers
# ---------------------------------------------------------------------------

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


def intersect(a: Sequence[Interval], b: Sequence[Interval]) -> List[Interval]:
    a, b = union(a), union(b)
    i = j = 0
    out: List[Interval] = []
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if lo < hi:
            out.append((lo, hi))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def clip(intervals: Sequence[Interval], lo: float, hi: float) -> List[Interval]:
    out = []
    for a, b in intervals:
        a2, b2 = max(a, lo), min(b, hi)
        if a2 < b2:
            out.append((a2, b2))
    return out


# ---------------------------------------------------------------------------
# Trace event classification
# ---------------------------------------------------------------------------

NCCL_PATTERNS = [
    re.compile(r"ncclDevKernel", re.IGNORECASE),
    re.compile(r"^ncclKernel", re.IGNORECASE),
    re.compile(r"nccl[A-Z]", re.IGNORECASE),
    re.compile(r"^nccl::", re.IGNORECASE),
    re.compile(r"send_recv|Memcpy.*P2P", re.IGNORECASE),
]
FLASH_PATTERNS = [
    re.compile(r"flash[_-]?attn", re.IGNORECASE),
    re.compile(r"fmha", re.IGNORECASE),
    re.compile(r"_fwd_kernel|_bwd_kernel", re.IGNORECASE),
    re.compile(r"flash_fwd|flash_bwd", re.IGNORECASE),
]


def classify_kernel(name: str) -> str:
    for p in NCCL_PATTERNS:
        if p.search(name):
            return "nccl"
    for p in FLASH_PATTERNS:
        if p.search(name):
            return "flash"
    return "other"


def parse_trace(path: str) -> Tuple[List[Tuple[str, float, float]], List[Tuple[str, float, float]]]:
    """Return (cpu_record_functions, gpu_kernels) as lists of (name, ts_us, dur_us)."""
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
            cpu_rf.append((name, ts, dur))
        elif "kernel" in cat or cat == "gpu_user_annotation":
            # gpu_user_annotation: record_function projected onto GPU timeline.
            # We want raw CUDA kernels: cat=="kernel" in torch.profiler output.
            if "kernel" in cat:
                gpu_k.append((name, ts, dur))
    return cpu_rf, gpu_k


# ---------------------------------------------------------------------------
# Step window extraction
# ---------------------------------------------------------------------------

_STEP_FULL_RE = {
    "fwd": re.compile(r"^ring_fwd_step_(\d+)$"),
    "bwd": re.compile(r"^ring_bwd_step_(\d+)$"),
}
_PHASE_RE = {
    direction: {
        phase: re.compile(rf"^ring_{direction}_step_(\d+)_{phase}$")
        for phase in ("p2p_launch", "compute", "wait")
    }
    for direction in ("fwd", "bwd")
}


def collect_step_windows(
    cpu_rf: Sequence[Tuple[str, float, float]],
    direction: str,
) -> Dict[Tuple[int, int], Interval]:
    """Return {(iter_idx, step): (ts_lo, ts_hi)}.

    When --active > 1, each step appears once per iteration. We assign iter_idx
    by order of appearance per step number, so iter 0 is the first time we see
    `ring_{dir}_step_0`, iter 1 is the second, etc.
    """
    pat = _STEP_FULL_RE[direction]
    per_step: Dict[int, List[Interval]] = defaultdict(list)
    for name, ts, dur in cpu_rf:
        m = pat.match(name)
        if m:
            per_step[int(m.group(1))].append((ts, ts + dur))
    for s in per_step:
        per_step[s].sort()
    out: Dict[Tuple[int, int], Interval] = {}
    for step, ivs in per_step.items():
        for idx, iv in enumerate(ivs):
            out[(idx, step)] = iv
    return out


def collect_phase_windows(
    cpu_rf: Sequence[Tuple[str, float, float]],
    direction: str,
    phase: str,
) -> Dict[Tuple[int, int], Interval]:
    pat = _PHASE_RE[direction][phase]
    per_step: Dict[int, List[Interval]] = defaultdict(list)
    for name, ts, dur in cpu_rf:
        m = pat.match(name)
        if m:
            per_step[int(m.group(1))].append((ts, ts + dur))
    for s in per_step:
        per_step[s].sort()
    out: Dict[Tuple[int, int], Interval] = {}
    for step, ivs in per_step.items():
        for idx, iv in enumerate(ivs):
            out[(idx, step)] = iv
    return out


# ---------------------------------------------------------------------------
# Per-trace analysis
# ---------------------------------------------------------------------------

def analyze_trace(path: str, meta: Dict) -> List[Dict]:
    cpu_rf, gpu_k = parse_trace(path)
    nccl_intervals = [(ts, ts + dur) for n, ts, dur in gpu_k if classify_kernel(n) == "nccl"]
    flash_intervals = [(ts, ts + dur) for n, ts, dur in gpu_k if classify_kernel(n) == "flash"]

    rows: List[Dict] = []
    for direction in ("fwd", "bwd"):
        steps = collect_step_windows(cpu_rf, direction)
        if not steps:
            continue
        phases = {
            phase: collect_phase_windows(cpu_rf, direction, phase)
            for phase in ("p2p_launch", "compute", "wait")
        }
        for (iter_idx, step), (lo, hi) in sorted(steps.items()):
            nccl_in = clip(nccl_intervals, lo, hi)
            flash_in = clip(flash_intervals, lo, hi)
            flash_us = union_length(flash_in)
            nccl_us = union_length(nccl_in)
            overlap_us = union_length(intersect(nccl_in, flash_in))
            step_total = hi - lo
            tail = abs(flash_us - nccl_us)
            min_cm = min(flash_us, nccl_us)
            max_cm = max(flash_us, nccl_us)
            overlap_ratio_min = overlap_us / min_cm if min_cm > 0 else 0.0
            eff_slowdown = None
            if min_cm > 0:
                eff_slowdown = max(0.0, (step_total - tail) / min_cm)

            def _phase_dur(phase: str) -> float:
                w = phases[phase].get((iter_idx, step))
                return (w[1] - w[0]) if w else 0.0

            rows.append({
                **meta,
                "iter_idx": iter_idx,
                "direction": direction,
                "step": step,
                "step_total_us": step_total,
                "p2p_launch_us": _phase_dur("p2p_launch"),
                "compute_phase_us": _phase_dur("compute"),
                "wait_phase_us": _phase_dur("wait"),
                "gpu_flash_us": flash_us,
                "gpu_nccl_us": nccl_us,
                "gpu_overlap_us": overlap_us,
                "gpu_max_us": max_cm,
                "overlap_ratio_min": overlap_ratio_min,
                "eff_slowdown": eff_slowdown,
            })
    return rows


# ---------------------------------------------------------------------------
# CLI / sweep entry point
# ---------------------------------------------------------------------------

CASE_DIR_RE = re.compile(r"seq(\d+)_cp(\d+)_([A-Za-z]+)$")
RANK_FILE_RE = re.compile(r"rank(\d+)\.[\d\.\-]+\.pt\.trace\.json$")


def discover_trace_files(trace_root: str, ranks: Optional[Sequence[int]]) -> List[Tuple[Dict, str]]:
    out: List[Tuple[Dict, str]] = []
    rank_filter = set(ranks) if ranks else None
    for case_dir in sorted(glob.glob(os.path.join(trace_root, "seq*_cp*_*"))):
        base = os.path.basename(case_dir)
        m = CASE_DIR_RE.match(base)
        if not m:
            continue
        seq_len = int(m.group(1))
        cp_size = int(m.group(2))
        topology = m.group(3)
        for trace_file in sorted(glob.glob(os.path.join(case_dir, "*.pt.trace.json"))):
            fname = os.path.basename(trace_file)
            rm = RANK_FILE_RE.match(fname)
            if rm is None:
                # Best effort: also accept files like "rank0.something.pt.trace.json"
                guess = re.match(r"rank(\d+)", fname)
                if guess is None:
                    continue
                rank = int(guess.group(1))
            else:
                rank = int(rm.group(1))
            if rank_filter is not None and rank not in rank_filter:
                continue
            out.append((
                {
                    "seq_len": seq_len,
                    "cp_size": cp_size,
                    "topology": topology,
                    "rank": rank,
                    "trace_file": trace_file,
                },
                trace_file,
            ))
    return out


def write_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _agg(values: List[float], stat: str = "median") -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    if stat == "median":
        return statistics.median(vals)
    if stat == "mean":
        return statistics.mean(vals)
    raise ValueError(stat)


def _keep_typical(rows: List[Dict]) -> List[Dict]:
    """Drop step==0 (no prior P2P to overlap with) and iter_idx==0 (cold start).

    Returns the "typical" Ring steps used for overlap statistics. If after this
    filter we have nothing left (e.g. active=1), fall back to step>=1 only.
    """
    typical = [r for r in rows if r["step"] >= 1 and r.get("iter_idx", 0) >= 1]
    if typical:
        return typical
    fallback = [r for r in rows if r["step"] >= 1]
    return fallback if fallback else rows


def write_summary(rows: List[Dict], path: str) -> None:
    by_case_dir: Dict[Tuple[int, int, str], List[Dict]] = defaultdict(list)
    for r in rows:
        by_case_dir[(r["seq_len"], r["cp_size"], r["direction"])].append(r)

    with open(path, "w") as fh:
        fh.write("# Ring Attention Trace Summary\n\n")
        fh.write("Each row aggregates the *typical* Ring steps for one "
                 "(seq_len, cp_size, direction): step>=1 and iter_idx>=1 "
                 "(drops the cold-start iteration and the launch-only step 0). "
                 "If only one active iter was profiled, iter_idx==0 is kept.\n\n")
        fh.write("Columns are medians (µs) unless stated. `eff_slowdown` is the "
                 "effective overlap_slowdown the cost model would need: 1.0 = "
                 "perfect, 2.0 = serial.\n\n")
        fh.write("| seq | cp | dir | n | step_total | gpu_flash | gpu_nccl | gpu_overlap |"
                 " overlap_ratio_min | eff_slowdown |\n")
        fh.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for (seq, cp, direction), rs in sorted(by_case_dir.items()):
            keep = _keep_typical(rs)
            step_total = _agg([r["step_total_us"] for r in keep])
            flash = _agg([r["gpu_flash_us"] for r in keep])
            nccl = _agg([r["gpu_nccl_us"] for r in keep])
            ov = _agg([r["gpu_overlap_us"] for r in keep])
            ratio = _agg([r["overlap_ratio_min"] for r in keep])
            slowdown = _agg([r["eff_slowdown"] for r in keep])
            fh.write(
                f"| {seq} | {cp} | {direction} | {len(keep)} | "
                f"{step_total:.1f} | {flash:.1f} | {nccl:.1f} | {ov:.1f} | "
                f"{ratio:.3f} | "
                f"{(f'{slowdown:.3f}' if slowdown is not None else 'n/a')} |\n"
            )

        seqs = sorted({r["seq_len"] for r in rows})
        cps = sorted({r["cp_size"] for r in rows})

        fh.write("\n## Hypothesis snapshot\n\n")
        fh.write("### H1: GPU overlap_ratio_min vs (seq, cp) — fwd\n\n")
        fh.write("If H1 holds, this number should be near 1.0 for long seq /"
                 " small cp (perfect overlap), and approach 0 for short seq /"
                 " large cp (overlap fails).\n\n")
        fh.write("| seq \\ cp | " + " | ".join(str(c) for c in cps) + " |\n")
        fh.write("|" + "|".join(["---"] * (1 + len(cps))) + "|\n")
        for seq in seqs:
            row_vals = []
            for cp in cps:
                cell = _keep_typical([
                    r for r in rows
                    if r["seq_len"] == seq and r["cp_size"] == cp and r["direction"] == "fwd"
                ])
                v = _agg([r["overlap_ratio_min"] for r in cell])
                row_vals.append(f"{v:.2f}" if v is not None else "n/a")
            fh.write(f"| {seq} | " + " | ".join(row_vals) + " |\n")

        fh.write("\n### H1b: eff_slowdown vs (seq, cp) — fwd\n\n")
        fh.write("Effective `overlap_slowdown` the cost model would need.\n\n")
        fh.write("| seq \\ cp | " + " | ".join(str(c) for c in cps) + " |\n")
        fh.write("|" + "|".join(["---"] * (1 + len(cps))) + "|\n")
        for seq in seqs:
            row_vals = []
            for cp in cps:
                cell = _keep_typical([
                    r for r in rows
                    if r["seq_len"] == seq and r["cp_size"] == cp and r["direction"] == "fwd"
                ])
                v = _agg([r["eff_slowdown"] for r in cell])
                row_vals.append(f"{v:.2f}" if v is not None else "n/a")
            fh.write(f"| {seq} | " + " | ".join(row_vals) + " |\n")

        fh.write("\n### H2: ring bwd_step / fwd_step ratio\n\n")
        fh.write("Cost model assumes `bwd_step ≈ 3.2 × fwd_step`.\n\n")
        fh.write("| seq \\ cp | " + " | ".join(str(c) for c in cps) + " |\n")
        fh.write("|" + "|".join(["---"] * (1 + len(cps))) + "|\n")
        for seq in seqs:
            row_vals = []
            for cp in cps:
                fwd_vals = _keep_typical([
                    r for r in rows
                    if r["seq_len"] == seq and r["cp_size"] == cp and r["direction"] == "fwd"
                ])
                bwd_vals = _keep_typical([
                    r for r in rows
                    if r["seq_len"] == seq and r["cp_size"] == cp and r["direction"] == "bwd"
                ])
                fwd = _agg([r["step_total_us"] for r in fwd_vals])
                bwd = _agg([r["step_total_us"] for r in bwd_vals])
                if fwd and bwd:
                    row_vals.append(f"{bwd / fwd:.2f}")
                else:
                    row_vals.append("n/a")
            fh.write(f"| {seq} | " + " | ".join(row_vals) + " |\n")

        fh.write("\n### H4: per-step overhead = step_total − max(flash, nccl) — fwd\n\n")
        fh.write("Time inside the step that is not explained by GPU flash or nccl"
                 " kernels (µs).\n\n")
        fh.write("| seq \\ cp | " + " | ".join(str(c) for c in cps) + " |\n")
        fh.write("|" + "|".join(["---"] * (1 + len(cps))) + "|\n")
        for seq in seqs:
            row_vals = []
            for cp in cps:
                cell = _keep_typical([
                    r for r in rows
                    if r["seq_len"] == seq and r["cp_size"] == cp and r["direction"] == "fwd"
                ])
                overheads = [
                    r["step_total_us"] - r["gpu_max_us"] for r in cell
                ]
                v = _agg(overheads)
                row_vals.append(f"{v:.1f}" if v is not None else "n/a")
            fh.write(f"| {seq} | " + " | ".join(row_vals) + " |\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-root", required=True,
                    help="Directory containing seq{S}_cp{C}_{topo}/ subdirs.")
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--output-summary-md", required=True)
    ap.add_argument("--ranks", type=int, nargs="*", default=None,
                    help="Only analyze these ranks (default: all found).")
    args = ap.parse_args()

    pairs = discover_trace_files(args.trace_root, args.ranks)
    if not pairs:
        raise SystemExit(f"No trace files under {args.trace_root}")

    rows: List[Dict] = []
    for meta, path in pairs:
        try:
            rows.extend(analyze_trace(path, meta))
        except Exception as exc:  # noqa: BLE001
            print(f"[!] failed {path}: {exc}")
    if not rows:
        raise SystemExit("No rows extracted; check categorization patterns.")
    write_csv(rows, args.output_csv)
    write_summary(rows, args.output_summary_md)
    print(f"[OK] {len(rows)} rows -> {args.output_csv}")
    print(f"[OK] summary -> {args.output_summary_md}")


if __name__ == "__main__":
    main()
