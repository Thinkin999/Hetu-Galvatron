"""Parse 17_trace_usp chrome trace files and break down per-op time.

Decomposes one fwd+bwd iteration into:
  uly_fwd_q_a2a, uly_fwd_k_a2a, uly_fwd_v_a2a, uly_fwd_local_attn,
  uly_fwd_o_a2a, uly_bwd_q_a2a, uly_bwd_k_a2a, uly_bwd_v_a2a, uly_bwd_o_a2a.

For each marker, reports CPU duration, NCCL device time, FlashAttention device time.
"""

import argparse
import collections
import glob
import gzip
import json
import os


def load_trace(path: str):
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as f:
            return json.load(f)
    return json.load(open(path))


MARKERS = [
    "uly_pad_q", "uly_pad_k", "uly_pad_v",
    "uly_fwd_q_a2a", "uly_fwd_k_a2a", "uly_fwd_v_a2a", "uly_fwd_local_attn",
    "uly_fwd_o_a2a", "uly_unpad_o",
    "uly_bwd_q_a2a", "uly_bwd_k_a2a", "uly_bwd_v_a2a", "uly_bwd_o_a2a",
    "usp_fwd", "usp_bwd",
]


def in_range(child, parent):
    return child["ts"] >= parent["ts"] and child["ts"] + child.get("dur", 0) <= \
            parent["ts"] + parent.get("dur", 0)


def is_nccl(name: str) -> bool:
    n = name or ""
    return ("ncclKernel" in n or "ncclDevKernel" in n or "nccl::" in n
            or "AllToAll" in n or "Send" in n and "ncclSend" in n
            or "recv" in n.lower() and "nccl" in n.lower())


def is_flash(name: str) -> bool:
    return name and ("flash" in name.lower() or "FlashAttn" in name)


def analyze(trace_json):
    events = trace_json.get("traceEvents", [])
    # CPU markers (cat user_annotation)
    cpu_markers = [e for e in events if e.get("ph") == "X"
                   and e.get("cat") in ("user_annotation",)
                   and e.get("name") in MARKERS]
    # Sort by ts
    cpu_markers.sort(key=lambda e: e["ts"])

    # Build per-marker CPU duration aggregated (per occurrence).
    # We'll group by name, average across iterations.
    by_name = collections.defaultdict(list)
    for m in cpu_markers:
        by_name[m["name"]].append(m["dur"])

    # GPU activity events on CUDA device (cat = kernel)
    gpu_events = [e for e in events if e.get("ph") == "X" and e.get("cat") == "kernel"]
    # We also need CUDA runtime/api events to correlate. But here let's just
    # measure overlap of GPU events within each CPU marker span using a
    # "flow" or just by timestamp. Trace's CPU markers are wall-clock; GPU
    # events run async. So we can approximate by counting GPU activity that
    # started while CPU marker was running OR within a short delay.

    # We'll measure per-marker GPU NCCL + flash by intersecting GPU event
    # spans with CPU marker spans (approximation).
    gpu_by_marker_nccl = collections.defaultdict(list)
    gpu_by_marker_flash = collections.defaultdict(list)
    gpu_by_marker_other = collections.defaultdict(list)
    for m in cpu_markers:
        m_start = m["ts"]
        m_end = m["ts"] + m["dur"]
        for g in gpu_events:
            g_start = g["ts"]
            g_end = g["ts"] + g.get("dur", 0)
            # GPU event must overlap with CPU marker
            if g_end < m_start or g_start > m_end:
                continue
            if is_nccl(g.get("name", "")):
                gpu_by_marker_nccl[m["name"]].append(g.get("dur", 0))
            elif is_flash(g.get("name", "")):
                gpu_by_marker_flash[m["name"]].append(g.get("dur", 0))
            else:
                gpu_by_marker_other[m["name"]].append(g.get("dur", 0))

    return by_name, gpu_by_marker_nccl, gpu_by_marker_flash, gpu_by_marker_other


def fmt_us(us):
    return f"{us/1000:.2f}ms" if us > 1000 else f"{us:.0f}us"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("trace_dir")
    args = p.parse_args()

    # Find all rank0 traces
    cases = sorted(glob.glob(os.path.join(args.trace_dir, "seq*_sp*_cp*")))
    for case in cases:
        # Pick rank0 trace
        trace_files = sorted(glob.glob(os.path.join(case, "*rank0*.pt.trace*")))
        if not trace_files:
            trace_files = sorted(glob.glob(os.path.join(case, "*rank0*.json*")))
        if not trace_files:
            continue
        print(f"\n=== {os.path.basename(case)} ===")
        print(f"  trace = {os.path.basename(trace_files[0])}")
        t = load_trace(trace_files[0])
        cpu, nccl, flash, other = analyze(t)

        all_markers = [m for m in MARKERS if m in cpu]
        print(f"{'marker':25s} {'iter':>5s} {'cpu_dur (avg)':>14s} {'nccl_gpu':>14s} {'flash_gpu':>14s} {'other_gpu':>14s}")
        for name in all_markers:
            durs = cpu.get(name, [])
            n_iters = len(durs)
            if not durs:
                continue
            cpu_avg = sum(durs) / n_iters
            nccl_total = sum(nccl.get(name, [])) / max(1, n_iters)
            flash_total = sum(flash.get(name, [])) / max(1, n_iters)
            other_total = sum(other.get(name, [])) / max(1, n_iters)
            print(f"{name:25s} {n_iters:>5d} {fmt_us(cpu_avg):>14s} "
                  f"{fmt_us(nccl_total):>14s} {fmt_us(flash_total):>14s} "
                  f"{fmt_us(other_total):>14s}")


if __name__ == "__main__":
    main()
