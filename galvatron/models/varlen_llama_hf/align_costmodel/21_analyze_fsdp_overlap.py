"""Quantify compute / NCCL overlap from FSDP training step traces.

Reads chrome traces produced by 20_trace_fsdp_step_dispatch.sh (timeline
profiler) and reports per-iteration:

  - total compute kernel time on GPU (matmul, flash_attn, elementwise, etc.)
  - total NCCL kernel time on GPU
  - exposed NCCL time (NCCL running while no compute kernel is concurrently
    running on the compute stream)
  - hidden NCCL time = total NCCL - exposed
  - breakdown of NCCL by kernel category (AllGather / ReduceScatter /
    AllToAll / SendRecv / AllReduce / other)
  - saturation_ratio = exposed / total NCCL (lower is better)

Designed for Stage 0 of non-attention residual modeling: validates the
"compute hides ZeRO3 all-gather / reduce-scatter" assumption.

Usage:
  python 21_analyze_fsdp_overlap.py <trace_root>
where <trace_root> is e.g. results/fsdp_step_*/traces. The tool will walk
subdirectories (seq<L>_<strategy>/) and pick rank0 traces.
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Tuple


# --------------------------- trace IO --------------------------- #

def load_trace(path: str) -> dict:
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as f:
            return json.load(f)
    with open(path) as f:
        return json.load(f)


# --------------------------- classification --------------------------- #

NCCL_RE = re.compile(r"(ncclKernel|ncclDevKernel|nccl::|c10d::)", re.IGNORECASE)


def classify_nccl(name: str) -> str | None:
    """Return one of {AllGather, ReduceScatter, AllReduce, AllToAll, SendRecv,
    Broadcast, Other}, or None if not an NCCL kernel."""
    n = (name or "")
    if not NCCL_RE.search(n) and "AllToAll" not in n and "_send_recv" not in n:
        # Heuristic for raw NCCL kernel names without nccl prefix:
        return None
    low = n.lower()
    if "allgather" in low or "all_gather" in low:
        return "AllGather"
    if "reducescatter" in low or "reduce_scatter" in low:
        return "ReduceScatter"
    if "allreduce" in low or "all_reduce" in low:
        return "AllReduce"
    if "alltoall" in low or "all_to_all" in low:
        return "AllToAll"
    if "send" in low or "recv" in low or "p2p" in low:
        return "SendRecv"
    if "broadcast" in low:
        return "Broadcast"
    return "OtherNCCL"


COMPUTE_RE_PARTS = [
    "gemm", "matmul", "cublas", "cutlass",
    "flash", "FlashAttn", "fmha", "mha_fwd", "mha_bwd",
    "softmax", "rmsnorm", "layernorm", "layer_norm",
    "elementwise", "vectorized", "reduce_kernel",
    "rope", "rotary",
    "transpose", "copy", "memcpy", "fill",
    "embedding", "scatter", "gather",
    "triton",  # triton-generated act fn / fused kernels
    "multi_tensor", "cunn_Soft", "CatArray", "indexSelect",
    "nll_loss", "RadixSort", "DeviceScan", "Compact", "Unique",
    "cleanup",
]
COMPUTE_RE = re.compile("|".join(COMPUTE_RE_PARTS), re.IGNORECASE)


def is_compute(name: str) -> bool:
    if not name:
        return False
    if classify_nccl(name) is not None:
        return False
    return bool(COMPUTE_RE.search(name)) or _looks_like_kernel(name)


def _looks_like_kernel(name: str) -> bool:
    # Catch CUDA kernels with C++ mangling we didn't enumerate.
    if name.startswith("void ") or name.startswith("__nv_"):
        return True
    if "<<<" in name:
        return True
    return False


# --------------------------- interval analysis --------------------------- #

@dataclass
class Interval:
    start: float  # us
    end: float    # us

    @property
    def dur(self) -> float:
        return self.end - self.start


def merge(intervals: List[Interval]) -> List[Interval]:
    if not intervals:
        return []
    sorted_iv = sorted(intervals, key=lambda x: x.start)
    out = [Interval(sorted_iv[0].start, sorted_iv[0].end)]
    for iv in sorted_iv[1:]:
        if iv.start <= out[-1].end:
            out[-1].end = max(out[-1].end, iv.end)
        else:
            out.append(Interval(iv.start, iv.end))
    return out


def total_dur(intervals: List[Interval]) -> float:
    return sum(iv.dur for iv in intervals)


def difference(a: List[Interval], b: List[Interval]) -> List[Interval]:
    """Return a \\ b (subtract b from a). Assumes a, b already merged."""
    out: List[Interval] = []
    j = 0
    for iv in a:
        cur_start = iv.start
        cur_end = iv.end
        while j < len(b) and b[j].end <= cur_start:
            j += 1
        k = j
        while k < len(b) and b[k].start < cur_end:
            bk = b[k]
            if bk.start > cur_start:
                out.append(Interval(cur_start, min(bk.start, cur_end)))
            cur_start = max(cur_start, bk.end)
            if cur_start >= cur_end:
                break
            k += 1
        if cur_start < cur_end:
            out.append(Interval(cur_start, cur_end))
    return out


# --------------------------- trace walking --------------------------- #

ITER_SPAN_CANDIDATES = [
    # Each candidate is a tuple of marker names that bound a single iteration.
    # We try them in order; first one that yields >=1 span wins.
    ("adacpsp::forward_backward",),  # fwd+bwd only (the main interest)
    ("ProfilerStep#",),
    ("adacpsp::solve_and_dispatch",),
]


def find_iteration_spans(events: list, label_out: dict | None = None) -> List[Interval]:
    """Locate iteration spans. Tries several candidate markers in order."""
    for cand_tuple in ITER_SPAN_CANDIDATES:
        iters: List[Interval] = []
        for e in events:
            if e.get("ph") != "X":
                continue
            name = e.get("name", "")
            if any(name.startswith(c) or name == c for c in cand_tuple):
                iters.append(Interval(e["ts"], e["ts"] + e.get("dur", 0)))
        if iters:
            if label_out is not None:
                label_out["marker"] = cand_tuple[0]
            return sorted(iters, key=lambda x: x.start)
    return []


def gpu_kernel_events(events: list) -> list:
    out = []
    for e in events:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat", "")
        if "kernel" not in cat:
            continue
        out.append(e)
    return out


def per_iteration_breakdown(events: list, iters: List[Interval]) -> List[dict]:
    """For each iteration span, compute compute / NCCL / exposed-NCCL totals."""
    gpu_ev = gpu_kernel_events(events)
    # Pre-tag events for speed.
    tagged = []
    for e in gpu_ev:
        nccl_cls = classify_nccl(e.get("name", ""))
        is_cmp = (nccl_cls is None) and is_compute(e.get("name", ""))
        tagged.append((e["ts"], e["ts"] + e.get("dur", 0), nccl_cls, is_cmp, e.get("name", "")))

    out = []
    for iter_idx, span in enumerate(iters):
        compute_iv: List[Interval] = []
        nccl_iv: List[Interval] = []
        nccl_by_class = collections.defaultdict(list)  # class -> intervals
        unclass_names = collections.Counter()
        for s, t, ncl, is_cmp, name in tagged:
            # Restrict to events whose body is inside span [span.start, span.end]
            if t <= span.start or s >= span.end:
                continue
            ss = max(s, span.start)
            tt = min(t, span.end)
            if tt <= ss:
                continue
            iv = Interval(ss, tt)
            if ncl is not None:
                nccl_iv.append(iv)
                nccl_by_class[ncl].append(iv)
            elif is_cmp:
                compute_iv.append(iv)
            else:
                # Untagged GPU activity — count separately
                unclass_names[name] += 1

        compute_merged = merge(compute_iv)
        nccl_merged = merge(nccl_iv)
        exposed = difference(nccl_merged, compute_merged)
        nccl_class_totals = {}
        for cls, ivs in nccl_by_class.items():
            nccl_class_totals[cls] = total_dur(merge(ivs))

        out.append({
            "iter_idx": iter_idx,
            "span_start_us": span.start,
            "span_end_us": span.end,
            "span_dur_us": span.dur,
            "compute_us": total_dur(compute_merged),
            "nccl_us": total_dur(nccl_merged),
            "exposed_nccl_us": total_dur(exposed),
            "hidden_nccl_us": total_dur(nccl_merged) - total_dur(exposed),
            "saturation_ratio": (total_dur(exposed) / total_dur(nccl_merged)) if total_dur(nccl_merged) > 0 else 0.0,
            "nccl_by_class_us": nccl_class_totals,
            "unclass_top": unclass_names.most_common(8),
        })
    return out


# --------------------------- pretty printing --------------------------- #

def fmt_ms(us: float) -> str:
    return f"{us/1000:7.2f}ms"


def print_summary(case_label: str, rank: int, breakdown: List[dict]) -> None:
    print(f"\n=== {case_label}  (rank{rank}, {len(breakdown)} iter spans) ===")
    if not breakdown:
        print("  no ProfilerStep markers found")
        return
    header = (
        f"{'iter':>4s} {'span':>10s} {'compute':>10s} {'nccl':>10s} "
        f"{'exposed':>10s} {'hidden':>10s} {'sat%':>6s}"
    )
    print(header)
    for it in breakdown:
        sat = it["saturation_ratio"] * 100.0
        print(
            f"{it['iter_idx']:>4d} "
            f"{fmt_ms(it['span_dur_us']):>10s} "
            f"{fmt_ms(it['compute_us']):>10s} "
            f"{fmt_ms(it['nccl_us']):>10s} "
            f"{fmt_ms(it['exposed_nccl_us']):>10s} "
            f"{fmt_ms(it['hidden_nccl_us']):>10s} "
            f"{sat:>5.1f}%"
        )
        cls = it["nccl_by_class_us"]
        if cls:
            parts = [f"{k}={fmt_ms(v).strip()}" for k, v in sorted(cls.items(), key=lambda kv: -kv[1])]
            print("       breakdown: " + ", ".join(parts))
    # Aggregate across iterations (drop iter 0 if outside trace span).
    use = [it for it in breakdown if it["span_dur_us"] > 0]
    if use:
        avg = {
            "compute_us": sum(it["compute_us"] for it in use) / len(use),
            "nccl_us": sum(it["nccl_us"] for it in use) / len(use),
            "exposed_nccl_us": sum(it["exposed_nccl_us"] for it in use) / len(use),
            "hidden_nccl_us": sum(it["hidden_nccl_us"] for it in use) / len(use),
            "span_dur_us": sum(it["span_dur_us"] for it in use) / len(use),
        }
        avg["saturation_ratio"] = (avg["exposed_nccl_us"] / avg["nccl_us"]) if avg["nccl_us"] > 0 else 0.0
        print(
            f"avg:                {fmt_ms(avg['span_dur_us'])} {fmt_ms(avg['compute_us'])} "
            f"{fmt_ms(avg['nccl_us'])} {fmt_ms(avg['exposed_nccl_us'])} "
            f"{fmt_ms(avg['hidden_nccl_us'])} {avg['saturation_ratio']*100:.1f}%"
        )


# --------------------------- main --------------------------- #

def discover_traces(trace_root: str) -> List[Tuple[str, str, int]]:
    """Walk trace_root; return (case_label, trace_file, rank)."""
    out: List[Tuple[str, str, int]] = []
    for case_dir in sorted(glob.glob(os.path.join(trace_root, "*"))):
        if not os.path.isdir(case_dir):
            continue
        # Look in case_dir/* for rank-named traces
        files = (glob.glob(os.path.join(case_dir, "**", "*.pt.trace.json*"), recursive=True)
                 or glob.glob(os.path.join(case_dir, "**", "*.json*"), recursive=True))
        rank_files = {}
        for f in files:
            m = re.search(r"rank(\d+)", f)
            if not m:
                continue
            r = int(m.group(1))
            rank_files.setdefault(r, []).append(f)
        for r in sorted(rank_files.keys()):
            out.append((os.path.basename(case_dir), sorted(rank_files[r])[0], r))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("trace_root")
    p.add_argument("--json-out", default=None)
    p.add_argument("--ranks", default=None, help="space-separated ranks to include")
    args = p.parse_args()

    allowed_ranks = None
    if args.ranks:
        allowed_ranks = set(int(x) for x in args.ranks.replace(",", " ").split() if x)

    cases = discover_traces(args.trace_root)
    if not cases:
        print(f"No traces found under {args.trace_root}")
        return 1

    json_out = {}
    for case_label, trace_file, rank in cases:
        if allowed_ranks is not None and rank not in allowed_ranks:
            continue
        try:
            t = load_trace(trace_file)
        except Exception as exc:
            print(f"[warn] failed to load {trace_file}: {exc}")
            continue
        events = t.get("traceEvents", [])
        label_info = {}
        iters = find_iteration_spans(events, label_info)
        bd = per_iteration_breakdown(events, iters)
        marker = label_info.get("marker", "<unknown>")
        case_full = f"{case_label}  [iter_marker={marker}]"
        print_summary(case_full, rank, bd)
        json_out[f"{case_label}__rank{rank}"] = {
            "trace_file": trace_file,
            "iterations": bd,
        }

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(json_out, f, indent=2)
        print(f"\nWrote JSON to {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
