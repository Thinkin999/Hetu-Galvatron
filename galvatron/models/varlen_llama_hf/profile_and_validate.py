#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AdaCPSP Comprehensive Profiling & Validation Suite
===================================================
1. Attention profiling with AUTOMATIC breakpoint detection (time/x² derivative)
2. Communication profiling with linear fitting y = a*x + b
3. CostModel validation: predicted vs measured (Ulysses / Ring / USP)
4. Memory model validation
5. End-to-end test with real varlen dataset (wikipedia/common_crawl/github)

Usage:
  Single GPU (attention only):
    python profile_and_validate.py --mode attention --n_heads 32 --head_dim 128

  8 GPUs (comm profiling + cost model validation):
    torchrun --nproc_per_node=8 profile_and_validate.py --mode comm
    torchrun --nproc_per_node=8 profile_and_validate.py --mode validate_cost_model
    torchrun --nproc_per_node=8 profile_and_validate.py --mode validate_memory
    torchrun --nproc_per_node=8 profile_and_validate.py --mode all
"""

import os
import sys
import json
import argparse
import math
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime

import torch

# ─── Flash Attention imports ──────────────────────────────────────────
HAS_FLASH_ATTN = False
flash_attn_func = None
flash_attn_varlen_func = None

try:
    from flash_attn import flash_attn_func as _fa_func
    from flash_attn import flash_attn_varlen_func as _fa_varlen_func
    flash_attn_func = _fa_func
    flash_attn_varlen_func = _fa_varlen_func
    HAS_FLASH_ATTN = True
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════════════
# PART 1: Attention Profiling with Automatic Breakpoint Detection
# ═══════════════════════════════════════════════════════════════════════

def _extract_attention_xy(data: List[Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize attention profiling records into aligned seq/time arrays."""
    xs, ts = [], []
    for item in data:
        if isinstance(item, dict):
            xs.append(item["seq_len"])
            ts.append(item["time_ms"])
        else:
            xs.append(item[0])
            ts.append(item[1])
    return np.array(xs, dtype=np.float64), np.array(ts, dtype=np.float64)


def _segment_mask(xs: np.ndarray, lo: float, hi: float, include_hi: bool) -> np.ndarray:
    """Build a half-open mask for segments, only keeping the last hi inclusive."""
    if include_hi:
        return (xs >= lo) & (xs <= hi)
    return (xs >= lo) & (xs < hi)


def _eval_quadratic(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Evaluate quadratic in the original seq_len basis."""
    return a * x ** 2 + b * x + c


def _fit_centered_quadratic(seg_x: np.ndarray, seg_t: np.ndarray) -> Dict[str, Any]:
    """Fit quadratic on centered/scaled seq_len for better numerical stability."""
    x_center = float((seg_x.min() + seg_x.max()) / 2.0)
    x_scale = float(max(1.0, (seg_x.max() - seg_x.min()) / 2.0))
    z = (seg_x - x_center) / x_scale

    alpha, beta, gamma = np.polyfit(z, seg_t, deg=2)
    y_pred = alpha * z ** 2 + beta * z + gamma

    ss_res = np.sum((seg_t - y_pred) ** 2)
    ss_tot = np.sum((seg_t - np.mean(seg_t)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Convert local basis back into the original seq_len basis.
    a = alpha / (x_scale ** 2)
    b = beta / x_scale - 2.0 * alpha * x_center / (x_scale ** 2)
    c = gamma - beta * x_center / x_scale + alpha * (x_center ** 2) / (x_scale ** 2)

    return {
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "r_squared": float(r2),
        "max_error_ms": float(np.max(np.abs(seg_t - y_pred))),
        "mean_error_ms": float(np.mean(np.abs(seg_t - y_pred))),
        "y_pred": y_pred,
        "fit_basis": {
            "type": "centered_quadratic",
            "variable": "z=(seq_len-center)/scale",
            "center": x_center,
            "scale": x_scale,
        },
        "local_fit_params": {
            "alpha": float(alpha),
            "beta": float(beta),
            "gamma": float(gamma),
        },
    }


def _serialize_attention_measurements(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert attention measurements into JSON-friendly objects."""
    output = []
    for item in data:
        output.append({
            "seq_len": int(item["seq_len"]),
            "time_ms": float(item["time_ms"]),
            "mean_ms": float(item["mean_ms"]),
            "std_ms": float(item["std_ms"]),
            "min_ms": float(item["min_ms"]),
            "max_ms": float(item["max_ms"]),
            "samples_ms": [float(v) for v in item["samples_ms"]],
            "timing_groups": int(item["timing_groups"]),
            "base_iters_per_group": int(item["base_iters_per_group"]),
            "iters_per_group": int(item["iters_per_group"]),
            "actual_profile_iters": int(item["actual_profile_iters"]),
            "pilot_total_ms": float(item["pilot_total_ms"]),
        })
    return output


def _profile_attention_point(
    seq_len: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    warmup: int,
    iters: int,
    timing_groups: int,
    timing_stat: str,
    min_group_elapsed_ms: float,
    max_iters_per_group: int,
    use_varlen: bool,
    device: str,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    """Profile one seq_len and return robust timing statistics."""
    if use_varlen and flash_attn_varlen_func is not None:
        q = torch.randn(seq_len, n_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(seq_len, n_kv_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(seq_len, n_kv_heads, head_dim, dtype=dtype, device=device)
        cu = torch.tensor([0, seq_len], dtype=torch.int32, device=device)

        def attn_call():
            return flash_attn_varlen_func(q, k, v, cu, cu, seq_len, seq_len, causal=True)

        cleanup_tensors = (q, k, v, cu)
    elif flash_attn_func is not None:
        q = torch.randn(1, seq_len, n_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(1, seq_len, n_kv_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(1, seq_len, n_kv_heads, head_dim, dtype=dtype, device=device)

        def attn_call():
            return flash_attn_func(q, k, v, causal=True)

        cleanup_tensors = (q, k, v)
    else:
        raise RuntimeError("No flash_attn implementation available")

    try:
        for _ in range(warmup):
            attn_call()
        torch.cuda.synchronize()

        group_count = max(1, min(timing_groups, iters))
        base_iters_per_group = max(1, int(math.ceil(iters / group_count)))
        iters_per_group = base_iters_per_group
        samples_ms = []

        # Pilot timing to ensure each measurement window is long enough.
        pilot_start = torch.cuda.Event(enable_timing=True)
        pilot_end = torch.cuda.Event(enable_timing=True)
        pilot_start.record()
        for _ in range(base_iters_per_group):
            attn_call()
        pilot_end.record()
        torch.cuda.synchronize()
        pilot_total_ms = max(pilot_start.elapsed_time(pilot_end), 1e-6)
        if pilot_total_ms < min_group_elapsed_ms:
            scale = int(math.ceil(min_group_elapsed_ms / pilot_total_ms))
            iters_per_group = min(max_iters_per_group, max(base_iters_per_group, base_iters_per_group * scale))

        for _ in range(group_count):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters_per_group):
                attn_call()
            end.record()
            torch.cuda.synchronize()
            samples_ms.append(start.elapsed_time(end) / iters_per_group)

        samples_arr = np.array(samples_ms, dtype=np.float64)
        if timing_stat == "mean":
            time_ms = float(np.mean(samples_arr))
        else:
            time_ms = float(np.median(samples_arr))

        return {
            "seq_len": int(seq_len),
            "time_ms": time_ms,
            "mean_ms": float(np.mean(samples_arr)),
            "std_ms": float(np.std(samples_arr)),
            "min_ms": float(np.min(samples_arr)),
            "max_ms": float(np.max(samples_arr)),
            "samples_ms": [float(v) for v in samples_ms],
            "timing_groups": int(group_count),
            "base_iters_per_group": int(base_iters_per_group),
            "iters_per_group": int(iters_per_group),
            "actual_profile_iters": int(group_count * iters_per_group),
            "pilot_total_ms": float(pilot_total_ms),
        }
    finally:
        for tensor in cleanup_tensors:
            del tensor
        torch.cuda.empty_cache()


def profile_attention_dense(
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    seq_range: Tuple[int, int] = (64, 32768),
    step: int = 64,
    warmup: int = 5,
    iters: int = 20,
    timing_groups: int = 3,
    timing_stat: str = "median",
    min_group_elapsed_ms: float = 1.0,
    max_iters_per_group: int = 512,
    use_varlen: bool = True,
    device: str = "cuda",
    dtype=torch.bfloat16,
) -> List[Dict[str, Any]]:
    """
    Densely profile Flash Attention across the full seq_len range.
    Returns list of per-seq measurements with robust timing statistics.
    """
    results = []
    lo, hi = seq_range
    seq_lengths = list(range(lo, hi + 1, step))
    print(f"\n[Attention Profiling] Dense sampling: {lo} -> {hi}, step={step}, "
          f"total={len(seq_lengths)} points, {'varlen' if use_varlen else 'padded'}")
    print(f"  Timing groups: {timing_groups}, aggregation: {timing_stat}, "
          f"requested iters per point: {iters}")
    print(f"  Adaptive timing: min_group_elapsed_ms={min_group_elapsed_ms}, "
          f"max_iters_per_group={max_iters_per_group}")

    for i, seq_len in enumerate(seq_lengths):
        try:
            if flash_attn_varlen_func is None and flash_attn_func is None:
                print("  No flash_attn available, aborting")
                return results

            measurement = _profile_attention_point(
                seq_len=seq_len,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                warmup=warmup,
                iters=iters,
                timing_groups=timing_groups,
                timing_stat=timing_stat,
                min_group_elapsed_ms=min_group_elapsed_ms,
                max_iters_per_group=max_iters_per_group,
                use_varlen=use_varlen,
                device=device,
                dtype=dtype,
            )
            results.append(measurement)
            if (i + 1) % 50 == 0 or i == 0:
                print(f"  [{i+1}/{len(seq_lengths)}] seq={seq_len}: "
                      f"{measurement['time_ms']:.4f} ms "
                      f"(std={measurement['std_ms']:.4f}, groups={measurement['timing_groups']})")

        except RuntimeError as ex:
            if "out of memory" in str(ex).lower():
                print(f"  OOM at seq_len={seq_len}, stopping")
                torch.cuda.empty_cache()
                break
            raise

    print(f"  Collected {len(results)} data points")
    return results


def detect_breakpoints(
    data: List[Any],
    window: int = 5,
    threshold_factor: float = 3.0,
    min_segment_points: int = 8,
) -> List[int]:
    """
    Automatically detect kernel switch breakpoints by analyzing time/x².
    
    Method:
    1. Compute normalized metric: r(x) = time(x) / x²
       If flash attention uses a single O(n²) kernel, r(x) should be roughly constant.
       Kernel switches cause discontinuities in r(x).
    
    2. Compute sliding-window derivative |Δr / Δx| 
    
    3. Find peaks in the derivative that exceed threshold_factor × median
       These peaks indicate breakpoints where the kernel changes.
    
    4. Merge nearby breakpoints and ensure minimum segment size.
    
    Returns: sorted list of breakpoint seq_lens (excluding start/end).
    """
    if len(data) < 2 * window + min_segment_points:
        return []

    xs, ts = _extract_attention_xy(data)

    # 1. Compute r(x) = time / x²
    r = ts / (xs ** 2)
    r_smooth = np.zeros_like(r)
    for i in range(len(r)):
        lo = max(0, i - window)
        hi = min(len(r), i + window + 1)
        r_smooth[i] = np.median(r[lo:hi])

    # 2. Compute smoothed derivative of r
    dr = np.zeros(len(r))
    for i in range(window, len(r) - window):
        r_left = np.mean(r_smooth[max(0, i - window):i])
        r_right = np.mean(r_smooth[i:min(len(r_smooth), i + window)])
        dx = xs[min(len(xs)-1, i + window//2)] - xs[max(0, i - window//2)]
        if dx > 0:
            dr[i] = abs(r_right - r_left) / dx
        else:
            dr[i] = 0

    # 3. Find peaks: dr > threshold_factor * median(dr[nonzero])
    dr_valid = dr[window:-window]
    if len(dr_valid) == 0:
        return []
    
    median_dr = np.median(dr_valid[dr_valid > 0]) if np.any(dr_valid > 0) else 0
    threshold = threshold_factor * median_dr if median_dr > 0 else np.inf

    peaks = []
    for i in range(window, len(dr) - window):
        if dr[i] > threshold:
            # Check it's a local maximum
            if dr[i] >= max(dr[max(0, i-2):i]) and dr[i] >= max(dr[i+1:min(len(dr), i+3)]):
                peaks.append(int(xs[i]))

    # 4. Merge nearby peaks (within 2*step of each other)
    if not peaks:
        return []
    
    step = int(xs[1] - xs[0]) if len(xs) > 1 else 64
    merged = [peaks[0]]
    for p in peaks[1:]:
        if p - merged[-1] > 4 * step:
            merged.append(p)
        else:
            # Keep the one with higher derivative
            idx_old = int(np.argmin(np.abs(xs - merged[-1])))
            idx_new = int(np.argmin(np.abs(xs - p)))
            if dr[idx_new] > dr[idx_old]:
                merged[-1] = p

    # 5. Filter: ensure minimum segment size
    filtered = []
    prev = int(xs[0])
    for bp in merged:
        if bp - prev >= min_segment_points * step:
            filtered.append(bp)
            prev = bp
    # Also ensure last segment has enough points
    if filtered and int(xs[-1]) - filtered[-1] < min_segment_points * step:
        filtered.pop()

    print(f"\n[Breakpoint Detection] Detected {len(filtered)} breakpoints: {filtered}")
    print(f"  Segments: ", end="")
    bounds = [int(xs[0])] + filtered + [int(xs[-1])]
    for i in range(len(bounds) - 1):
        pts = sum(1 for x in xs if bounds[i] <= x <= bounds[i+1])
        print(f"[{bounds[i]}, {bounds[i+1]}]({pts}pts) ", end="")
    print()

    return filtered


def consolidate_breakpoints(
    data: List[Any],
    raw_breakpoints: List[int],
    max_segments: int = 5,
    min_r2: float = 0.995,
) -> List[int]:
    """
    Consolidate many fine-grained breakpoints into a small practical set.
    
    Strategy: iteratively merge the pair of adjacent segments whose removal
    causes the least drop in overall R². Stop when we have ≤ max_segments
    or removing any breakpoint drops R² below min_r2.
    """
    xs, ts = _extract_attention_xy(data)

    def segmentation_stats(bounds: List[float]) -> Dict[str, Any]:
        fits = []
        r2s = []
        jumps = []
        rel_errs = []
        for j in range(len(bounds) - 1):
            lo, hi = bounds[j], bounds[j + 1]
            mask = _segment_mask(xs, lo, hi, include_hi=(j == len(bounds) - 2))
            sx, st = xs[mask], ts[mask]
            if len(sx) < 3:
                return {
                    "r2s": [0.0],
                    "jumps": [1.0],
                    "mean_rel_err": float("inf"),
                    "max_rel_err": float("inf"),
                }
            fit = _fit_centered_quadratic(sx, st)
            fits.append(fit)
            r2s.append(fit["r_squared"])
            y_pred = _eval_quadratic(sx, fit["a"], fit["b"], fit["c"])
            rel_errs.extend(np.abs(st - y_pred) / np.maximum(st, 1e-6) * 100.0)

        for j, bp in enumerate(bounds[1:-1]):
            left_fit = fits[j]
            right_fit = fits[j + 1]
            left_val = float(_eval_quadratic(np.array([bp], dtype=np.float64),
                                             left_fit["a"], left_fit["b"], left_fit["c"])[0])
            right_val = float(_eval_quadratic(np.array([bp], dtype=np.float64),
                                              right_fit["a"], right_fit["b"], right_fit["c"])[0])
            denom = max(abs(left_val), abs(right_val), 1e-6)
            jumps.append(abs(left_val - right_val) / denom)
        return {
            "r2s": r2s,
            "jumps": jumps,
            "mean_rel_err": float(np.mean(rel_errs)) if rel_errs else 0.0,
            "max_rel_err": float(np.max(rel_errs)) if rel_errs else 0.0,
        }

    def segment_r2(lo, hi):
        """Compute R² for a single segment [lo, hi]."""
        mask = _segment_mask(xs, lo, hi, include_hi=(hi >= xs[-1]))
        sx, st = xs[mask], ts[mask]
        if len(sx) < 3:
            return 0.0
        try:
            return _fit_centered_quadratic(sx, st)["r_squared"]
        except Exception:
            return 0.0

    current_bps = list(raw_breakpoints)
    start_x, end_x = float(xs[0]), float(xs[-1])

    while current_bps:
        current_bounds = [start_x] + current_bps + [end_x]
        current_stats = segmentation_stats(current_bounds)
        current_r2s = current_stats["r2s"]
        current_jumps = current_stats["jumps"]
        current_min_r2 = min(current_r2s) if current_r2s else 0.0
        current_max_jump = max(current_jumps) if current_jumps else 0.0
        current_score = current_min_r2 - 0.02 * current_max_jump

        need_reduce_segments = len(current_bps) + 1 > max_segments
        need_quality_improvement = current_min_r2 < min_r2
        if not need_reduce_segments and not need_quality_improvement:
            break

        # Try removing each breakpoint and compute the resulting score.
        best_remove_idx = -1
        best_score_after = -1e18
        best_candidate_stats = None

        for i in range(len(current_bps)):
            trial_bps = current_bps[:i] + current_bps[i+1:]
            bounds = [start_x] + trial_bps + [end_x]
            trial_stats = segmentation_stats(bounds)
            seg_r2s = trial_stats["r2s"]
            jumps = trial_stats["jumps"]
            min_r2_val = min(seg_r2s) if seg_r2s else 0
            max_jump = max(jumps) if jumps else 0.0
            score = min_r2_val - 0.02 * max_jump
            if score > best_score_after:
                best_score_after = score
                best_remove_idx = i
                best_candidate_stats = trial_stats

        if best_remove_idx < 0:
            break

        if not need_reduce_segments:
            if best_score_after <= current_score:
                break
            if best_candidate_stats["mean_rel_err"] > current_stats["mean_rel_err"] * 1.05:
                break
            if best_candidate_stats["max_rel_err"] > current_stats["max_rel_err"] * 1.10:
                break

        if (not need_reduce_segments) and best_score_after <= current_score:
            break

        removed = current_bps.pop(best_remove_idx)
        seg_r2s = best_candidate_stats["r2s"] if best_candidate_stats is not None else []
        jumps = best_candidate_stats["jumps"] if best_candidate_stats is not None else []
        print(f"  Merged: removed bp={int(removed)}, now {len(current_bps)+1} segments, "
              f"min_R²={min(seg_r2s):.5f}, max_jump={max(jumps) * 100 if jumps else 0.0:.2f}%, "
              f"mean_rel_err={best_candidate_stats['mean_rel_err']:.2f}%")

    print(f"\n[Consolidation] {len(raw_breakpoints)} → {len(current_bps)} breakpoints: {[int(b) for b in current_bps]}")
    bounds = [start_x] + current_bps + [end_x]
    for j in range(len(bounds) - 1):
        pts = sum(1 for x in xs if bounds[j] <= x <= bounds[j+1])
        r2 = segment_r2(bounds[j], bounds[j+1])
        print(f"  Segment [{int(bounds[j]):>6}, {int(bounds[j+1]):>6}]: {pts} pts, R²={r2:.6f}")

    return [int(b) for b in current_bps]


def fit_segments_quadratic(
    data: List[Any],
    breakpoints: List[int],
) -> List[Dict]:
    """
    Fit quadratic time = a*x² + b*x + c for each segment defined by breakpoints.
    """
    xs, ts = _extract_attention_xy(data)

    bounds_list = [xs[0]] + breakpoints + [xs[-1]]
    segments = []

    for i in range(len(bounds_list) - 1):
        lo, hi = bounds_list[i], bounds_list[i + 1]
        mask = _segment_mask(xs, lo, hi, include_hi=(i == len(bounds_list) - 2))
        seg_x = xs[mask]
        seg_t = ts[mask]

        if len(seg_x) < 3:
            print(f"  Segment [{int(lo)}, {int(hi)}]: only {len(seg_x)} points, skipping")
            continue

        try:
            fit = _fit_centered_quadratic(seg_x, seg_t)
            a, b, c = fit["a"], fit["b"], fit["c"]
            seg_info = {
                "range": [int(lo), int(hi)],
                "a": float(a),
                "b": float(b),
                "c": float(c),
                "r_squared": fit["r_squared"],
                "max_error_ms": fit["max_error_ms"],
                "mean_error_ms": fit["mean_error_ms"],
                "n_points": len(seg_x),
                "fit_basis": fit["fit_basis"],
                "local_fit_params": fit["local_fit_params"],
            }
            segments.append(seg_info)
            print(f"  Segment [{int(lo):>6}, {int(hi):>6}]: a={a:.6e}, b={b:.6e}, c={c:.4f}, "
                  f"R²={fit['r_squared']:.6f}, maxErr={fit['max_error_ms']:.4f}ms "
                  f"({len(seg_x)} pts, center={fit['fit_basis']['center']:.0f}, "
                  f"scale={fit['fit_basis']['scale']:.0f})")
        except Exception as e:
            print(f"  Segment [{int(lo)}, {int(hi)}]: fit failed: {e}")

    return segments


def summarize_attention_segments(
    data: List[Any],
    segments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build diagnostics for segment quality, continuity, and timing noise."""
    xs, ts = _extract_attention_xy(data)
    measurement_lookup = {
        int(item["seq_len"]): item for item in data if isinstance(item, dict)
    }

    per_segment = []
    boundary_jumps = []

    for idx, seg in enumerate(segments):
        lo, hi = seg["range"]
        mask = _segment_mask(xs, lo, hi, include_hi=(idx == len(segments) - 1))
        seg_x = xs[mask]
        seg_t = ts[mask]
        y_pred = _eval_quadratic(seg_x, seg["a"], seg["b"], seg["c"])
        rel_err_pct = np.abs(seg_t - y_pred) / np.maximum(seg_t, 1e-6) * 100.0
        monotonic_violations = int(np.sum(np.diff(y_pred) < -1e-6))
        stds = [
            measurement_lookup[int(x)]["std_ms"]
            for x in seg_x
            if int(x) in measurement_lookup
        ]
        per_segment.append({
            "range": [int(lo), int(hi)],
            "n_points": int(len(seg_x)),
            "mean_rel_error_pct": float(np.mean(rel_err_pct)) if len(rel_err_pct) else 0.0,
            "max_rel_error_pct": float(np.max(rel_err_pct)) if len(rel_err_pct) else 0.0,
            "mean_sample_std_ms": float(np.mean(stds)) if stds else None,
            "monotonic_violations": monotonic_violations,
        })

    for idx, bp in enumerate([seg["range"][1] for seg in segments[:-1]]):
        left = segments[idx]
        right = segments[idx + 1]
        left_val = float(_eval_quadratic(np.array([bp], dtype=np.float64),
                                         left["a"], left["b"], left["c"])[0])
        right_val = float(_eval_quadratic(np.array([bp], dtype=np.float64),
                                          right["a"], right["b"], right["c"])[0])
        denom = max(abs(left_val), abs(right_val), 1e-6)
        boundary_jumps.append({
            "breakpoint": int(bp),
            "left_range": left["range"],
            "right_range": right["range"],
            "left_ms": left_val,
            "right_ms": right_val,
            "abs_jump_ms": abs(left_val - right_val),
            "rel_jump_pct": abs(left_val - right_val) / denom * 100.0,
        })

    min_r2 = min((seg["r_squared"] for seg in segments), default=0.0)
    max_jump_pct = max((j["rel_jump_pct"] for j in boundary_jumps), default=0.0)
    mean_jump_pct = float(np.mean([j["rel_jump_pct"] for j in boundary_jumps])) if boundary_jumps else 0.0

    return {
        "per_segment": per_segment,
        "boundary_jumps": boundary_jumps,
        "selection_score": {
            "min_r_squared": float(min_r2),
            "max_boundary_rel_jump_pct": float(max_jump_pct),
            "mean_boundary_rel_jump_pct": float(mean_jump_pct),
        },
    }


def _predict_attention_piecewise(seq_len: int, segments: List[Dict[str, Any]]) -> float:
    """Evaluate piecewise quadratic attention fit for one seq_len."""
    if not segments:
        return 0.0
    for seg in segments:
        lo, hi = seg["range"]
        if lo <= seq_len <= hi:
            return float(_eval_quadratic(np.array([seq_len], dtype=np.float64),
                                         seg["a"], seg["b"], seg["c"])[0])
    if seq_len < segments[0]["range"][0]:
        seg = segments[0]
    else:
        seg = segments[-1]
    return float(_eval_quadratic(np.array([seq_len], dtype=np.float64),
                                 seg["a"], seg["b"], seg["c"])[0])


def validate_head_scaling(
    full_measurements: List[Dict[str, Any]],
    fitted_segments: List[Dict[str, Any]],
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    seq_lengths: List[int],
    sp_sizes: List[int],
    warmup: int = 5,
    iters: int = 20,
    timing_groups: int = 3,
    timing_stat: str = "median",
    min_group_elapsed_ms: float = 1.0,
    max_iters_per_group: int = 512,
    use_varlen: bool = True,
    device: str = "cuda",
    dtype=torch.bfloat16,
) -> List[Dict[str, Any]]:
    """Validate how well reduced local head counts follow simple Ulysses scaling."""
    full_lookup = {int(item["seq_len"]): item for item in full_measurements}
    results = []

    for sp in sp_sizes:
        if sp <= 1 or n_heads % sp != 0 or n_kv_heads % sp != 0:
            continue
        local_heads = n_heads // sp
        local_kv_heads = n_kv_heads // sp

        for seq_len in seq_lengths:
            if seq_len not in full_lookup:
                continue
            full_measured = full_lookup[seq_len]["time_ms"]
            full_fit = _predict_attention_piecewise(seq_len, fitted_segments)
            scaled_from_measured = full_measured / sp
            scaled_from_fit = full_fit / sp

            try:
                local_measurement = _profile_attention_point(
                    seq_len=seq_len,
                    n_heads=local_heads,
                    n_kv_heads=local_kv_heads,
                    head_dim=head_dim,
                    warmup=warmup,
                    iters=iters,
                    timing_groups=timing_groups,
                    timing_stat=timing_stat,
                    min_group_elapsed_ms=min_group_elapsed_ms,
                    max_iters_per_group=max_iters_per_group,
                    use_varlen=use_varlen,
                    device=device,
                    dtype=dtype,
                )
            except RuntimeError as ex:
                if "out of memory" in str(ex).lower():
                    torch.cuda.empty_cache()
                    continue
                raise

            actual = local_measurement["time_ms"]
            err_measured = abs(actual - scaled_from_measured) / max(actual, 1e-6) * 100.0
            err_fit = abs(actual - scaled_from_fit) / max(actual, 1e-6) * 100.0

            results.append({
                "seq_len": int(seq_len),
                "sp_size": int(sp),
                "local_heads": int(local_heads),
                "local_kv_heads": int(local_kv_heads),
                "actual_local_ms": float(actual),
                "actual_local_std_ms": float(local_measurement["std_ms"]),
                "full_measured_ms": float(full_measured),
                "full_fit_ms": float(full_fit),
                "scaled_from_measured_ms": float(scaled_from_measured),
                "scaled_from_fit_ms": float(scaled_from_fit),
                "error_scaled_measured_pct": float(err_measured),
                "error_scaled_fit_pct": float(err_fit),
            })
            print(f"  head-scaling sp={sp} seq={seq_len:>6}: actual={actual:.4f}ms, "
                  f"full/sp={scaled_from_measured:.4f}ms, fit/sp={scaled_from_fit:.4f}ms, "
                  f"err_fit={err_fit:.1f}%")

    return results


# ═══════════════════════════════════════════════════════════════════════
# PART 2: Communication Profiling with Linear Fitting
# ═══════════════════════════════════════════════════════════════════════

def profile_comm_linear(
    comm_type: str,  # "alltoall" or "p2p"
    group,
    group_size: int,
    msg_sizes_mb: List[float] = None,
    warmup: int = 5,
    iters: int = 50,
    dtype=torch.bfloat16,
) -> List[Dict]:
    """
    Profile communication and collect (msg_bytes, time_ms) pairs.
    More data points for precise linear fitting.
    """
    import torch.distributed as dist

    rank = dist.get_rank()
    local_rank = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{local_rank}")
    bpe = 2 if dtype == torch.bfloat16 else 4

    if msg_sizes_mb is None:
        # Dense sampling for better linear fit
        msg_sizes_mb = [0.5, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96,
                        128, 192, 256, 384, 512, 768, 1024]

    results = []

    if comm_type == "p2p":
        grp_rank = dist.get_rank(group)
        next_r = dist.get_global_rank(group, (grp_rank + 1) % group_size)
        prev_r = dist.get_global_rank(group, (grp_rank - 1) % group_size)

    for msg_mb in msg_sizes_mb:
        n_elem = int(msg_mb * 1024 * 1024 / bpe)
        if comm_type == "alltoall":
            n_elem = (n_elem // group_size) * group_size

        send_t = torch.randn(n_elem, dtype=dtype, device=device)
        recv_t = torch.empty_like(send_t)
        total_bytes = n_elem * bpe

        if comm_type == "alltoall":
            in_chunks = list(send_t.chunk(group_size))
            out_chunks = list(recv_t.chunk(group_size))

            for _ in range(warmup):
                dist.all_to_all(out_chunks, in_chunks, group=group)
            torch.cuda.synchronize()

            se = torch.cuda.Event(enable_timing=True)
            ee = torch.cuda.Event(enable_timing=True)
            se.record()
            for _ in range(iters):
                dist.all_to_all(out_chunks, in_chunks, group=group)
            ee.record()
            torch.cuda.synchronize()
            t_ms = se.elapsed_time(ee) / iters
            data_moved = total_bytes * (group_size - 1) / group_size

        elif comm_type == "p2p":
            def do_step():
                ops = [
                    dist.P2POp(dist.isend, send_t, next_r, group=group),
                    dist.P2POp(dist.irecv, recv_t, prev_r, group=group),
                ]
                reqs = dist.batch_isend_irecv(ops)
                for r in reqs:
                    r.wait()

            for _ in range(warmup):
                do_step()
            torch.cuda.synchronize()
            dist.barrier(group=group)

            se = torch.cuda.Event(enable_timing=True)
            ee = torch.cuda.Event(enable_timing=True)
            se.record()
            for _ in range(iters):
                do_step()
            ee.record()
            torch.cuda.synchronize()
            t_ms = se.elapsed_time(ee) / iters
            data_moved = total_bytes  # bidirectional
        else:
            raise ValueError(f"Unknown comm_type: {comm_type}")

        bw = data_moved / (t_ms / 1000) / 1e9 if t_ms > 0 else 0
        results.append({
            "msg_size_MB": msg_mb,
            "total_bytes": total_bytes,
            "data_moved_bytes": data_moved,
            "time_ms": t_ms,
            "bandwidth_GBs": bw,
        })

        if rank == 0:
            print(f"  {comm_type} gs={group_size}, msg={msg_mb:>7.1f}MB: "
                  f"{t_ms:.4f}ms, BW={bw:.2f} GB/s")

        del send_t, recv_t
        torch.cuda.empty_cache()

    return results


def fit_comm_linear(results: List[Dict]) -> Dict:
    """
    Fit communication time as:  time_ms = alpha * msg_size_MB + beta
    
    Where alpha = 1/bandwidth and beta = latency.
    
    We filter out very small messages (< 8MB) where latency dominates,
    and fit on the linear (bandwidth-limited) regime.
    """
    # Filter to bandwidth-limited regime (msg >= 8 MB)
    large = [(r["msg_size_MB"], r["time_ms"]) for r in results if r["msg_size_MB"] >= 8]
    all_pts = [(r["msg_size_MB"], r["time_ms"]) for r in results]
    
    if len(large) < 3:
        large = all_pts

    xs = np.array([p[0] for p in large])
    ys = np.array([p[1] for p in large])

    # Linear fit: time = alpha * msg_MB + beta
    from numpy.polynomial.polynomial import polyfit
    coeffs = polyfit(xs, ys, 1)  # [beta, alpha] in numpy convention
    beta, alpha = coeffs[0], coeffs[1]

    # R²
    y_pred = alpha * xs + beta
    ss_res = np.sum((ys - y_pred) ** 2)
    ss_tot = np.sum((ys - np.mean(ys)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Effective bandwidth (MB/s → GB/s)
    # alpha is ms per MB, so BW = 1/alpha (MB/ms) = 1000/alpha (MB/s) = 1/alpha (GB/s) 
    eff_bw_gbs = 1.0 / alpha if alpha > 0 else float('inf')

    # Also fit all points for comparison
    xs_all = np.array([p[0] for p in all_pts])
    ys_all = np.array([p[1] for p in all_pts])
    c_all = polyfit(xs_all, ys_all, 1)
    beta_all, alpha_all = c_all[0], c_all[1]

    return {
        "alpha_ms_per_MB": float(alpha),
        "beta_ms": float(beta),
        "r_squared": float(r2),
        "effective_bandwidth_GBs": float(eff_bw_gbs),
        "latency_ms": float(beta),
        "fit_range_MB": [float(xs.min()), float(xs.max())],
        "n_points": len(xs),
        # Full range fit
        "alpha_full": float(alpha_all),
        "beta_full": float(beta_all),
    }


# ═══════════════════════════════════════════════════════════════════════
# PART 3: CostModel Validation
# ═══════════════════════════════════════════════════════════════════════

def validate_cost_model_comm(
    group,
    group_size: int,
    comm_type: str,  # "alltoall" or "p2p"
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    num_layers: int,
    seq_lengths: List[int],
    costmodel,
    warmup: int = 5,
    iters: int = 20,
    dtype=torch.bfloat16,
) -> List[Dict]:
    """
    Measure actual communication time matching the CostModel's definitions.
    
    CostModel.alltoall_time: total comm for ALL layers, ALL directions (4*2*L ops).
      We measure a single all-to-all and multiply by 4*2*L to get total.
    
    CostModel.p2p_ring_time: total ring comm for ALL layers.
      We measure one full ring (cp_size-1 steps) and multiply by L.
    """
    import torch.distributed as dist
    from galvatron.models.varlen_llama_hf.adacpsp_solver import ParallelStrategy

    rank = dist.get_rank()
    local_rank = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{local_rank}")
    bpe = 2 if dtype == torch.bfloat16 else 4
    head_dim = hidden_size // num_heads

    results = []

    for seq_len in seq_lengths:
        if comm_type == "alltoall":
            # Measure single all-to-all (matching actual Ulysses tensor shape)
            heads_local = num_heads // group_size
            t_shape = (seq_len, 1, heads_local, head_dim)
            send = torch.randn(t_shape, dtype=dtype, device=device)
            recv = torch.empty_like(send)
            in_c = list(send.chunk(group_size, dim=0))
            out_c = list(recv.chunk(group_size, dim=0))

            for _ in range(warmup):
                dist.all_to_all(out_c, in_c, group=group)
            torch.cuda.synchronize()
            se = torch.cuda.Event(enable_timing=True)
            ee = torch.cuda.Event(enable_timing=True)
            se.record()
            for _ in range(iters):
                dist.all_to_all(out_c, in_c, group=group)
            ee.record()
            torch.cuda.synchronize()
            single_op_ms = se.elapsed_time(ee) / iters
            del send, recv

            # Scale to total: 4 (Q,K,V,O) × 2 (fwd+bwd) × L (layers) ops
            num_ops = 4 * 2 * num_layers
            measured_total_ms = single_op_ms * num_ops
            predicted_ms = costmodel.alltoall_time([seq_len], group_size)

            extra = {"single_op_ms": single_op_ms, "num_ops": num_ops}

        elif comm_type == "p2p":
            grp_rank = dist.get_rank(group)
            next_r = dist.get_global_rank(group, (grp_rank + 1) % group_size)
            prev_r = dist.get_global_rank(group, (grp_rank - 1) % group_size)

            local_seq = seq_len // group_size
            kv_shape = (local_seq, 1, num_kv_heads, head_dim)
            send_k = torch.randn(kv_shape, dtype=dtype, device=device)
            send_v = torch.randn(kv_shape, dtype=dtype, device=device)
            recv_k = torch.empty_like(send_k)
            recv_v = torch.empty_like(send_v)
            num_steps = group_size - 1

            def ring_step():
                ops = [
                    dist.P2POp(dist.isend, send_k, next_r, group=group),
                    dist.P2POp(dist.isend, send_v, next_r, group=group),
                    dist.P2POp(dist.irecv, recv_k, prev_r, group=group),
                    dist.P2POp(dist.irecv, recv_v, prev_r, group=group),
                ]
                reqs = dist.batch_isend_irecv(ops)
                for r in reqs:
                    r.wait()

            for _ in range(warmup):
                ring_step()
            torch.cuda.synchronize()
            dist.barrier(group=group)

            se = torch.cuda.Event(enable_timing=True)
            ee = torch.cuda.Event(enable_timing=True)
            se.record()
            for _ in range(iters):
                for _ in range(num_steps):
                    ring_step()
            ee.record()
            torch.cuda.synchronize()
            ring_time_ms = se.elapsed_time(ee) / iters  # one full ring pass
            del send_k, send_v, recv_k, recv_v

            # Scale to total: L (layers) full ring passes
            measured_total_ms = ring_time_ms * num_layers
            predicted_ms = costmodel.p2p_ring_time([seq_len], group_size)

            extra = {"ring_time_ms": ring_time_ms, "num_layers": num_layers}
        else:
            raise ValueError(f"Unknown comm_type: {comm_type}")

        torch.cuda.empty_cache()

        error_pct = abs(predicted_ms - measured_total_ms) / measured_total_ms * 100 \
            if measured_total_ms > 0 else 0
        result = {
            "seq_len": seq_len,
            "comm_type": comm_type,
            "group_size": group_size,
            "measured_ms": measured_total_ms,
            "predicted_ms": predicted_ms,
            "error_pct": error_pct,
        }
        result.update(extra)
        results.append(result)

        if rank == 0:
            print(f"  {comm_type} gs={group_size} seq={seq_len:>6}: "
                  f"measured_total={measured_total_ms:.4f}ms, predicted={predicted_ms:.4f}ms, "
                  f"error={error_pct:.1f}%")

    return results


def validate_cost_model_compute(
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    num_layers: int,
    costmodel,
    seq_lengths: List[int],
    warmup: int = 5,
    iters: int = 20,
    timing_groups: int = 3,
    timing_stat: str = "median",
    min_group_elapsed_ms: float = 1.0,
    max_iters_per_group: int = 512,
    use_varlen: bool = True,
    device: str = "cuda",
    dtype=torch.bfloat16,
) -> List[Dict]:
    """
    Measure actual attention compute time per layer and compare with CostModel.
    """
    from galvatron.models.varlen_llama_hf.adacpsp_solver import ParallelStrategy

    results = []
    strategy = ParallelStrategy("ulysses", 1)  # sp=1 for pure compute

    for seq_len in seq_lengths:
        try:
            measurement = _profile_attention_point(
                seq_len=seq_len,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                warmup=warmup,
                iters=iters,
                timing_groups=timing_groups,
                timing_stat=timing_stat,
                min_group_elapsed_ms=min_group_elapsed_ms,
                max_iters_per_group=max_iters_per_group,
                use_varlen=use_varlen,
                device=device,
                dtype=dtype,
            )
            measured_per_layer = measurement["time_ms"]

            predicted_per_layer = costmodel.compute_time_single(seq_len, strategy)
            error_pct = abs(predicted_per_layer - measured_per_layer) / measured_per_layer * 100 \
                if measured_per_layer > 0 else 0

            results.append({
                "seq_len": seq_len,
                "measured_per_layer_ms": measured_per_layer,
                "predicted_per_layer_ms": predicted_per_layer,
                "error_pct": error_pct,
                "measurement_std_ms": measurement["std_ms"],
            })
            print(f"  seq={seq_len:>6}: measured={measured_per_layer:.4f}ms/layer, "
                  f"predicted={predicted_per_layer:.4f}ms/layer, error={error_pct:.1f}% "
                  f"(std={measurement['std_ms']:.4f})")

        except RuntimeError as ex:
            if "out of memory" in str(ex).lower():
                torch.cuda.empty_cache()
                print(f"  seq={seq_len}: OOM")
                break
            raise

    return results


# ═══════════════════════════════════════════════════════════════════════
# PART 4: Memory Model Validation
# ═══════════════════════════════════════════════════════════════════════

def validate_memory_model(
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    hidden_size: int,
    num_layers: int,
    costmodel,
    seq_lengths: List[int],
    device: str = "cuda",
    dtype=torch.bfloat16,
) -> List[Dict]:
    """
    Validate memory model by measuring per-layer activation and scaling to total.
    
    The CostModel's activation_size returns TOTAL activation across all layers.
    We measure per-layer activation (attention + FFN) and multiply by num_layers.
    """
    bpe = 2 if dtype == torch.bfloat16 else 4
    intermediate_size = int(hidden_size * 2.6875)  # LLaMA: 11008 for h=4096
    
    results = []
    
    print(f"  Model config: hidden={hidden_size}, heads={n_heads}, kv_heads={n_kv_heads}, "
          f"head_dim={head_dim}, layers={num_layers}")
    print(f"  CostModel: act_per_token = {costmodel.act_per_token:.4f} MB/token (all layers)")
    print(f"  CostModel: act_per_token_per_layer = {costmodel.act_per_token / num_layers:.4f} MB/token")
    
    # Theoretical per-layer per-token activation (bytes)
    per_layer_per_token_bytes = (
        hidden_size * bpe +           # input hidden state
        3 * hidden_size * bpe +       # Q, K, V projections
        hidden_size * bpe +           # attention output 
        hidden_size * bpe +           # O projection output
        2 * intermediate_size * bpe + # gate + up projection
        intermediate_size * bpe +     # activated (gate * up)
        2 * hidden_size * bpe         # two layernorm inputs
    )
    per_layer_per_token_mb = per_layer_per_token_bytes / 1024 / 1024
    theoretical_total_per_token = per_layer_per_token_mb * num_layers
    print(f"  Theoretical: {per_layer_per_token_mb:.4f} MB/token/layer, "
          f"{theoretical_total_per_token:.4f} MB/token total")
    
    for seq_len in seq_lengths:
        try:
            # ─── Measure attention peak activation ───
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            before_attn = torch.cuda.memory_allocated() / 1024 / 1024
            
            q = torch.randn(seq_len, n_heads, head_dim, dtype=dtype, device=device)
            k = torch.randn(seq_len, n_kv_heads, head_dim, dtype=dtype, device=device)
            v = torch.randn(seq_len, n_kv_heads, head_dim, dtype=dtype, device=device)
            
            if flash_attn_varlen_func is not None:
                cu = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
                out = flash_attn_varlen_func(q, k, v, cu, cu, seq_len, seq_len, causal=True)
            elif flash_attn_func is not None:
                q2 = q.unsqueeze(0)
                k2 = k.unsqueeze(0)
                v2 = v.unsqueeze(0)
                out = flash_attn_func(q2, k2, v2, causal=True)
            
            torch.cuda.synchronize()
            peak_attn = torch.cuda.max_memory_allocated() / 1024 / 1024
            attn_peak_mb = peak_attn - before_attn
            del q, k, v, out
            torch.cuda.empty_cache()
            
            # ─── Measure FFN peak activation ───
            torch.cuda.reset_peak_memory_stats()
            before_ffn = torch.cuda.memory_allocated() / 1024 / 1024
            
            x = torch.randn(seq_len, hidden_size, dtype=dtype, device=device)
            gate_w = torch.randn(intermediate_size, hidden_size, dtype=dtype, device=device)
            up_w = torch.randn(intermediate_size, hidden_size, dtype=dtype, device=device)
            down_w = torch.randn(hidden_size, intermediate_size, dtype=dtype, device=device)
            
            gate_out = torch.nn.functional.linear(x, gate_w)
            up_out = torch.nn.functional.linear(x, up_w)
            activated = torch.nn.functional.silu(gate_out) * up_out
            ffn_out = torch.nn.functional.linear(activated, down_w)
            
            torch.cuda.synchronize()
            peak_ffn = torch.cuda.max_memory_allocated() / 1024 / 1024
            ffn_peak_mb = peak_ffn - before_ffn
            del x, gate_w, up_w, down_w, gate_out, up_out, activated, ffn_out
            torch.cuda.empty_cache()
            
            # ─── Estimate total ───
            # attn_peak includes QKV + output + flash internal buffers
            # ffn_peak includes weight tensors — subtract them
            weight_size_mb = (intermediate_size * hidden_size * 3) * bpe / 1024 / 1024
            ffn_activation_mb = max(0, ffn_peak_mb - weight_size_mb)
            
            per_layer_measured_mb = attn_peak_mb + ffn_activation_mb
            total_measured_mb = per_layer_measured_mb * num_layers
            
            predicted_total_mb = costmodel.activation_size(seq_len, parallel_size=1)
            
            error_pct = abs(predicted_total_mb - total_measured_mb) / total_measured_mb * 100 \
                if total_measured_mb > 0 else 0
            
            per_token_measured = per_layer_measured_mb / seq_len * num_layers
            per_token_predicted = costmodel.act_per_token
            
            results.append({
                "seq_len": seq_len,
                "attn_peak_MB": attn_peak_mb,
                "ffn_activation_MB": ffn_activation_mb,
                "per_layer_MB": per_layer_measured_mb,
                "total_measured_MB": total_measured_mb,
                "total_predicted_MB": predicted_total_mb,
                "per_token_measured": per_token_measured,
                "per_token_predicted": per_token_predicted,
                "error_pct": error_pct,
            })
            print(f"  seq={seq_len:>6}: per_layer={per_layer_measured_mb:.1f}MB "
                  f"(attn={attn_peak_mb:.1f} + ffn={ffn_activation_mb:.1f}), "
                  f"total_est={total_measured_mb:.1f}MB vs predicted={predicted_total_mb:.1f}MB, "
                  f"error={error_pct:.1f}%")

        except RuntimeError as ex:
            if "out of memory" in str(ex).lower():
                torch.cuda.empty_cache()
                print(f"  seq={seq_len}: OOM")
                break
            raise

    return results


# ═══════════════════════════════════════════════════════════════════════
# PART 5: Plotting utilities
# ═══════════════════════════════════════════════════════════════════════

def plot_all(
    attn_data=None, attn_segments=None, breakpoints=None,
    comm_results=None, comm_fits=None,
    compute_val=None, comm_val=None, memory_val=None,
    save_dir="./configs",
):
    """Generate comprehensive plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Plot] matplotlib not available")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ─── Plot 1: Attention profiling with auto breakpoints ───
    if attn_data and attn_segments:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1a: raw data + fitted curves
        ax = axes[0, 0]
        xs, ts = _extract_attention_xy(attn_data)
        ax.scatter(xs, ts, s=3, alpha=0.5, label="measured", color="gray")
        colors = ["blue", "green", "orange", "red", "purple", "cyan"]
        for i, seg in enumerate(attn_segments):
            lo, hi = seg["range"]
            x_fit = np.linspace(lo, hi, 200)
            y_fit = seg["a"] * x_fit**2 + seg["b"] * x_fit + seg["c"]
            ax.plot(x_fit, y_fit, color=colors[i % len(colors)], linewidth=2,
                    label=f'[{lo},{hi}] R²={seg["r_squared"]:.5f}')
        if breakpoints:
            for bp in breakpoints:
                ax.axvline(x=bp, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Time (ms)")
        ax.set_title("Attention: Auto-Segmented Piecewise Quadratic Fit")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # 1b: time/x² to show kernel switches
        ax = axes[0, 1]
        r = ts / (xs ** 2)
        ax.plot(xs, r * 1e9, color="blue", linewidth=0.5, alpha=0.7)
        ax.scatter(xs, r * 1e9, s=2, color="blue")
        if breakpoints:
            for bp in breakpoints:
                ax.axvline(x=bp, color="red", linestyle="--", alpha=0.7, label=f"bp={bp}")
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("time / x² (×10⁹)")
        ax.set_title("Normalized Metric: time/x² (kernel switch indicator)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # 1c: R² per segment
        ax = axes[1, 0]
        names = [f'[{s["range"][0]},{s["range"][1]}]' for s in attn_segments]
        r2s = [s["r_squared"] for s in attn_segments]
        bars = ax.bar(range(len(names)), r2s, color=[colors[i % len(colors)] for i in range(len(names))])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=7, rotation=15)
        ax.axhline(y=0.999, color="r", linestyle="--", label="R²=0.999")
        ax.set_ylim(min(0.99, min(r2s) - 0.005) if r2s else 0.9, 1.002)
        ax.set_title("Fitting Quality per Segment")
        ax.legend()
        ax.grid(True, alpha=0.3)
        for bar, r2 in zip(bars, r2s):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                    f"{r2:.5f}", ha="center", fontsize=7)

        # 1d: residuals
        ax = axes[1, 1]
        for i, seg in enumerate(attn_segments):
            lo, hi = seg["range"]
            mask = _segment_mask(xs, lo, hi, include_hi=(i == len(attn_segments) - 1))
            seg_x = xs[mask]
            seg_t = ts[mask]
            y_pred = seg["a"] * seg_x**2 + seg["b"] * seg_x + seg["c"]
            residuals = seg_t - y_pred
            ax.scatter(seg_x, residuals, s=3, color=colors[i % len(colors)],
                       label=f'[{lo},{hi}]', alpha=0.6)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Residual (ms)")
        ax.set_title("Fitting Residuals")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(save_dir, f"attention_autofit_{timestamp}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Attention plot saved: {path}")
        plt.close()

    # ─── Plot 2: Communication linear fit ───
    if comm_results and comm_fits:
        n_plots = len(comm_results)
        fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]
        
        for idx, (key, data) in enumerate(comm_results.items()):
            ax = axes[idx]
            xs = np.array([d["msg_size_MB"] for d in data])
            ys = np.array([d["time_ms"] for d in data])
            ax.scatter(xs, ys, s=15, alpha=0.7, label="measured")
            
            if key in comm_fits:
                fit = comm_fits[key]
                x_fit = np.linspace(0, xs.max(), 200)
                y_fit = fit["alpha_ms_per_MB"] * x_fit + fit["beta_ms"]
                ax.plot(x_fit, y_fit, "r--", linewidth=2,
                        label=f'fit: t={fit["alpha_ms_per_MB"]:.4e}*x + {fit["beta_ms"]:.4f}\n'
                              f'R²={fit["r_squared"]:.5f}, BW={fit["effective_bandwidth_GBs"]:.1f}GB/s')
            
            ax.set_xlabel("Message Size (MB)")
            ax.set_ylabel("Time (ms)")
            ax.set_title(f"{key}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(save_dir, f"comm_linear_fit_{timestamp}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Comm plot saved: {path}")
        plt.close()

    # ─── Plot 3: CostModel validation ───
    if compute_val or comm_val:
        vals = []
        if compute_val:
            vals.append(("Compute (attention)", compute_val, "seq_len",
                         "measured_per_layer_ms", "predicted_per_layer_ms"))
        if comm_val:
            for item in comm_val:
                label = f'{item["comm_type"]} gs={item.get("group_size", "?")}'
                vals.append((label, [item], "seq_len", "measured_ms", "predicted_ms"))
        
        # Flatten comm_val if it's a list of lists
        if comm_val and isinstance(comm_val[0], dict):
            # Group by (comm_type, group_size)
            from collections import defaultdict
            groups = defaultdict(list)
            for item in comm_val:
                key = f'{item["comm_type"]} gs={item["group_size"]}'
                groups[key].append(item)
            vals = []
            if compute_val:
                vals.append(("Compute (attention)", compute_val, "seq_len",
                             "measured_per_layer_ms", "predicted_per_layer_ms"))
            for key, items in groups.items():
                vals.append((key, items, "seq_len", "measured_ms", "predicted_ms"))

        n = len(vals)
        if n > 0:
            cols = min(n, 3)
            rows = (n + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
            if n == 1:
                axes = np.array([axes])
            axes = np.array(axes).flatten()

            for i, (title, data, x_key, meas_key, pred_key) in enumerate(vals):
                ax = axes[i]
                xs = [d[x_key] for d in data]
                ms = [d[meas_key] for d in data]
                ps = [d[pred_key] for d in data]
                ax.plot(xs, ms, "o-", label="measured", markersize=4)
                ax.plot(xs, ps, "s--", label="predicted", markersize=4)
                ax.set_xlabel("Sequence Length")
                ax.set_ylabel("Time (ms)")
                ax.set_title(title)
                ax.legend()
                ax.grid(True, alpha=0.3)

            for i in range(n, len(axes)):
                axes[i].set_visible(False)

            plt.tight_layout()
            path = os.path.join(save_dir, f"costmodel_validation_{timestamp}.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"[Plot] Validation plot saved: {path}")
            plt.close()


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="AdaCPSP Profiling & Validation Suite")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["attention", "comm", "validate_cost_model",
                                 "validate_memory", "all"],
                        help="Which profiling/validation to run")
    # Model config
    parser.add_argument("--n_heads", type=int, default=32)
    parser.add_argument("--n_kv_heads", type=int, default=32)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=4096)
    parser.add_argument("--num_layers", type=int, default=32)
    parser.add_argument("--param_size_B", type=float, default=7.0)
    # Profiling
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--attn_timing_groups", type=int, default=3,
                        help="Split attention timing into multiple groups and aggregate them robustly")
    parser.add_argument("--attn_timing_stat", type=str, default="median",
                        choices=["median", "mean"],
                        help="How to aggregate attention timing groups for each seq_len")
    parser.add_argument("--attn_min_group_elapsed_ms", type=float, default=1.0,
                        help="Adaptive timing target: each attention timing group should last at least this long")
    parser.add_argument("--attn_max_iters_per_group", type=int, default=512,
                        help="Upper bound on adaptive iterations per attention timing group")
    parser.add_argument("--attn_step", type=int, default=64,
                        help="Dense attention profiling step size")
    parser.add_argument("--attn_max", type=int, default=32768,
                        help="Max seq_len for attention profiling")
    parser.add_argument("--use_varlen", action="store_true", default=True)
    parser.add_argument("--skip_head_scaling_check", action="store_true",
                        help="Skip the reduced-head scaling validation after fitting the baseline")
    parser.add_argument("--head_scaling_seqs", type=int, nargs="+", default=[2048, 8192, 16384],
                        help="Seq lengths used to validate reduced-head scaling")
    parser.add_argument("--head_scaling_sp_sizes", type=int, nargs="+", default=[2, 4, 8],
                        help="Ulysses SP sizes used to validate reduced-head scaling")
    # Breakpoint detection
    parser.add_argument("--bp_window", type=int, default=5,
                        help="Sliding window for breakpoint detection")
    parser.add_argument("--bp_threshold", type=float, default=3.0,
                        help="Threshold factor for breakpoint detection")
    parser.add_argument("--bp_min_segment", type=int, default=8,
                        help="Min data points per segment")
    parser.add_argument("--max_segments", type=int, default=5,
                        help="Max number of segments after consolidation")
    parser.add_argument("--min_r2", type=float, default=0.995,
                        help="Min R² threshold for consolidation")
    # Output
    parser.add_argument("--save_dir", type=str, default="./configs")
    parser.add_argument("--model_name", type=str, default="llama-7b")
    # Existing profile files (for validation modes)
    parser.add_argument("--attn_json", type=str, default=None,
                        help="Existing attention profile JSON (skip re-profiling)")
    parser.add_argument("--alltoall_json", type=str, default=None)
    parser.add_argument("--p2p_json", type=str, default=None)
    parser.add_argument("--comm_profile_json", type=str, default=None,
                        help="Unified topology-aware communication profile JSON")
    parser.add_argument("--profile_json", type=str, default=None,
                        help="Unified profile JSON from previous run (loads attn segments + comm linear fits)")
    # Dataset
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name for varlen test (wikipedia/common_crawl/github)")
    parser.add_argument("--dataset_path", type=str,
                        default="/home/pkuhetu/lqs/flexsp/Hetu-Galvatron/galvatron/datasets",
                        help="Path to dataset directory")
    # Distributed
    parser.add_argument("--local-rank", "--local_rank", type=int, default=-1)
    args, _ = parser.parse_known_args()

    os.makedirs(args.save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    is_distributed = args.mode in ["comm", "validate_cost_model", "validate_memory", "all"]
    rank = 0
    world_size = 1

    if is_distributed:
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = rank % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)
    else:
        torch.cuda.set_device(0)

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    if rank == 0:
        print("=" * 80)
        print(" AdaCPSP Comprehensive Profiling & Validation Suite")
        print("=" * 80)
        print(f" Mode: {args.mode}")
        print(f" Model: {args.model_name} (h={args.hidden_size}, heads={args.n_heads}, "
              f"kv_heads={args.n_kv_heads}, layers={args.num_layers})")
        if is_distributed:
            print(f" World size: {world_size}")
        print("=" * 80)

    all_output = {"timestamp": timestamp, "model_name": args.model_name}
    attn_data = None
    attn_segments = None
    breakpoints = None
    comm_results_all = {}
    comm_fits_all = {}
    compute_val = None
    comm_val_all = []
    memory_val = None
    attn_segment_diagnostics = None
    head_scaling_val = None

    # ── Auto-detect best profile JSONs from configs dir if not specified ──
    if args.mode in ["validate_cost_model", "all"] and (not args.profile_json or not args.comm_profile_json):
        import glob as _glob
        profile_files = sorted(_glob.glob(os.path.join(args.save_dir, "profile_validate_*.json")))
        comm_profile_files = sorted(_glob.glob(os.path.join(args.save_dir, "comm_profile_*.json")))
        best_attn_json = None
        best_comm_json = None
        best_validation_json = None
        best_comm_profile_json = comm_profile_files[-1] if comm_profile_files else None
        for pf in reversed(profile_files):  # newest first
            try:
                with open(pf) as _f:
                    _d = json.load(_f)
                if best_attn_json is None and "attention" in _d and "segments" in _d.get("attention", {}):
                    best_attn_json = pf
                if best_validation_json is None and "comm_validation" in _d:
                    best_validation_json = pf
                if best_comm_json is None and "communication" in _d and "linear_fits" in _d.get("communication", {}):
                    best_comm_json = pf
            except Exception:
                pass
        if rank == 0:
            if best_attn_json:
                print(f"  Auto-detected attention profile: {best_attn_json}")
            if best_comm_profile_json:
                print(f"  Auto-detected topology-aware comm profile: {best_comm_profile_json}")
            if best_comm_json:
                print(f"  Auto-detected comm profile: {best_comm_json}")
        # Store for later use (will be loaded in Part 3)
        args._auto_attn_json = best_attn_json
        args._auto_comm_json = best_comm_json
        args._auto_validation_json = best_validation_json
        args._auto_comm_profile_json = best_comm_profile_json

    # ═══ PART 1: Attention Profiling ═══
    if args.mode in ["attention", "all"]:
        if rank == 0:
            print(f"\n{'='*80}")
            print(" PART 1: Attention Profiling with Auto Breakpoint Detection")
            print(f"{'='*80}")

            attn_data = profile_attention_dense(
                args.n_heads, args.n_kv_heads, args.head_dim,
                seq_range=(64, args.attn_max), step=args.attn_step,
                warmup=args.warmup, iters=args.iters,
                timing_groups=args.attn_timing_groups,
                timing_stat=args.attn_timing_stat,
                min_group_elapsed_ms=args.attn_min_group_elapsed_ms,
                max_iters_per_group=args.attn_max_iters_per_group,
                use_varlen=args.use_varlen,
            )

            if attn_data:
                print(f"\n--- Auto Breakpoint Detection ---")
                raw_breakpoints = detect_breakpoints(
                    attn_data,
                    window=args.bp_window,
                    threshold_factor=args.bp_threshold,
                    min_segment_points=args.bp_min_segment,
                )

                print(f"\n--- Breakpoint Consolidation (target ≤{args.max_segments} segments) ---")
                breakpoints = consolidate_breakpoints(
                    attn_data, raw_breakpoints,
                    max_segments=args.max_segments,
                    min_r2=args.min_r2,
                )

                print(f"\n--- Piecewise Quadratic Fitting (consolidated) ---")
                attn_segments = fit_segments_quadratic(attn_data, breakpoints)
                attn_segment_diagnostics = summarize_attention_segments(attn_data, attn_segments)
                score = attn_segment_diagnostics["selection_score"]
                print(f"\n--- Segment Diagnostics ---")
                print(f"  min_R²={score['min_r_squared']:.6f}, "
                      f"max_boundary_jump={score['max_boundary_rel_jump_pct']:.2f}%, "
                      f"mean_boundary_jump={score['mean_boundary_rel_jump_pct']:.2f}%")

                if not args.skip_head_scaling_check:
                    head_scaling_seqs = [s for s in args.head_scaling_seqs if s <= args.attn_max]
                    if head_scaling_seqs:
                        print(f"\n--- Head Scaling Validation ---")
                        head_scaling_val = validate_head_scaling(
                            full_measurements=attn_data,
                            fitted_segments=attn_segments,
                            n_heads=args.n_heads,
                            n_kv_heads=args.n_kv_heads,
                            head_dim=args.head_dim,
                            seq_lengths=head_scaling_seqs,
                            sp_sizes=args.head_scaling_sp_sizes,
                            warmup=args.warmup,
                            iters=args.iters,
                            timing_groups=args.attn_timing_groups,
                            timing_stat=args.attn_timing_stat,
                            min_group_elapsed_ms=args.attn_min_group_elapsed_ms,
                            max_iters_per_group=args.attn_max_iters_per_group,
                            use_varlen=args.use_varlen,
                        )

                all_output["attention"] = {
                    "raw_breakpoints": raw_breakpoints,
                    "consolidated_breakpoints": breakpoints,
                    "segments": attn_segments,
                    "raw_data": [(int(x), float(t)) for x, t in zip(*_extract_attention_xy(attn_data))],
                    "raw_measurements": _serialize_attention_measurements(attn_data),
                    "segment_diagnostics": attn_segment_diagnostics,
                    "head_scaling_validation": head_scaling_val,
                    "config": {
                        "n_heads": args.n_heads,
                        "n_kv_heads": args.n_kv_heads,
                        "head_dim": args.head_dim,
                        "hidden_size": args.hidden_size,
                        "num_layers": args.num_layers,
                        "step": args.attn_step,
                        "use_varlen": args.use_varlen,
                        "timing_groups": args.attn_timing_groups,
                        "timing_stat": args.attn_timing_stat,
                        "min_group_elapsed_ms": args.attn_min_group_elapsed_ms,
                        "max_iters_per_group": args.attn_max_iters_per_group,
                        "seed": args.seed,
                        "head_scaling_seqs": args.head_scaling_seqs,
                        "head_scaling_sp_sizes": args.head_scaling_sp_sizes,
                    },
                }

        if is_distributed:
            import torch.distributed as dist
            dist.barrier()

    # ═══ PART 2: Communication Profiling ═══
    if args.mode in ["comm", "all"] and is_distributed:
        import torch.distributed as dist

        if rank == 0:
            print(f"\n{'='*80}")
            print(" PART 2: Communication Profiling with Linear Fitting")
            print(f"{'='*80}")
            print("  [Legacy] This simplified comm path is kept for compatibility.")
            print("  [Legacy] Prefer profile_comm.py for topology-aware cross-node profiling.")

        for gs in [2, 4, 8]:
            if gs > world_size:
                continue

            num_groups = world_size // gs
            my_group = None
            for g in range(num_groups):
                ranks = list(range(g * gs, (g + 1) * gs))
                group = dist.new_group(ranks=ranks)
                if rank in ranks:
                    my_group = group

            for comm_type in ["alltoall", "p2p"]:
                key = f"{comm_type}_gs{gs}"
                if rank == 0:
                    print(f"\n--- {comm_type} group_size={gs} ---")

                results = profile_comm_linear(
                    comm_type, my_group, gs,
                    warmup=args.warmup, iters=max(args.iters, 50),
                )
                comm_results_all[key] = results

                if rank == 0:
                    fit = fit_comm_linear(results)
                    comm_fits_all[key] = fit
                    print(f"  → Linear fit: time = {fit['alpha_ms_per_MB']:.6e} * msg_MB + {fit['beta_ms']:.4f}")
                    print(f"    R² = {fit['r_squared']:.6f}, Effective BW = {fit['effective_bandwidth_GBs']:.2f} GB/s")
                    print(f"    Latency = {fit['beta_ms']:.4f} ms")

            dist.barrier()

        if rank == 0:
            all_output["communication"] = {
                "results": {k: v for k, v in comm_results_all.items()},
                "linear_fits": comm_fits_all,
            }

    # ═══ PART 3: CostModel Validation ═══
    if args.mode in ["validate_cost_model", "all"]:
        if rank == 0:
            print(f"\n{'='*80}")
            print(" PART 3: CostModel Validation (Predicted vs Measured)")
            print(f"{'='*80}")

        # Build cost model from existing or fresh profile data
        sys.path.insert(0, os.path.dirname(__file__))
        from adacpsp_solver import AdaCPSPCostModel, ParallelStrategy
        def _json_has(path: Optional[str], key: str) -> bool:
            if not path or not os.path.exists(path):
                return False
            try:
                with open(path) as f:
                    return key in json.load(f)
            except Exception:
                return False

        attention_source_json = None
        if args.attn_json and os.path.exists(args.attn_json):
            attention_source_json = args.attn_json
        elif args.profile_json and _json_has(args.profile_json, "attention"):
            attention_source_json = args.profile_json
        elif hasattr(args, "_auto_attn_json") and args._auto_attn_json:
            attention_source_json = args._auto_attn_json

        comm_profile_json = None
        if args.comm_profile_json and os.path.exists(args.comm_profile_json):
            comm_profile_json = args.comm_profile_json
        elif hasattr(args, "_auto_comm_profile_json") and args._auto_comm_profile_json:
            comm_profile_json = args._auto_comm_profile_json

        validation_json = None
        if args.profile_json and _json_has(args.profile_json, "comm_validation"):
            validation_json = args.profile_json
        elif hasattr(args, "_auto_validation_json") and args._auto_validation_json:
            validation_json = args._auto_validation_json

        if attention_source_json and comm_profile_json:
            costmodel = AdaCPSPCostModel.from_attention_and_comm_profiles(
                attention_json=attention_source_json,
                comm_profile_json=comm_profile_json,
                cluster_size=world_size,
                param_size_B=args.param_size_B,
                gpus_per_node=torch.cuda.device_count(),
                validation_json=validation_json,
            )
            if rank == 0:
                print(
                    f"  Using topology-aware comm profile: "
                    f"attn={attention_source_json}, comm={comm_profile_json}, validation={validation_json}"
                )
        else:
            # ── Load profiling data (from this run, saved JSON, or defaults) ──
            piecewise = None
            alltoall_linear = {}
            p2p_linear = {}
            a2a_bw = None
            p2p_bw = None

            # Helper to load from a profile JSON
            def _load_from_json(path, label=""):
                nonlocal piecewise, alltoall_linear, p2p_linear
                with open(path) as f:
                    prev_data = json.load(f)
                if rank == 0:
                    print(f"  Loading {label} from: {path}")
                if piecewise is None and "attention" in prev_data and "segments" in prev_data["attention"]:
                    piecewise = prev_data["attention"]["segments"]
                if not alltoall_linear and "communication" in prev_data and "linear_fits" in prev_data["communication"]:
                    for key, fit in prev_data["communication"]["linear_fits"].items():
                        gs = int(key.split("gs")[1])
                        entry = {"alpha": fit["alpha_ms_per_MB"], "beta": fit["beta_ms"]}
                        if key.startswith("alltoall"):
                            alltoall_linear[gs] = entry
                        elif key.startswith("p2p"):
                            p2p_linear[gs] = entry

            # Source 1: explicitly specified unified profile JSON
            if args.profile_json and os.path.exists(args.profile_json):
                _load_from_json(args.profile_json, "unified profile")

            # Source 1b: auto-detected profile JSONs (separate attn + comm)
            if piecewise is None and hasattr(args, '_auto_attn_json') and args._auto_attn_json:
                _load_from_json(args._auto_attn_json, "auto-detected attention")
            if not alltoall_linear and hasattr(args, '_auto_comm_json') and args._auto_comm_json:
                _load_from_json(args._auto_comm_json, "auto-detected comm")

            # Source 2: separate JSON files
            if piecewise is None and args.attn_json and os.path.exists(args.attn_json):
                with open(args.attn_json) as f:
                    attn_prof = json.load(f)
                piecewise = []
                if "attention" in attn_prof and "segments" in attn_prof["attention"]:
                    piecewise = attn_prof["attention"]["segments"]
                else:
                    for seg_name, coeff in attn_prof.get("coefficients", {}).items():
                        if coeff:
                            piecewise.append({"range": coeff["seq_range"], "a": coeff["a"],
                                              "b": coeff["b"], "c": coeff["c"]})

            if not alltoall_linear and args.alltoall_json and os.path.exists(args.alltoall_json):
                with open(args.alltoall_json) as f:
                    a2a_bw = {int(k): v for k, v in json.load(f)["bandwidth_dict_GBs"].items()}

            if not p2p_linear and args.p2p_json and os.path.exists(args.p2p_json):
                with open(args.p2p_json) as f:
                    p2p_bw = {int(k): v for k, v in json.load(f)["bandwidth_dict_GBs"].items()}

            # Source 3: from this run's Part 1/2
            if piecewise is None and attn_segments:
                piecewise = attn_segments

            if not alltoall_linear and comm_fits_all:
                for key, fit in comm_fits_all.items():
                    gs = int(key.split("gs")[1])
                    entry = {"alpha": fit["alpha_ms_per_MB"], "beta": fit["beta_ms"]}
                    if key.startswith("alltoall"):
                        alltoall_linear[gs] = entry
                    elif key.startswith("p2p"):
                        p2p_linear[gs] = entry

            if rank == 0:
                if alltoall_linear:
                    print(f"  Using linear fit for alltoall: {sorted(alltoall_linear.keys())}")
                else:
                    print(f"  Using bandwidth model for alltoall (no linear fit)")
                if p2p_linear:
                    print(f"  Using linear fit for p2p: {sorted(p2p_linear.keys())}")
                else:
                    print(f"  Using bandwidth model for p2p (no linear fit)")

            costmodel = AdaCPSPCostModel(
                cluster_size=world_size,
                hidden_size=args.hidden_size,
                layer_num=args.num_layers,
                param_size_B=args.param_size_B,
                piecewise_compute_coeffs=piecewise,
                alltoall_bandwidth_dict_gbs=a2a_bw,
                p2p_bandwidth_dict_gbs=p2p_bw,
                alltoall_linear_fit=alltoall_linear if alltoall_linear else None,
                p2p_linear_fit=p2p_linear if p2p_linear else None,
            )

        # 3a: Compute validation
        if rank == 0:
            print(f"\n--- 3a: Compute Time Validation ---")
            test_seqs = [512, 1024, 2048, 4096, 8192, 16384, 32768]
            test_seqs = [s for s in test_seqs if s <= args.attn_max]
            compute_val = validate_cost_model_compute(
                args.n_heads, args.n_kv_heads, args.head_dim,
                args.num_layers, costmodel, test_seqs,
                warmup=args.warmup, iters=args.iters,
                timing_groups=args.attn_timing_groups,
                timing_stat=args.attn_timing_stat,
                min_group_elapsed_ms=args.attn_min_group_elapsed_ms,
                max_iters_per_group=args.attn_max_iters_per_group,
                use_varlen=args.use_varlen,
            )
            all_output["compute_validation"] = compute_val

        # 3b: Communication validation
        if is_distributed:
            import torch.distributed as dist

            for gs in [2, 4, 8]:
                if gs > world_size:
                    continue
                num_groups = world_size // gs
                my_group = None
                for g in range(num_groups):
                    ranks_list = list(range(g * gs, (g + 1) * gs))
                    group = dist.new_group(ranks=ranks_list)
                    if rank in ranks_list:
                        my_group = group

                for comm_type in ["alltoall", "p2p"]:
                    if rank == 0:
                        print(f"\n--- 3b: {comm_type} gs={gs} Comm Validation ---")

                    test_seqs = [2048, 4096, 8192, 16384, 32768]
                    test_seqs = [s for s in test_seqs if s >= gs * 2]

                    val_results = validate_cost_model_comm(
                        my_group, gs, comm_type,
                        args.hidden_size, args.n_heads, args.n_kv_heads,
                        args.num_layers, test_seqs, costmodel,
                        warmup=args.warmup, iters=args.iters,
                    )
                    comm_val_all.extend(val_results)

                dist.barrier()

            if rank == 0:
                all_output["comm_validation"] = comm_val_all

    # ═══ PART 4: Memory Validation ═══
    if args.mode in ["validate_memory", "all"]:
        if rank == 0:
            print(f"\n{'='*80}")
            print(" PART 4: Memory Model Validation")
            print(f"{'='*80}")

            # Need costmodel for memory validation
            if 'costmodel' not in dir():
                sys.path.insert(0, os.path.dirname(__file__))
                from adacpsp_solver import AdaCPSPCostModel
                costmodel = AdaCPSPCostModel(
                    cluster_size=world_size,
                    hidden_size=args.hidden_size,
                    layer_num=args.num_layers,
                    param_size_B=args.param_size_B,
                )

            test_seqs = [1024, 2048, 4096, 8192, 16384]
            memory_val = validate_memory_model(
                args.n_heads, args.n_kv_heads, args.head_dim,
                args.hidden_size, args.num_layers,
                costmodel, test_seqs,
            )
            all_output["memory_validation"] = memory_val

    # ═══ Save all results ═══
    if rank == 0:
        save_path = os.path.join(args.save_dir, f"profile_validate_{args.model_name}_{timestamp}.json")
        with open(save_path, "w") as f:
            json.dump(all_output, f, indent=2, default=str)
        print(f"\n[Save] Full results: {save_path}")

        # Generate plots
        plot_all(
            attn_data=attn_data,
            attn_segments=attn_segments,
            breakpoints=breakpoints,
            comm_results=comm_results_all if comm_results_all else None,
            comm_fits=comm_fits_all if comm_fits_all else None,
            compute_val=compute_val,
            comm_val=comm_val_all if comm_val_all else None,
            memory_val=memory_val,
            save_dir=args.save_dir,
        )

        # ─── Summary ───
        print(f"\n{'='*80}")
        print(" SUMMARY")
        print(f"{'='*80}")

        if attn_segments:
            print(f"\n Attention (auto {len(breakpoints)} breakpoints → {len(attn_segments)} segments):")
            for seg in attn_segments:
                print(f"   [{seg['range'][0]:>6}, {seg['range'][1]:>6}]: "
                      f"a={seg['a']:.6e}, b={seg['b']:.6e}, c={seg['c']:.4f}, "
                      f"R²={seg['r_squared']:.6f}")
            if attn_segment_diagnostics:
                score = attn_segment_diagnostics["selection_score"]
                print(f"   Diagnostics: min_R²={score['min_r_squared']:.6f}, "
                      f"max_boundary_jump={score['max_boundary_rel_jump_pct']:.2f}%, "
                      f"mean_boundary_jump={score['mean_boundary_rel_jump_pct']:.2f}%")
            if head_scaling_val:
                errs = [v["error_scaled_fit_pct"] for v in head_scaling_val]
                print(f"   Head scaling (/sp) error: mean={np.mean(errs):.1f}%, max={np.max(errs):.1f}%")

        if comm_fits_all:
            print(f"\n Communication (linear fit: time = α*msg_MB + β):")
            for key, fit in comm_fits_all.items():
                print(f"   {key:>15}: α={fit['alpha_ms_per_MB']:.6e}, β={fit['beta_ms']:.4f}ms, "
                      f"R²={fit['r_squared']:.5f}, BW={fit['effective_bandwidth_GBs']:.1f}GB/s")

        if compute_val:
            errs = [v["error_pct"] for v in compute_val]
            print(f"\n Compute model error: mean={np.mean(errs):.1f}%, max={np.max(errs):.1f}%")

        if comm_val_all:
            errs = [v["error_pct"] for v in comm_val_all]
            print(f" Comm model error: mean={np.mean(errs):.1f}%, max={np.max(errs):.1f}%")

        if memory_val:
            errs = [v["error_pct"] for v in memory_val]
            print(f" Memory model error: mean={np.mean(errs):.1f}%, max={np.max(errs):.1f}%")

        print(f"\n{'='*80}")

    if is_distributed:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

