#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Attention Computation Profiling + Piecewise Quadratic Fitting for AdaCPSP

Profiles Flash Attention computation time across different sequence lengths
and fits piecewise quadratic functions: time = a*x² + b*x + c

Key insight: Flash Attention uses different CUDA kernels for different seqlen ranges,
so a single quadratic fit is insufficient. We fit separate segments.

Segments (configurable):
  - short:       128 - 1024    (small kernel, dominated by launch overhead)
  - medium_low:  1024 - 4096   (transition region)
  - medium_high: 4096 - 8192   (main kernel)
  - long:        8192 - 32768  (main kernel, O(n²) dominant)
  - very_long:   32768+        (if GPU memory allows)

Usage:
    python profile_attention_fit.py \
        --n_heads 32 --n_kv_heads 32 --head_dim 128 \
        --model_name llama-7b --save_dir ./configs

Output:
    JSON with piecewise coefficients + PNG plot
"""

import os
import json
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import torch

# Import flash_attn
HAS_FLASH_ATTN = False
flash_attn_func = None
flash_attn_varlen_func = None

try:
    from flash_attn import flash_attn_func as _flash_attn_func
    flash_attn_func = _flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    pass

try:
    from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func
    flash_attn_varlen_func = _flash_attn_varlen_func
except ImportError:
    pass

if HAS_FLASH_ATTN:
    try:
        import flash_attn
        print(f"flash_attn version: {flash_attn.__version__}")
    except Exception:
        pass


def quadratic(x, a, b, c):
    """Quadratic function: f(x) = a*x² + b*x + c"""
    return a * x**2 + b * x + c


class AttentionProfiler:
    """Profile Flash Attention and fit piecewise quadratic functions."""

    def __init__(
        self,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        warmup_iters: int = 5,
        profile_iters: int = 20,
    ):
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.warmup_iters = warmup_iters
        self.profile_iters = profile_iters

        print(f"AttentionProfiler: n_heads={n_heads}, n_kv_heads={n_kv_heads}, "
              f"head_dim={head_dim}, GQA ratio={n_heads // n_kv_heads}")

    def profile_single_padded(self, seq_len: int, batch_size: int = 1) -> Optional[float]:
        """Profile with standard (padded) flash attention. Returns time in ms."""
        if flash_attn_func is None:
            return self._profile_manual(seq_len, batch_size)

        try:
            q = torch.randn(batch_size, seq_len, self.n_heads, self.head_dim,
                            dtype=self.dtype, device=self.device)
            k = torch.randn(batch_size, seq_len, self.n_kv_heads, self.head_dim,
                            dtype=self.dtype, device=self.device)
            v = torch.randn(batch_size, seq_len, self.n_kv_heads, self.head_dim,
                            dtype=self.dtype, device=self.device)

            for _ in range(self.warmup_iters):
                flash_attn_func(q, k, v, causal=True)
            torch.cuda.synchronize()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            for _ in range(self.profile_iters):
                flash_attn_func(q, k, v, causal=True)
            end.record()
            torch.cuda.synchronize()

            time_ms = start.elapsed_time(end) / self.profile_iters

            del q, k, v
            torch.cuda.empty_cache()
            return time_ms

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  OOM at seq_len={seq_len}")
                torch.cuda.empty_cache()
                return None
            raise

    def profile_single_varlen(self, seq_len: int, num_seqs: int = 1) -> Optional[float]:
        """Profile with varlen flash attention (packed sequences). Returns time in ms."""
        if flash_attn_varlen_func is None:
            return self.profile_single_padded(seq_len, batch_size=1)

        try:
            total_tokens = seq_len * num_seqs
            q = torch.randn(total_tokens, self.n_heads, self.head_dim,
                            dtype=self.dtype, device=self.device)
            k = torch.randn(total_tokens, self.n_kv_heads, self.head_dim,
                            dtype=self.dtype, device=self.device)
            v = torch.randn(total_tokens, self.n_kv_heads, self.head_dim,
                            dtype=self.dtype, device=self.device)
            cu_seqlens = torch.arange(0, total_tokens + 1, seq_len,
                                      dtype=torch.int32, device=self.device)

            for _ in range(self.warmup_iters):
                flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, seq_len, seq_len, causal=True)
            torch.cuda.synchronize()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            for _ in range(self.profile_iters):
                flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, seq_len, seq_len, causal=True)
            end.record()
            torch.cuda.synchronize()

            time_ms = start.elapsed_time(end) / self.profile_iters

            del q, k, v, cu_seqlens
            torch.cuda.empty_cache()
            return time_ms

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  OOM at seq_len={seq_len} (varlen)")
                torch.cuda.empty_cache()
                return None
            raise

    def _profile_manual(self, seq_len: int, batch_size: int = 1) -> Optional[float]:
        """Fallback: manual scaled dot-product attention."""
        try:
            q = torch.randn(batch_size, self.n_heads, seq_len, self.head_dim,
                            dtype=self.dtype, device=self.device)
            k = torch.randn(batch_size, self.n_kv_heads, seq_len, self.head_dim,
                            dtype=self.dtype, device=self.device)
            v = torch.randn(batch_size, self.n_kv_heads, seq_len, self.head_dim,
                            dtype=self.dtype, device=self.device)

            k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
            v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)

            scale = 1.0 / (self.head_dim ** 0.5)

            for _ in range(self.warmup_iters):
                attn = torch.matmul(q, k.transpose(-2, -1)) * scale
                attn = torch.softmax(attn, dim=-1)
                _ = torch.matmul(attn, v)
            torch.cuda.synchronize()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            for _ in range(self.profile_iters):
                attn = torch.matmul(q, k.transpose(-2, -1)) * scale
                attn = torch.softmax(attn, dim=-1)
                _ = torch.matmul(attn, v)
            end.record()
            torch.cuda.synchronize()

            time_ms = start.elapsed_time(end) / self.profile_iters
            del q, k, v
            torch.cuda.empty_cache()
            return time_ms

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                return None
            raise

    def profile_range(
        self,
        start: int,
        end: int,
        step: int,
        use_varlen: bool = False,
    ) -> List[Tuple[int, float]]:
        """Profile a range of sequence lengths."""
        results = []
        seq_lengths = list(range(start, end + 1, step))

        print(f"\nProfiling range [{start}, {end}] step={step} ({'varlen' if use_varlen else 'padded'})")
        print(f"  Total points: {len(seq_lengths)}")

        for i, seq_len in enumerate(seq_lengths):
            if use_varlen:
                time_ms = self.profile_single_varlen(seq_len)
            else:
                time_ms = self.profile_single_padded(seq_len)

            if time_ms is not None:
                results.append((seq_len, time_ms))
                print(f"  [{i+1}/{len(seq_lengths)}] seq={seq_len:>7}: {time_ms:.4f} ms")
            else:
                print(f"  [{i+1}/{len(seq_lengths)}] seq={seq_len:>7}: OOM, stopping range")
                break

        return results


def fit_piecewise_quadratic(
    all_results: Dict[str, List[Tuple[int, float]]],
) -> Dict[str, Optional[Dict]]:
    """Fit piecewise quadratic functions to profiled data."""
    from scipy.optimize import curve_fit

    coefficients = {}

    for segment_name, results in all_results.items():
        if len(results) < 3:
            print(f"  {segment_name}: Not enough data points ({len(results)})")
            coefficients[segment_name] = None
            continue

        seq_lens = np.array([r[0] for r in results], dtype=np.float64)
        times = np.array([r[1] for r in results], dtype=np.float64)

        try:
            p0 = [1e-9, 1e-6, 0.01]
            popt, _ = curve_fit(quadratic, seq_lens, times, p0=p0, maxfev=10000)
            a, b, c = popt

            y_pred = quadratic(seq_lens, a, b, c)
            ss_res = np.sum((times - y_pred) ** 2)
            ss_tot = np.sum((times - np.mean(times)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            errors = np.abs(times - y_pred)
            max_error = float(np.max(errors))
            mean_error = float(np.mean(errors))

            # Contribution analysis at midpoint
            mid_seq = seq_lens[len(seq_lens) // 2]
            term_a = a * mid_seq**2
            term_b = b * mid_seq
            term_c = c
            total = term_a + term_b + term_c

            coef = {
                "a": float(a),
                "b": float(b),
                "c": float(c),
                "r_squared": float(r_squared),
                "max_error_ms": max_error,
                "mean_error_ms": mean_error,
                "n_points": len(seq_lens),
                "seq_range": [int(seq_lens.min()), int(seq_lens.max())],
                "contribution_at_mid": {
                    "mid_seq": int(mid_seq),
                    "ax2_pct": float(term_a / total * 100) if total > 0 else 0,
                    "bx_pct": float(term_b / total * 100) if total > 0 else 0,
                    "c_pct": float(term_c / total * 100) if total > 0 else 0,
                },
            }
            coefficients[segment_name] = coef

            print(f"\n  {segment_name} fit: a={a:.6e}, b={b:.6e}, c={c:.4f}, R²={r_squared:.6f}")
            print(f"    Range: [{int(seq_lens.min())}, {int(seq_lens.max())}], Points: {len(seq_lens)}")
            print(f"    Max error: {max_error:.4f} ms, Mean error: {mean_error:.4f} ms")

        except Exception as e:
            print(f"  {segment_name}: Fitting failed: {e}")
            coefficients[segment_name] = None

    return coefficients


def plot_results(
    all_results: Dict[str, List[Tuple[int, float]]],
    coefficients: Dict[str, Optional[Dict]],
    save_path: str = None,
):
    """Plot profiling results with fitted curves."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    colors = {
        "short": "blue",
        "medium_low": "cyan",
        "medium_high": "green",
        "long": "orange",
        "very_long": "red",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. All data + fitted curves
    ax = axes[0, 0]
    for seg, results in all_results.items():
        if not results:
            continue
        xs = [r[0] for r in results]
        ys = [r[1] for r in results]
        color = colors.get(seg, "gray")
        ax.scatter(xs, ys, label=f"{seg} data", alpha=0.6, color=color, s=10)

        if seg in coefficients and coefficients[seg]:
            c = coefficients[seg]
            x_fit = np.linspace(min(xs), max(xs), 100)
            y_fit = quadratic(x_fit, c["a"], c["b"], c["c"])
            ax.plot(x_fit, y_fit, "--", color=color,
                    label=f'{seg} fit (R²={c["r_squared"]:.4f})')

    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Flash Attention Piecewise Profiling")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 2. Short sequences zoom
    ax = axes[0, 1]
    for seg in ["short", "medium_low"]:
        if seg not in all_results or not all_results[seg]:
            continue
        results = all_results[seg]
        xs = [r[0] for r in results]
        ys = [r[1] for r in results]
        color = colors.get(seg, "gray")
        ax.scatter(xs, ys, color=color, alpha=0.6, s=10, label=seg)
        if seg in coefficients and coefficients[seg]:
            c = coefficients[seg]
            x_fit = np.linspace(min(xs), max(xs), 100)
            y_fit = quadratic(x_fit, c["a"], c["b"], c["c"])
            ax.plot(x_fit, y_fit, "--", color=color)
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Short/Medium Sequences")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Coefficient 'a' comparison
    ax = axes[1, 0]
    valid = [s for s in coefficients if coefficients[s]]
    if valid:
        a_vals = [coefficients[s]["a"] * 1e9 for s in valid]
        x = np.arange(len(valid))
        ax.bar(x, a_vals, color=[colors.get(s, "gray") for s in valid], alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(valid, fontsize=8, rotation=15)
        ax.set_ylabel("a (×10⁹)")
        ax.set_title("Coefficient 'a' (O(n²) term)")
        ax.grid(True, alpha=0.3)

    # 4. R² comparison
    ax = axes[1, 1]
    if valid:
        r2_vals = [coefficients[s]["r_squared"] for s in valid]
        bars = ax.bar(valid, r2_vals, color=[colors.get(s, "gray") for s in valid], alpha=0.7)
        ax.axhline(y=0.99, color="r", linestyle="--", label="R²=0.99")
        ax.set_ylabel("R²")
        ax.set_title("Fitting Quality")
        ax.set_ylim(min(0.9, min(r2_vals) - 0.02), 1.005)
        ax.legend()
        ax.grid(True, alpha=0.3)
        for bar, r2 in zip(bars, r2_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{r2:.4f}", ha="center", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Attention Piecewise Profiling for AdaCPSP")
    parser.add_argument("--n_heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--n_kv_heads", type=int, default=32, help="Number of KV heads (GQA)")
    parser.add_argument("--head_dim", type=int, default=128, help="Head dimension")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--save_dir", type=str, default="./configs")
    parser.add_argument("--model_name", type=str, default="llama-7b")
    parser.add_argument("--use_varlen", action="store_true",
                        help="Use flash_attn_varlen_func instead of flash_attn_func")
    parser.add_argument("--skip_very_long", action="store_true",
                        help="Skip very long sequences (>32K)")
    args = parser.parse_args()

    print("=" * 70)
    print(" Attention Piecewise Profiler for AdaCPSP")
    print("=" * 70)
    print(f" Model: {args.model_name}")
    print(f" Config: n_heads={args.n_heads}, n_kv_heads={args.n_kv_heads}, head_dim={args.head_dim}")
    print(f" hidden_size = n_heads × head_dim = {args.n_heads * args.head_dim}")
    print(f" Mode: {'varlen' if args.use_varlen else 'padded'}")
    print("=" * 70)

    profiler = AttentionProfiler(
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        head_dim=args.head_dim,
        warmup_iters=args.warmup,
        profile_iters=args.iters,
    )

    # Define segments (piecewise regions)
    segments = {
        "short": {"start": 128, "end": 1024, "step": 128},
        "medium_low": {"start": 1024, "end": 4096, "step": 256},
        "medium_high": {"start": 4096, "end": 8192, "step": 256},
        "long": {"start": 8192, "end": 32768, "step": 1024},
    }
    if not args.skip_very_long:
        segments["very_long"] = {"start": 32768, "end": 131072, "step": 2048}

    # Profile each segment
    all_results = {}
    for seg_name, cfg in segments.items():
        print(f"\n{'=' * 70}")
        print(f" Segment: {seg_name}")
        print(f"{'=' * 70}")
        results = profiler.profile_range(
            cfg["start"], cfg["end"], cfg["step"],
            use_varlen=args.use_varlen,
        )
        all_results[seg_name] = results

    # Fit piecewise quadratic
    print(f"\n{'=' * 70}")
    print(" Piecewise Quadratic Fitting")
    print(f"{'=' * 70}")
    coefficients = fit_piecewise_quadratic(all_results)

    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output = {
        "type": "attention_piecewise_profiling",
        "model_name": args.model_name,
        "config": {
            "n_heads": args.n_heads,
            "n_kv_heads": args.n_kv_heads,
            "head_dim": args.head_dim,
            "hidden_size": args.n_heads * args.head_dim,
            "mode": "varlen" if args.use_varlen else "padded",
        },
        "segments": {k: v for k, v in segments.items()},
        "coefficients": coefficients,
        "raw_data": {k: [(int(s), float(t)) for s, t in v] for k, v in all_results.items()},
        "timestamp": timestamp,
    }

    json_path = os.path.join(args.save_dir, f"attention_fit_{args.model_name}_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # Plot
    plot_path = os.path.join(args.save_dir, f"attention_fit_{args.model_name}_{timestamp}.png")
    plot_results(all_results, coefficients, plot_path)

    # Summary
    print(f"\n{'=' * 70}")
    print(" Piecewise Coefficients Summary")
    print(f"{'=' * 70}")
    print(f"{'Segment':<14} | {'a (x²)':<14} | {'b (x)':<14} | {'c (1)':<10} | {'R²':<8} | {'Points':<6}")
    print("-" * 80)
    for seg in segments:
        if seg in coefficients and coefficients[seg]:
            c = coefficients[seg]
            print(f"{seg:<14} | {c['a']:<14.6e} | {c['b']:<14.6e} | {c['c']:<10.4f} | "
                  f"{c['r_squared']:<8.4f} | {c['n_points']:<6}")
        else:
            print(f"{seg:<14} | {'N/A':^14} | {'N/A':^14} | {'N/A':^10} | {'N/A':^8} | {'N/A':^6}")

    # Generate cost model helper output
    print(f"\n{'=' * 70}")
    print(" Cost Model Integration")
    print(f"{'=' * 70}")
    print("Piecewise function for AdaCPSPCostModel:")
    print("  def compute_time_single(self, seqlen, parallel_size=1):")
    for seg in segments:
        if seg in coefficients and coefficients[seg]:
            c = coefficients[seg]
            lo, hi = c["seq_range"]
            print(f"    # {seg}: [{lo}, {hi}]")
            print(f"    # time = {c['a']:.6e} * x² + {c['b']:.6e} * x + {c['c']:.4f}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

