#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Profile local attention backward/forward ratio.

This script is intentionally scoped to the attention wrapper only:
FlashSelfAttentionVarlen(q, k, v, cu_seqlens, max_seqlen).

The reported attn_bwd_fwd_ratio is:
    (forward_plus_backward_time - forward_only_time) / forward_only_time

Do not reuse this ratio for MLP/projection/optimizer.  Those parts are mostly
GEMM-dominated and may have a different backward/forward ratio.
"""

import argparse
import csv
import importlib.util
import json
import os
import statistics
import sys
from typing import Dict, List

import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(MODEL_DIR, "../../.."))
SITE_PACKAGE_DIR = os.path.join(REPO_ROOT, "galvatron", "site_package")
for path in (REPO_ROOT, MODEL_DIR, SITE_PACKAGE_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)


def load_attention_impl():
    module_path = os.path.join(
        REPO_ROOT, "galvatron", "core", "runtime", "tensor_parallel", "attention_impl.py"
    )
    spec = importlib.util.spec_from_file_location("ratio_attention_impl", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load attention_impl from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = (len(xs) - 1) * pct / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(xs) - 1)
    if lo == hi:
        return xs[lo]
    frac = idx - lo
    return xs[lo] + (xs[hi] - xs[lo]) * frac


def summarize(values: List[float]) -> Dict[str, float]:
    return {
        "min": min(values),
        "p50": statistics.median(values),
        "mean": statistics.mean(values),
        "p90": percentile(values, 90),
        "max": max(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def measure(fn, warmup: int, iters: int) -> List[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        values.append(float(start.elapsed_time(end)))
    return values


def profile_seq_len(module, seq_len: int, args, device, dtype) -> Dict:
    q_base = torch.randn(1, seq_len, args.n_heads, args.head_dim, device=device, dtype=dtype)
    k_base = torch.randn(1, seq_len, args.n_kv_heads, args.head_dim, device=device, dtype=dtype)
    v_base = torch.randn(1, seq_len, args.n_kv_heads, args.head_dim, device=device, dtype=dtype)
    cu = torch.tensor([0, seq_len], dtype=torch.int32, device=device)

    def fwd_only():
        q = q_base.detach().clone().requires_grad_(True)
        k = k_base.detach().clone().requires_grad_(True)
        v = v_base.detach().clone().requires_grad_(True)
        return module(q, k, v, cu, seq_len)

    def fwd_bwd():
        q = q_base.detach().clone().requires_grad_(True)
        k = k_base.detach().clone().requires_grad_(True)
        v = v_base.detach().clone().requires_grad_(True)
        out = module(q, k, v, cu, seq_len)
        loss = out.float().square().mean()
        loss.backward()
        return loss

    fwd_samples = measure(fwd_only, args.warmup, args.iters)
    fwd_bwd_samples = measure(fwd_bwd, args.warmup, args.iters)
    fwd_ms = statistics.median(fwd_samples)
    fwd_bwd_ms = statistics.median(fwd_bwd_samples)
    bwd_ms = fwd_bwd_ms - fwd_ms
    ratio = bwd_ms / fwd_ms if fwd_ms > 0 else 0.0

    del q_base, k_base, v_base, cu
    torch.cuda.empty_cache()

    return {
        "seq_len": seq_len,
        "forward_median_ms": fwd_ms,
        "fwd_bwd_median_ms": fwd_bwd_ms,
        "backward_estimated_ms": bwd_ms,
        "attn_bwd_fwd_ratio": ratio,
        "attn_fwd_bwd_over_fwd": fwd_bwd_ms / fwd_ms if fwd_ms > 0 else 0.0,
        "forward_samples_ms": fwd_samples,
        "fwd_bwd_samples_ms": fwd_bwd_samples,
        "forward_stats_ms": summarize(fwd_samples),
        "fwd_bwd_stats_ms": summarize(fwd_bwd_samples),
    }


def main():
    parser = argparse.ArgumentParser(description="Profile attention-only backward/forward ratio")
    parser.add_argument("--seq-start", type=int, default=4096)
    parser.add_argument("--seq-end", type=int, default=16384)
    parser.add_argument("--seq-step", type=int, default=1024)
    parser.add_argument("--n-heads", type=int, default=28)
    parser.add_argument("--n-kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=6)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    attn_impl = load_attention_impl()
    module = attn_impl.FlashSelfAttentionVarlen(
        causal=True, attention_dropout=0.0
    ).to(device)
    module.train()

    results = []
    for seq_len in range(args.seq_start, args.seq_end + 1, args.seq_step):
        row = profile_seq_len(module, seq_len, args, device, dtype)
        results.append(row)
        print(
            f"seq={seq_len:>6} "
            f"fwd={row['forward_median_ms']:.4f}ms "
            f"fwd+bwd={row['fwd_bwd_median_ms']:.4f}ms "
            f"attn_bwd_fwd_ratio={row['attn_bwd_fwd_ratio']:.4f} "
            f"total/fwd={row['attn_fwd_bwd_over_fwd']:.4f}"
        )

    output = {
        "scope": "attention_only_flash_self_attention_varlen",
        "note": "attn_bwd_fwd_ratio is for attention only; do not apply to MLP/projection/optimizer.",
        "config": vars(args),
        "results": results,
    }

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
        with open(args.output_csv, "w", newline="") as f:
            fieldnames = [
                "seq_len",
                "forward_median_ms",
                "fwd_bwd_median_ms",
                "backward_estimated_ms",
                "attn_bwd_fwd_ratio",
                "attn_fwd_bwd_over_fwd",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow({k: row[k] for k in fieldnames})


if __name__ == "__main__":
    main()
