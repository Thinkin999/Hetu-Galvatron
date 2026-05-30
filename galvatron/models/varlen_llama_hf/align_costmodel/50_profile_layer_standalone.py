"""
Standalone single-GPU layer-diff profiler for the NON-ATTENTION residual.

First-principles design (no train_dist, no optimizer, no distributed → no OOM,
no all-gather, fully controlled):

  * Build a REAL Qwen2ForCausalLM with N_LAYERS decoder layers (real GQA kv=4,
    hidden, ffn, vocab, flash_attention_2).
  * Time forward+backward with CUDA events (warmup + median), NO optimizer step
    (we want the fwd+bwd compute that the cost model's `a*tokens + b` predicts;
    optimizer is a separate, separately-measured term).
  * Vary N_LAYERS ∈ {1,2,4} and seq. Layer-diff:
        per_layer(seq) = (T(N_hi,seq) - T(N_lo,seq)) / (N_hi - N_lo)
    isolates the per-decoder-layer fwd+bwd time (attention + linear), and the
    intercept T(N_lo) - N_lo*per_layer = embed + LM-head + fixed.
  * Also time flash-attention ALONE (same shapes) so the caller can subtract it:
        per_layer_linear = per_layer - attn_per_layer
    (the cost model already models flash-attn compute via its LUT; the residual
     `a` is the NON-flash-attention linear part: QKVO proj + MLP + LN.)
  * Memory: torch.cuda.max_memory_allocated for activation per token.

Usage:
  python 50_profile_layer_standalone.py --layers 1 2 4 --seqs 2048 4096 8192 16384 \
      --ckpt 0 --out results/layerdiff_<ts>.json
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from statistics import median

import torch

QWEN_HIDDEN = 3584
QWEN_HEADS = 28
QWEN_KV = 4
QWEN_FFN = 18944
QWEN_VOCAB = 152064


def build_model(n_layers: int, vocab: int, ckpt: bool, dtype=torch.bfloat16):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2Config
    cfg = Qwen2Config(
        hidden_size=QWEN_HIDDEN, num_attention_heads=QWEN_HEADS,
        num_key_value_heads=QWEN_KV, intermediate_size=QWEN_FFN,
        num_hidden_layers=n_layers, vocab_size=vocab,
        max_position_embeddings=200000, rms_norm_eps=1e-6,
        attn_implementation="flash_attention_2", use_cache=False,
        tie_word_embeddings=False,
    )
    model = Qwen2ForCausalLM(cfg).to("cuda", dtype=dtype)
    if ckpt:
        model.gradient_checkpointing_enable()
    model.train()
    return model


def time_fwd_bwd(model, seq: int, iters: int = 8, warmup: int = 3):
    """Return (median_fwd_bwd_ms, peak_activation_mb). batch=1, single packed seq."""
    dev = "cuda"
    input_ids = torch.randint(0, model.config.vocab_size, (1, seq), device=dev)
    labels = input_ids.clone()
    # baseline memory (params+grad buffers) after a dummy step
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated() / 1024 / 1024

    times = []
    peak_mb = 0.0
    for it in range(warmup + iters):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = model(input_ids=input_ids, labels=labels)
        loss = out.loss
        loss.backward()
        end.record()
        torch.cuda.synchronize()
        model.zero_grad(set_to_none=True)
        if it >= warmup:
            times.append(start.elapsed_time(end))
            peak_mb = max(peak_mb, torch.cuda.max_memory_allocated() / 1024 / 1024)
    return median(times), peak_mb, base_mem


def time_flash_attn_per_layer(seq: int, iters: int = 10, warmup: int = 3, dtype=torch.bfloat16):
    """Time ONE flash-attention call (fwd+bwd) at this seq with Qwen GQA shapes,
    matching what the cost model's attention LUT represents."""
    from flash_attn import flash_attn_varlen_func
    dev = "cuda"
    hd = QWEN_HIDDEN // QWEN_HEADS  # 128
    q = torch.randn(seq, QWEN_HEADS, hd, device=dev, dtype=dtype, requires_grad=True)
    k = torch.randn(seq, QWEN_KV, hd, device=dev, dtype=dtype, requires_grad=True)
    v = torch.randn(seq, QWEN_KV, hd, device=dev, dtype=dtype, requires_grad=True)
    cu = torch.tensor([0, seq], device=dev, dtype=torch.int32)
    times = []
    for it in range(warmup + iters):
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        o = flash_attn_varlen_func(q, k, v, cu, cu, seq, seq, causal=True)
        o.sum().backward()
        e.record(); torch.cuda.synchronize()
        q.grad = k.grad = v.grad = None
        if it >= warmup:
            times.append(s.elapsed_time(e))
    return median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--seqs", type=int, nargs="+", default=[2048, 4096, 8192, 16384])
    ap.add_argument("--vocab", type=int, default=QWEN_VOCAB)
    ap.add_argument("--ckpt", type=int, default=0)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    print(f"# device: {torch.cuda.get_device_name(0)}")
    print(f"# config: hidden={QWEN_HIDDEN} heads={QWEN_HEADS} kv={QWEN_KV} ffn={QWEN_FFN} "
          f"vocab={args.vocab} ckpt={args.ckpt}")
    print(f"# layers={args.layers} seqs={args.seqs}\n")

    results = {"config": {"hidden": QWEN_HIDDEN, "heads": QWEN_HEADS, "kv": QWEN_KV,
                          "ffn": QWEN_FFN, "vocab": args.vocab, "ckpt": args.ckpt},
               "cells": [], "flash_attn_per_layer": {}}

    # flash-attn-only per seq (for subtraction)
    print("=== flash-attn-only (fwd+bwd) per seq ===")
    for seq in args.seqs:
        t = time_flash_attn_per_layer(seq)
        results["flash_attn_per_layer"][str(seq)] = t
        print(f"  seq={seq:>6}: flash_attn_fb={t:.3f} ms")
    print()

    # full-model fwd+bwd at each (layers, seq)
    print(f"{'layers':>6} {'seq':>7} {'fb_ms':>9} {'peak_mb':>9} {'base_mb':>9}")
    for L in args.layers:
        model = build_model(L, args.vocab, bool(args.ckpt))
        for seq in args.seqs:
            try:
                fb, peak, base = time_fwd_bwd(model, seq, iters=args.iters)
                results["cells"].append({"layers": L, "seq": seq, "fb_ms": fb,
                                          "peak_mb": peak, "base_mb": base})
                print(f"{L:>6} {seq:>7} {fb:>9.3f} {peak:>9.0f} {base:>9.0f}")
            except torch.cuda.OutOfMemoryError:
                print(f"{L:>6} {seq:>7}    OOM")
                torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()

    # ---- layer-diff analysis ----
    print("\n=== layer-diff: per-layer fwd+bwd, and per-layer LINEAR (minus flash-attn) ===")
    print(f"{'seq':>7} {'per_layer_ms':>13} {'attn_ms':>9} {'per_layer_linear':>17} {'lin/token(us)':>14}")
    by_seq = {}
    for seq in args.seqs:
        pts = [(c["layers"], c["fb_ms"]) for c in results["cells"] if c["seq"] == seq]
        if len(pts) < 2:
            continue
        pts.sort()
        # linear regression fb = per_layer*L + intercept
        n = len(pts); sx = sum(p[0] for p in pts); sy = sum(p[1] for p in pts)
        sxx = sum(p[0]**2 for p in pts); sxy = sum(p[0]*p[1] for p in pts)
        per_layer = (n*sxy - sx*sy) / (n*sxx - sx*sx)
        intercept = (sy - per_layer*sx) / n
        attn = results["flash_attn_per_layer"][str(seq)]
        per_layer_linear = per_layer - attn
        lin_per_tok_us = per_layer_linear / seq * 1000.0
        by_seq[seq] = {"per_layer_ms": per_layer, "intercept_ms": intercept,
                       "attn_ms": attn, "per_layer_linear_ms": per_layer_linear,
                       "linear_per_token_us": lin_per_tok_us}
        print(f"{seq:>7} {per_layer:>13.3f} {attn:>9.3f} {per_layer_linear:>17.3f} {lin_per_tok_us:>14.3f}")
    results["layer_diff"] = by_seq

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n# saved: {args.out}")


if __name__ == "__main__":
    main()
