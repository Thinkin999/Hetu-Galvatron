#!/usr/bin/env python3
"""
Correctness checks for varlen zigzag ring attention (attention_impl).

1) world_size == 1: ring forward reduces to one causal flash-varlen block →
   compare to direct _flash_attn_varlen_forward (same tolerance as bf16).

2) world_size >= 1: run forward twice on same inputs → max diff should be tiny
   (bf16 + FA non-determinism bound).

3) world_size >= 1: full forward + backward → grads finite (no nan/inf).

Usage (use megatorn_cu121_py39_lqs or any env with flash-attn + megatron deps):

  cd /path/to/Hetu-Galvatron
  export PYTHONPATH=galvatron/site_package:.

  # Single GPU: strict forward vs reference FA
  conda run -n megatorn_cu121_py39_lqs torchrun --nproc_per_node=1 \\
    galvatron/models/varlen_llama_hf/verify_varlen_zigzag_ring_correctness.py

  # 8 GPUs: stability + backward smoke (set an unused port if 23456 is busy)
  conda run -n megatorn_cu121_py39_lqs torchrun --nproc_per_node=8 --master_port=29501 \\
    galvatron/models/varlen_llama_hf/verify_varlen_zigzag_ring_correctness.py
"""
from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist
from flash_attn.flash_attn_interface import _flash_attn_varlen_forward

# Repo root: .../Hetu-Galvatron
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_site = os.path.join(_REPO_ROOT, "galvatron", "site_package")
if os.path.isdir(_site) and _site not in sys.path:
    sys.path.insert(0, _site)

from galvatron.core.runtime.tensor_parallel.attention_impl import (  # noqa: E402
    get_default_args,
    get_half_index,
    zigzag_ring_flash_attn_varlen_forward,
    zigzag_ring_flash_attn_varlen_func,
)


def _ref_flash_varlen_causal(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_secls: torch.Tensor,
    max_seqlen: int,
    softmax_scale: float,
) -> torch.Tensor:
    params = get_default_args(_flash_attn_varlen_forward).copy()
    params.update(
        {
            "q": q,
            "k": k,
            "v": v,
            "cu_seqlens_q": cu_secls,
            "cu_seqlens_k": cu_secls,
            "max_seqlen_q": max_seqlen,
            "max_seqlen_k": max_seqlen,
            "dropout_p": 0.0,
            "softmax_scale": softmax_scale,
            "causal": True,
            "alibi_slopes": None,
            "return_softmax": False,
        }
    )
    if "window_size" in params:
        params.update({"window_size": (-1, -1)})
    else:
        params.update({"window_size_left": -1, "window_size_right": -1})
    outputs = _flash_attn_varlen_forward(**params)
    if len(outputs) == 8:
        block_out, _, _, _, _, _, _, _ = outputs
    else:
        block_out, _, _, _ = outputs
    return block_out


def main() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.bfloat16
    softmax_scale = 64**-0.5

    failures = 0

    # ---- Build local packed batch (even total tokens for zigzag) ----
    local_seqlen = 512
    assert local_seqlen % 2 == 0
    # Multi-sequence packed (exercises non-slice half_index path when len(cu)>2)
    n_seqs = 4
    assert local_seqlen % n_seqs == 0
    chunk = local_seqlen // n_seqs
    cu = torch.tensor(
        list(range(0, local_seqlen + 1, chunk)), dtype=torch.int32, device=device
    )
    max_seqlen = chunk
    half0 = get_half_index(cu, front=True)
    half1 = get_half_index(cu, front=False)

    g = torch.Generator(device=device)
    g.manual_seed(12345 + rank)
    q = torch.randn(
        local_seqlen, 8, 64, generator=g, dtype=dtype, device=device, requires_grad=False
    )
    k = torch.randn(
        local_seqlen, 8, 64, generator=g, dtype=dtype, device=device, requires_grad=False
    )
    v = torch.randn(
        local_seqlen, 8, 64, generator=g, dtype=dtype, device=device, requires_grad=False
    )

    k = k.contiguous()
    v = v.contiguous()

    # ---- Test 1: world_size == 1 vs reference flash-varlen ----
    if world == 1:
        out_ring, _ = zigzag_ring_flash_attn_varlen_forward(
            dist.group.WORLD,
            q,
            k,
            v,
            cu,
            max_seqlen,
            half0,
            half1,
            softmax_scale,
            dropout_p=0.0,
            causal=True,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=False,
        )
        out_ref = _ref_flash_varlen_causal(q, k, v, cu, max_seqlen, softmax_scale)
        max_err = (out_ring.float() - out_ref.float()).abs().max().item()
        # One step of ring forward == single FA block for ws=1; allow bf16+kernel noise.
        tol = 0.02
        ok = max_err < tol
        if rank == 0:
            print(
                f"[1] ws=1 forward vs _flash_attn_varlen_forward: max_err={max_err:.6g} "
                f"(tol {tol}) {'PASS' if ok else 'FAIL'}"
            )
        failures += 0 if ok else 1

    # ---- Test 2: forward repeatability ----
    out_a, _ = zigzag_ring_flash_attn_varlen_forward(
        dist.group.WORLD,
        q,
        k,
        v,
        cu,
        max_seqlen,
        half0,
        half1,
        softmax_scale,
    )
    out_b, _ = zigzag_ring_flash_attn_varlen_forward(
        dist.group.WORLD,
        q,
        k,
        v,
        cu,
        max_seqlen,
        half0,
        half1,
        softmax_scale,
    )
    d_repeat = (out_a.float() - out_b.float()).abs().max().item()
    ok2 = d_repeat < 0.05
    if rank == 0:
        print(
            f"[2] forward repeat max |a-b|={d_repeat:.6g} (tol 0.05) {'PASS' if ok2 else 'FAIL'}"
        )
    failures += 0 if ok2 else 1

    # ---- Test 3: autograd backward, finite grads ----
    q2 = q.clone().detach().requires_grad_(True)
    k2 = k.clone().detach().requires_grad_(True)
    v2 = v.clone().detach().requires_grad_(True)

    out = zigzag_ring_flash_attn_varlen_func(
        q2,
        k2,
        v2,
        cu,
        max_seqlen,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        group=dist.group.WORLD,
    )
    loss = out.float().square().mean()
    loss.backward()
    bad = False
    for name, t in [("q", q2), ("k", k2), ("v", v2)]:
        g_ = t.grad
        if g_ is None or not torch.isfinite(g_).all():
            bad = True
            if rank == 0:
                print(f"[3] grad check: {name} has non-finite or None grad FAIL")
    if rank == 0 and not bad:
        gn = q2.grad.norm().item()
        print(f"[3] backward: grad finite, ||grad_q||={gn:.6g} PASS")
    failures += 1 if bad else 0

    dist.barrier()
    if rank == 0:
        if failures == 0:
            print("All checks PASSED.")
        else:
            print(f"FAILED with {failures} check(s).")
    dist.destroy_process_group()
    if failures > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
