#!/usr/bin/env python3
# Copyright 2026 — bench harness (standalone; avoids megatron import chain).
"""
8-GPU micro-benchmark: varlen zigzag ring attention forward, with / without the
small CUDA preempt kernel (same idea as non-varlen zigzag in attention_impl.py).

Production `attention_impl.zigzag_ring_flash_attn_varlen_forward` always applies
this preempt; this script only turns it on/off for A/B timing.

Run (single node):
  torchrun --nproc_per_node=8 bench_varlen_zigzag_ring_comm_preempt.py \\
      --local_seqlen 4096 --n_heads 8 --head_dim 64 --warmup 10 --iters 30

Optional: write JSON summary
  ... --json_out /tmp/varlen_ring_preempt.json
"""
from __future__ import annotations

import argparse
import json
import os
import inspect
import math
from functools import cache
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from flash_attn.flash_attn_interface import _flash_attn_varlen_forward


# ----- minimal copies of ring-flash-attn helpers (mirror attention_impl.py) -----


@torch.jit.script
def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)
    return out, lse


def update_out_and_lse(
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    slice_=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        if slice_ is not None:
            raise RuntimeError("first update_out_and_lse should not pass slice_ args")
        out = block_out.to(torch.float32)
        lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    elif slice_ is not None:
        slice_out, slice_lse = out[slice_], lse[slice_]
        slice_out, slice_lse = _update_out_and_lse(slice_out, slice_lse, block_out, block_lse)
        out[slice_], lse[slice_] = slice_out, slice_lse
    else:
        out, lse = _update_out_and_lse(out, lse, block_out, block_lse)
    return out, lse


@torch.jit.script
def flatten_varlen_lse(lse, cu_seqlens):
    new_lse = []
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        new_lse.append(lse[i, :, : end - start])
    return torch.cat(new_lse, dim=1)


@torch.jit.script
def unflatten_varlen_lse(lse, cu_seqlens, max_seqlen: int):
    num_seq = len(cu_seqlens) - 1
    num_head = lse.shape[-2]
    new_lse = torch.empty(
        (num_seq, max_seqlen, num_head, 1), dtype=torch.float32, device=lse.device
    )
    for i in range(num_seq):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        new_lse[i, : end - start] = lse[start:end]
    return new_lse.squeeze(dim=-1).transpose(1, 2).contiguous()


@torch.jit.script
def get_half_lse(lse, cu_seqlens, *, front: bool):
    if lse.dim() == 2:
        new_lse = torch.empty(
            (lse.shape[0], lse.shape[1] // 2),
            dtype=lse.dtype,
            device=lse.device,
        )
        for i in range(len(cu_seqlens) - 1):
            start, end = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
            new_start, new_end = start // 2, end // 2
            if front:
                end -= (end - start) // 2
            else:
                start += (end - start) // 2
            new_lse[:, new_start:new_end] = lse[:, start:end]
    else:
        new_lse = torch.empty(
            (lse.shape[0], lse.shape[1], lse.shape[2] // 2),
            dtype=lse.dtype,
            device=lse.device,
        )
        for i in range(len(cu_seqlens) - 1):
            seqlen = (cu_seqlens[i + 1] - cu_seqlens[i]).item()
            if front:
                start, end = 0, seqlen // 2
            else:
                start, end = seqlen // 2, seqlen
            new_lse[i, :, : seqlen // 2] = lse[i, :, start:end]
    return new_lse


def get_half_index(cu_seqlens, *, front: bool):
    if len(cu_seqlens) == 2:
        half = int(cu_seqlens[-1].item()) // 2
        if front:
            return slice(None, half)
        return slice(half, None)
    index = torch.zeros((cu_seqlens[-1].item(),), dtype=torch.bool, device=cu_seqlens.device)
    for i in range(len(cu_seqlens) - 1):
        start, end = int(cu_seqlens[i].item()), int(cu_seqlens[i + 1].item())
        if front:
            end = (start + end) // 2
        else:
            start = (start + end) // 2
        index[start:end] = True
    return index


@cache
def _get_default_args(func):
    spec = inspect.getfullargspec(func)
    defaults = spec.defaults if spec.defaults is not None else ()
    padded_defaults = (None,) * (len(spec.args) - len(defaults)) + defaults
    args = dict(zip(spec.args, padded_defaults))
    if "softcap" in args:
        args["softcap"] = 0.0
    return args


def get_default_args(func):
    return _get_default_args(func)


class RingComm:
    def __init__(self, process_group: dist.ProcessGroup, batch_comm: bool = True):
        self.batch_comm = batch_comm
        self._process_group = process_group
        self._ops = []
        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)
        self._reqs = None
        self._send_reqs = []
        self._recv_reqs = []
        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size
        if process_group is not None:
            self.send_rank = dist.get_global_rank(self._process_group, self.send_rank)
            self.recv_rank = dist.get_global_rank(self._process_group, self.recv_rank)

    def send_recv(
        self, to_send: torch.Tensor, recv_tensor: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if recv_tensor is None:
            res = torch.empty_like(to_send)
        else:
            res = recv_tensor
        if self.batch_comm:
            send_op = dist.P2POp(dist.isend, to_send, self.send_rank, group=self._process_group)
            recv_op = dist.P2POp(dist.irecv, res, self.recv_rank, group=self._process_group)
            self._ops.append(send_op)
            self._ops.append(recv_op)
        else:
            if self.rank % 2 == 0:
                send_req = dist.isend(to_send, self.send_rank, group=self._process_group)
                recv_req = dist.irecv(res, self.recv_rank, group=self._process_group)
            else:
                recv_req = dist.irecv(res, self.recv_rank, group=self._process_group)
                send_req = dist.isend(to_send, self.send_rank, group=self._process_group)
            self._recv_reqs.append(recv_req)
            self._send_reqs.append(send_req)
        return res

    def commit(self):
        if self.batch_comm:
            if self._reqs is not None:
                raise RuntimeError("commit called twice")
            self._reqs = dist.batch_isend_irecv(self._ops)
        pass

    def wait(self):
        if self.batch_comm:
            if self._reqs is None:
                raise RuntimeError("wait called before commit")
            for req in self._reqs:
                req.wait()
            self._reqs = None
            self._ops = []
        else:
            for req in self._recv_reqs:
                req.wait()
            self._send_reqs.clear()
            self._recv_reqs.clear()

    def send_recv_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_buffer: Optional[torch.Tensor] = None,
        v_buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        next_k, next_v = self.send_recv(k, k_buffer), self.send_recv(v, v_buffer)
        self.commit()
        return next_k, next_v


def zigzag_ring_flash_attn_varlen_forward_bench(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens,
    max_seqlen: int,
    half_index0,
    half_index1,
    softmax_scale: float,
    *,
    use_comm_preempt: bool,
    dropout_p: float = 0.0,
    causal: bool = True,
    window_size=(-1, -1),
    alibi_slopes=None,
):
    assert causal is True
    comm = RingComm(process_group)
    block_seq_len = q.shape[0] // 2
    q1 = q[half_index1]
    out = None
    lse = None
    next_k, next_v = None, None
    half_cu_seqlens = cu_seqlens // 2
    half_max_seqlen = max_seqlen // 2

    def forward_blk(q_, k_, v_, causal_):
        seqlen_q = q_.shape[0]
        seqlen_kv = k_.shape[0]
        cu_seqlens_q = half_cu_seqlens if seqlen_q == block_seq_len else cu_seqlens
        max_seqlen_q = half_max_seqlen if seqlen_q == block_seq_len else max_seqlen
        cu_seqlens_kv = half_cu_seqlens if seqlen_kv == block_seq_len else cu_seqlens
        max_seqlen_kv = half_max_seqlen if seqlen_kv == block_seq_len else max_seqlen
        params = get_default_args(_flash_attn_varlen_forward).copy()
        params.update(
            {
                "q": q_,
                "k": k_,
                "v": v_,
                "cu_seqlens_q": cu_seqlens_q,
                "cu_seqlens_k": cu_seqlens_kv,
                "max_seqlen_q": max_seqlen_q,
                "max_seqlen_k": max_seqlen_kv,
                "dropout_p": dropout_p,
                "softmax_scale": softmax_scale,
                "causal": causal_,
                "alibi_slopes": alibi_slopes,
                "return_softmax": dropout_p > 0,
            }
        )
        if "window_size" in params:
            params.update({"window_size": window_size})
        else:
            params.update(
                {"window_size_left": window_size[0], "window_size_right": window_size[1]}
            )
        outputs = _flash_attn_varlen_forward(**params)
        if len(outputs) == 8:
            block_out, _, _, _, _, block_lse, _, _ = outputs
        else:
            assert len(outputs) == 4
            block_out, block_lse, _, _ = outputs
        return block_out, block_lse

    old_lse = False
    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k, next_v = comm.send_recv_kv(k, v)

        if step == 0:
            if use_comm_preempt:
                _ = torch.zeros((1,), device=torch.cuda.current_device())
            block_out, block_lse = forward_blk(q, k, v, True)
            if block_lse.dim() == 3:
                old_lse = True
                block_lse = flatten_varlen_lse(block_lse, cu_seqlens)
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        elif step <= comm.rank:
            k0 = k[half_index0]
            v0 = v[half_index0]
            if use_comm_preempt:
                _ = torch.zeros((1,), device=torch.cuda.current_device())
            block_out, block_lse = forward_blk(q, k0, v0, False)
            if block_lse.dim() == 3:
                old_lse = True
                block_lse = flatten_varlen_lse(block_lse, cu_seqlens)
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
        else:
            if use_comm_preempt:
                _ = torch.zeros((1,), device=torch.cuda.current_device())
            block_out, block_lse = forward_blk(q1, k, v, False)
            if block_lse.dim() == 3:
                old_lse = True
                block_lse = flatten_varlen_lse(block_lse, half_cu_seqlens)
            out[half_index1], lse[half_index1] = update_out_and_lse(
                out[half_index1], lse[half_index1], block_out, block_lse
            )

        if step + 1 != comm.world_size:
            comm.wait()
            k, v = next_k, next_v

    out = out.to(q.dtype)
    if old_lse:
        lse = unflatten_varlen_lse(lse, cu_seqlens, max_seqlen)
    else:
        lse = lse.squeeze(dim=-1).transpose(0, 1)
    return out, lse


def _fixed_random(local_seqlen: int, n_heads: int, head_dim: int, dtype, device, seed_base: int):
    g = torch.Generator(device=device)
    g.manual_seed(seed_base + dist.get_rank())
    q = torch.randn(local_seqlen, n_heads, head_dim, generator=g, dtype=dtype, device=device)
    k = torch.randn(local_seqlen, n_heads, head_dim, generator=g, dtype=dtype, device=device)
    v = torch.randn(local_seqlen, n_heads, head_dim, generator=g, dtype=dtype, device=device)
    return q, k, v


def _build_cu_seqlens(local_seqlen: int, n_seqs: int, device):
    assert local_seqlen % n_seqs == 0
    chunk = local_seqlen // n_seqs
    offs = list(range(0, local_seqlen + 1, chunk))
    return torch.tensor(offs, dtype=torch.int32, device=device)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--local_seqlen", type=int, default=4096)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--head_dim", type=int, default=64)
    p.add_argument("--n_seqs", type=int, default=4, help="packed seqs per rank; divides local_seqlen")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--json_out", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 8:
        if rank == 0:
            print(f"[WARN] expect 8 processes for this bench, got {world_size}")

    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.bfloat16
    local_seqlen = args.local_seqlen
    assert local_seqlen % 2 == 0
    cu = _build_cu_seqlens(local_seqlen, args.n_seqs, device)
    max_seqlen = local_seqlen // args.n_seqs
    half0 = get_half_index(cu, front=True)
    half1 = get_half_index(cu, front=False)
    softmax_scale = args.head_dim ** -0.5

    # Reference outputs for equivalence (same inputs; preempt should not change math)
    qref, kref, vref = _fixed_random(
        local_seqlen, args.n_heads, args.head_dim, dtype, device, seed_base=12345
    )
    with torch.no_grad():
        out_on, _ = zigzag_ring_flash_attn_varlen_forward_bench(
            dist.group.WORLD,
            qref,
            kref,
            vref,
            cu,
            max_seqlen,
            half0,
            half1,
            softmax_scale,
            use_comm_preempt=True,
        )
        out_off, _ = zigzag_ring_flash_attn_varlen_forward_bench(
            dist.group.WORLD,
            qref,
            kref,
            vref,
            cu,
            max_seqlen,
            half0,
            half1,
            softmax_scale,
            use_comm_preempt=False,
        )
    diff = (out_on.float() - out_off.float()).abs().max().item()
    if rank == 0:
        print(f"[check] max |out_preempt_on - out_preempt_off| = {diff:g} (expect ~0)")

    def bench(use_preempt: bool):
        q, k, v = _fixed_random(
            local_seqlen, args.n_heads, args.head_dim, dtype, device, seed_base=99999
        )
        torch.cuda.synchronize()
        for _ in range(args.warmup):
            with torch.no_grad():
                zigzag_ring_flash_attn_varlen_forward_bench(
                    dist.group.WORLD,
                    q,
                    k,
                    v,
                    cu,
                    max_seqlen,
                    half0,
                    half1,
                    softmax_scale,
                    use_comm_preempt=use_preempt,
                )
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            with torch.no_grad():
                zigzag_ring_flash_attn_varlen_forward_bench(
                    dist.group.WORLD,
                    q,
                    k,
                    v,
                    cu,
                    max_seqlen,
                    half0,
                    half1,
                    softmax_scale,
                    use_comm_preempt=use_preempt,
                )
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / args.iters

    t_off = bench(False)
    t_on = bench(True)
    if rank == 0:
        faster = (t_off - t_on) / t_off * 100.0 if t_off > 0 else 0.0
        print(
            f"[time] preempt=OFF  {t_off:.3f} ms/iter | preempt=ON {t_on:.3f} ms/iter "
            f"({faster:+.2f}% vs off)"
        )
        summary = {
            "world_size": world_size,
            "local_seqlen": local_seqlen,
            "n_heads": args.n_heads,
            "head_dim": args.head_dim,
            "ms_iter_preempt_off": t_off,
            "ms_iter_preempt_on": t_on,
            "pct_faster_with_preempt": faster,
            "max_abs_out_diff_on_vs_off": diff,
        }
        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"[json] wrote {args.json_out}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
