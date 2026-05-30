"""Quick fresh measurement of Ring p16 attention layer time for 3 seq lengths.

Mirrors 03_benchmark_attention.run_once but loops over 3 seq lengths in a
single torchrun process so we save 5-7 minutes of init time.

Outputs a JSON {seq: measured_ms} and prints the table together with current
cost-model prediction for the same (seq, num_seqs=16) -- so we get a
self-consistent measured-vs-predicted snapshot.
"""

import argparse
import json
import os
import socket
import statistics
import sys
from typing import List

import torch
import torch.distributed as dist

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(MODEL_DIR, "../../.."))
for p in (REPO_ROOT, MODEL_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from galvatron.core.runtime.tensor_parallel.attention_impl import (  # noqa: E402
    ZigzagRingFlashAttentionVarlen,
)


def make_cu_seqlens(num_seqs: int, seq_len: int, device) -> torch.Tensor:
    return torch.arange(0, (num_seqs + 1) * seq_len, step=seq_len,
                        dtype=torch.int32, device=device)


def build_inputs(seq_len: int, num_seqs: int, n_heads: int, n_kv_heads: int,
                 head_dim: int, cp_size: int, device, dtype):
    cp_local_seq = seq_len // cp_size
    cu = make_cu_seqlens(num_seqs, cp_local_seq, device)
    local_tokens = num_seqs * cp_local_seq
    q = torch.randn(1, local_tokens, n_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(1, local_tokens, n_kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(1, local_tokens, n_kv_heads, head_dim, device=device, dtype=dtype)
    return q, k, v, cu, cp_local_seq


def run_one_seq(module, q, k, v, cu, max_seqlen) -> None:
    q_ = q.detach().clone().requires_grad_(True)
    k_ = k.detach().clone().requires_grad_(True)
    v_ = v.detach().clone().requires_grad_(True)
    out = module(q_, k_, v_, cu, max_seqlen)
    loss = out.float().square().mean()
    loss.backward()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seqs", default="8192,16384,32768")
    p.add_argument("--num-seqs", type=int, default=16)
    p.add_argument("--n-heads", type=int, default=28)
    p.add_argument("--n-kv-heads", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--output-json", default=None)
    args = p.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    if rank == 0:
        print(f"[bench] world={world} host={socket.gethostname()}", flush=True)

    cp_size = world
    cp_group = dist.new_group(list(range(world)))
    cp_ranks = list(range(world))

    seq_list: List[int] = [int(x) for x in args.seqs.split(",")]
    dtype = torch.bfloat16

    results = {}
    for seq_len in seq_list:
        if rank == 0:
            print(f"\n[bench] seq={seq_len} num_seqs={args.num_seqs} cp={cp_size}",
                  flush=True)
        try:
            q, k, v, cu, max_seqlen = build_inputs(
                seq_len, args.num_seqs, args.n_heads, args.n_kv_heads,
                args.head_dim, cp_size, torch.device("cuda"), dtype,
            )
            module = ZigzagRingFlashAttentionVarlen(
                attention_dropout=0.0, cp_group=cp_group,
                cp_ranks=cp_ranks, causal=True,
            )
            for _ in range(args.warmup):
                run_one_seq(module, q, k, v, cu, max_seqlen)
            torch.cuda.synchronize()
            dist.barrier()

            per_iter = []
            for _ in range(args.iters):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                run_one_seq(module, q, k, v, cu, max_seqlen)
                end.record()
                torch.cuda.synchronize()
                ms = start.elapsed_time(end)
                t = torch.tensor([ms], device="cuda", dtype=torch.float64)
                dist.all_reduce(t, op=dist.ReduceOp.MAX)
                per_iter.append(float(t.item()))
            measured = statistics.median(per_iter)
            results[seq_len] = measured
            if rank == 0:
                print(f"[bench] seq={seq_len} measured median={measured:.2f} ms "
                      f"(p50/p90/max={statistics.median(per_iter):.2f}/"
                      f"{sorted(per_iter)[int(0.9*len(per_iter))-1]:.2f}/"
                      f"{max(per_iter):.2f})", flush=True)
        except Exception as e:
            if rank == 0:
                print(f"[bench] seq={seq_len} FAILED: {e}", flush=True)
            results[seq_len] = None

    if rank == 0 and args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({"measured_per_layer_ms": results,
                       "num_seqs": args.num_seqs,
                       "world": world}, f, indent=2)
        print(f"[bench] wrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
