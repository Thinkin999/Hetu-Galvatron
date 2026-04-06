"""
AdaCPSP Training Script
=======================

Key design choices (differs from the old tp>=2 approach):
  - tp_deg = 1 (no tensor parallelism, every GPU has full weights)
  - FSDP covers ALL GPUs (dp = world_size)
  - sp_size and cp_size are BOTH dynamic, determined by solver per microbatch
  - Each microbatch can have HETEROGENEOUS groups with different attn_types
  - force_all_modules ensures all attention modules (Flash, Ulysses, Ring) are created
  - Dataloader gives ALL ranks the SAME full global batch;
    convert_microbatch_res distributes sequences to rank groups.
"""

import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True

import torch.distributed
from transformers import LlamaForCausalLM
from tqdm import tqdm
import os
import multiprocessing as mp
import time as time_module

from galvatron.core import (
    RuntimeProfiler,
    clip_grad_norm,
    get_optimizer_and_param_scheduler,
    initialize_galvatron,
    init_empty_weights,
)
from megatron.training.arguments import _print_args

from galvatron.models.varlen_llama_hf.LlamaModel_hybrid_parallel import (
    get_llama_config, get_runtime_profiler, llama_model_hp,
    get_hybrid_parallel_configs, construct_hybrid_parallel_model,
)
from galvatron.models.varlen_llama_hf.meta_configs import config_from_meta, set_model_config
from galvatron.models.varlen_llama_hf.arguments import model_args
from galvatron.models.varlen_llama_hf.varlen_dataloder import DataLoaderForVarlenLlama
from galvatron.utils import distributed_dataloader, print_loss, set_seed


def _parse_forced_strategy(strategy_str):
    """
    Parse forced strategy string into list of tuples.

    Formats:
      "ulysses:4,ring:4"     → [("ulysses", 4), ("ring", 4)]
      "usp:2x4"              → [("usp", 8, 2, 4)]   (sp_size=2, cp_size=4, total=8)
      "ulysses:4,usp:2x4"   → [("ulysses", 4), ("usp", 8, 2, 4)]
    """
    groups = []
    for part in strategy_str.split(","):
        part = part.strip()
        attn_type, size_str = part.split(":")
        attn_type = attn_type.strip()
        size_str = size_str.strip()
        if attn_type == "usp" and "x" in size_str:
            sp_str, cp_str = size_str.split("x")
            sp_size = int(sp_str)
            cp_size = int(cp_str)
            groups.append((attn_type, sp_size * cp_size, sp_size, cp_size))
        else:
            groups.append((attn_type, int(size_str)))
    return groups


def _build_forced_groups(seqs, world_size, forced_config):
    """Build forced heterogeneous groups for testing.

    forced_config: list of (attn_type, parallel_size) or
                          (attn_type, parallel_size, sp_size, cp_size).
    If a single entry doesn't cover all GPUs it is auto-replicated.
    """
    from galvatron.models.varlen_llama_hf.adacpsp_solver import ParallelStrategy

    normalised = []
    for entry in forced_config:
        if len(entry) == 2:
            attn_type, ps = entry
            if attn_type == "ulysses":
                normalised.append((attn_type, ps, ps, 1))
            elif attn_type == "ring":
                normalised.append((attn_type, ps, 1, ps))
            else:
                raise ValueError("USP requires 4-tuple")
        elif len(entry) == 4:
            normalised.append(tuple(entry))
        else:
            raise ValueError(f"Unexpected forced_config entry: {entry}")

    total_ps = sum(ps for _, ps, _, _ in normalised)
    if total_ps < world_size and len(normalised) == 1:
        at, ps, sp, cp = normalised[0]
        assert world_size % ps == 0
        normalised = [(at, ps, sp, cp)] * (world_size // ps)
        total_ps = sum(ps for _, ps, _, _ in normalised)
    assert total_ps == world_size

    group_seqs = [[] for _ in range(len(normalised))]
    for i, seq in enumerate(seqs):
        group_seqs[i % len(normalised)].append(seq)

    groups = []
    for (attn_type, parallel_size, sp_size, cp_size), g_seqs in zip(normalised, group_seqs):
        groups.append((
            ParallelStrategy(attn_type=attn_type, parallel_size=parallel_size,
                             sp_size=sp_size, cp_size=cp_size),
            g_seqs,
        ))
    return [groups]


def _unpack_packed_batch(batch):
    """Unpack [packed_tokens, cu_seqlens, optional_batch_meta]."""
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise ValueError(f"Unexpected packed batch format: {type(batch)!r}")
    packed_tokens = batch[0]
    cu_seqlens = batch[1]
    batch_meta = batch[2] if len(batch) > 2 else None
    return packed_tokens, cu_seqlens, batch_meta


def _build_step_context(loader_iter, use_async):
    if use_async:
        if loader_iter == 0:
            return {
                "loader_iter": 0,
                "train_step_id": None,
                "training_batch_id": None,
                "solver_batch_id": 0,
            }
        return {
            "loader_iter": loader_iter,
            "train_step_id": loader_iter,
            "training_batch_id": loader_iter - 1,
            "solver_batch_id": loader_iter,
        }

    return {
        "loader_iter": loader_iter,
        "train_step_id": loader_iter + 1,
        "training_batch_id": loader_iter,
        "solver_batch_id": loader_iter,
    }


def _format_adacpsp_prefix(step_ctx, phase, rank_scope=None):
    parts = ["[AdaCPSP]", f"[phase={phase}]"]
    loader_iter = step_ctx.get("loader_iter")
    if loader_iter is not None:
        parts.append(f"[loader_iter={loader_iter}]")

    train_step_id = step_ctx.get("train_step_id")
    if train_step_id is None:
        parts.append("[train_step=warmup]")
    else:
        parts.append(f"[train_step={train_step_id}]")

    training_batch_id = step_ctx.get("training_batch_id")
    if training_batch_id is not None:
        parts.append(f"[training_batch={training_batch_id}]")

    solver_batch_id = step_ctx.get("solver_batch_id")
    if solver_batch_id is not None:
        parts.append(f"[solver_batch={solver_batch_id}]")

    if rank_scope is not None:
        parts.append(f"[rank_scope={rank_scope}]")

    return "".join(parts)


def _sequence_lengths_from_cu(cu_seqlens):
    num_seqs = cu_seqlens.shape[0] - 1
    return [(cu_seqlens[i + 1] - cu_seqlens[i]).item() for i in range(num_seqs)]


def _normalize_micro_res_tuple(res_tuple):
    if len(res_tuple) == 5:
        attn_type, parallel_size, sp_size, cp_size, seq_ids = res_tuple
        return attn_type, parallel_size, sp_size, cp_size, "context_first", seq_ids
    return res_tuple


def _log_global_batch_layout(prefix, cu_seqlens, batch_meta):
    seq_lens = _sequence_lengths_from_cu(cu_seqlens)
    print(f"{prefix} Global batch summary: num_seqs={len(seq_lens)}, total_tokens={sum(seq_lens)}")
    for batch_seq_id, seq_len in enumerate(seq_lens):
        start = cu_seqlens[batch_seq_id].item()
        end = cu_seqlens[batch_seq_id + 1].item()
        if batch_meta is None:
            print(
                f"{prefix}   batch_seq={batch_seq_id}, sample_id={batch_seq_id}, "
                f"packed_len={seq_len}, packed_span=({start}, {end})"
            )
            continue

        meta = batch_meta[batch_seq_id]
        print(
            f"{prefix}   batch_seq={batch_seq_id}, sample_id={meta['sample_id']}, "
            f"raw_len={meta['raw_length']}, padded_len={meta['padded_length']}, "
            f"packed_len={seq_len}, packed_span=({start}, {end})"
        )


def _describe_microbatch_groups(micro_res, cu_seqlens, batch_meta, world_size):
    group_descs = []
    rank_cursor = 0

    for group_idx, res_tuple in enumerate(micro_res):
        attn_type, parallel_size, sp_size, cp_size, placement, seq_ids = _normalize_micro_res_tuple(res_tuple)
        ranks = list(range(rank_cursor, rank_cursor + parallel_size))
        seq_lens = []
        sample_ids = []
        raw_lengths = []
        padded_lengths = []
        source_spans = []
        offsets = [0]

        for sid in seq_ids:
            start = cu_seqlens[sid].item()
            end = cu_seqlens[sid + 1].item()
            seq_len = end - start
            seq_lens.append(seq_len)
            source_spans.append((start, end))
            offsets.append(offsets[-1] + seq_len)

            if batch_meta is not None:
                sample_ids.append(batch_meta[sid]["sample_id"])
                raw_lengths.append(batch_meta[sid]["raw_length"])
                padded_lengths.append(batch_meta[sid]["padded_length"])
            else:
                sample_ids.append(sid)
                raw_lengths.append(seq_len)
                padded_lengths.append(seq_len)

        group_descs.append({
            "group_idx": group_idx,
            "ranks": ranks,
            "attn_type": attn_type,
            "sp_size": sp_size,
            "cp_size": cp_size,
            "placement": placement,
            "batch_seq_ids": list(seq_ids),
            "sample_ids": sample_ids,
            "raw_lengths": raw_lengths,
            "padded_lengths": padded_lengths,
            "seqlens": seq_lens,
            "source_spans": source_spans,
            "mb_cu": offsets,
        })
        rank_cursor += parallel_size

    while rank_cursor < world_size:
        group_descs.append({
            "group_idx": len(group_descs),
            "ranks": [rank_cursor],
            "attn_type": "idle",
            "sp_size": 1,
            "cp_size": 1,
            "placement": "context_first",
            "batch_seq_ids": [],
            "sample_ids": [],
            "raw_lengths": [],
            "padded_lengths": [],
            "seqlens": [],
            "source_spans": [],
            "mb_cu": [0],
        })
        rank_cursor += 1

    return group_descs


def _log_microbatch_layout(prefix, mb_idx, group_descs):
    total_tokens = sum(sum(group["seqlens"]) for group in group_descs)
    print(f"{prefix} MB{mb_idx} summary: groups={len(group_descs)}, total_tokens={total_tokens}")
    for group in group_descs:
        placement_suffix = f", placement={group['placement']}" if group["attn_type"] == "usp" else ""
        print(
            f"{prefix}   MB{mb_idx}/Group{group['group_idx']}: ranks={group['ranks']}, "
            f"attn={group['attn_type']}, sp={group['sp_size']}, cp={group['cp_size']}{placement_suffix}"
        )
        print(
            f"{prefix}   MB{mb_idx}/Group{group['group_idx']}: "
            f"batch_seq_ids={group['batch_seq_ids']}, sample_ids={group['sample_ids']}"
        )
        print(
            f"{prefix}   MB{mb_idx}/Group{group['group_idx']}: "
            f"raw_lengths={group['raw_lengths']}, padded_lengths={group['padded_lengths']}, "
            f"seqlens={group['seqlens']}"
        )
        print(
            f"{prefix}   MB{mb_idx}/Group{group['group_idx']}: "
            f"source_spans={group['source_spans']}, mb_cu={group['mb_cu']}"
        )


def _get_adacpsp_solver_config(args):
    """Resolve runtime solver options from training args."""
    return {
        "method": getattr(args, "adaCPSP_method", "adaptive_bfd"),
        "solve_mode": getattr(args, "adaCPSP_solve_mode", "sequential"),
        "bucket_num": getattr(args, "adaCPSP_bucket_num", 16),
        "mb_option_num": getattr(args, "adaCPSP_mb_option_num", 5),
        "chunk_alg": getattr(args, "chunk_alg", "sort_consec"),
    }


def _solve_global_batch_with_config(optimizer, seqs, solver_cfg, log_context=None):
    """Dispatch to the selected global-batch solving mode."""
    method = solver_cfg["method"]
    solve_mode = solver_cfg["solve_mode"]
    bucket_num = solver_cfg["bucket_num"]
    mb_option_num = solver_cfg["mb_option_num"]
    chunk_alg = solver_cfg["chunk_alg"]

    if solve_mode == "mp_gbmb":
        return optimizer.solve_globalbatch_mp_gbmb(
            seqs,
            chunk_alg=chunk_alg,
            method=method,
            bucket_num=bucket_num,
            mb_option_num=mb_option_num,
        )
    if solve_mode == "mp":
        return optimizer.solve_globalbatch_mp(
            seqs,
            chunk_alg=chunk_alg,
            method=method,
            bucket_num=bucket_num,
        )
    return optimizer.solve_globalbatch(
        seqs,
        chunk_alg=chunk_alg,
        method=method,
        bucket_num=bucket_num,
        log_context=log_context,
    )


def _adacpsp_solve_and_assign(batch, adacpsp_optimizer, forced_strategy,
                              args, rank, world_size, device, step_ctx):
    """
    Rank 0 runs the solver, broadcasts the result, then ALL ranks
    collectively create communication groups and build per-group microbatches.

    Args:
        batch: [packed_tokens, cu_seqlens] from DataLoader collate_fn
    Returns:
        microbatches list expected by forward_backward:
          [[[tokens_mb0, cu_mb0]], [[tokens_mb1, cu_mb1]], ...]
    """
    from galvatron.models.varlen_llama_hf.adacpsp_solver import (
        Sequence, ParallelStrategy,
    )
    from galvatron.models.varlen_llama_hf.adacpsp_group_manager import convert_microbatch_res

    packed_tokens, cu_seqlens, batch_meta = _unpack_packed_batch(batch)
    seq_lens = _sequence_lengths_from_cu(cu_seqlens)
    solve_prefix = _format_adacpsp_prefix(step_ctx, "solve", "rank0")
    dispatch_prefix = _format_adacpsp_prefix(step_ctx, "dispatch", "rank0")

    # ─── Rank 0 solves ───
    all_micro_res = None
    if rank == 0:
        _log_global_batch_layout(solve_prefix, cu_seqlens, batch_meta)
        seqs = [Sequence(seq=sl, id=i) for i, sl in enumerate(seq_lens)]
        solver_cfg = _get_adacpsp_solver_config(args)

        if forced_strategy is not None:
            all_groups = _build_forced_groups(seqs, world_size, forced_strategy)
        else:
            all_groups, _ = _solve_global_batch_with_config(
                adacpsp_optimizer, seqs, solver_cfg, log_context=solve_prefix
            )

        if len(all_groups) == 0:
            print(f"{solve_prefix} Solver failed, fallback to Ulysses×{world_size}")
            fallback = ParallelStrategy("ulysses", world_size)
            all_groups = [[(fallback, seqs)]]

        all_micro_res = _groups_to_micro_res(all_groups)
        for mb_idx, micro_res in enumerate(all_micro_res):
            group_descs = _describe_microbatch_groups(micro_res, cu_seqlens, batch_meta, world_size)
            _log_microbatch_layout(dispatch_prefix, mb_idx, group_descs)

    # ─── Broadcast solver result to all ranks ───
    bcast_buf = [all_micro_res]
    torch.distributed.broadcast_object_list(bcast_buf, src=0)
    all_micro_res = bcast_buf[0]

    # ─── All ranks collectively create groups & build microbatches ───
    args.adacpsp_strategies = []
    args.adacpsp_sp_groups = []
    args.adacpsp_cp_groups = []

    microbatches = []
    for mb_idx, micro_res in enumerate(all_micro_res):
        (my_seq_ids, my_sp_group, my_cp_group,
         my_attn_type, my_sp_size, my_cp_size, my_placement) = convert_microbatch_res(micro_res)

        args.adacpsp_strategies.append({
            "sp_size": my_sp_size,
            "cp_size": my_cp_size,
            "attn_type": my_attn_type,
            "placement": my_placement,
        })
        args.adacpsp_sp_groups.append(my_sp_group)
        args.adacpsp_cp_groups.append(my_cp_group)

        if len(my_seq_ids) == 0:
            mb_tokens = torch.zeros(1, dtype=torch.long, device=device)
            mb_cu = torch.zeros(2, dtype=torch.int64, device=device)
            mb_cu[1] = 1
        else:
            parts = []
            offsets = [0]
            for sid in my_seq_ids:
                start = cu_seqlens[sid].item()
                end = cu_seqlens[sid + 1].item()
                parts.append(packed_tokens[start:end])
                offsets.append(offsets[-1] + (end - start))
            mb_tokens = torch.cat(parts)
            mb_cu = torch.tensor(offsets, dtype=torch.int64, device=device)

        microbatches.append([[mb_tokens, mb_cu]])

    if rank == 0:
        for mb_idx, strat in enumerate(args.adacpsp_strategies):
            pl_str = f", placement={strat['placement']}" if strat['attn_type'] == 'usp' else ""
            mb_tokens, mb_cu = microbatches[mb_idx][0]
            print(
                f"{dispatch_prefix} Local rank assignment MB{mb_idx}: "
                f"type={strat['attn_type']}, sp={strat['sp_size']}, cp={strat['cp_size']}{pl_str}, "
                f"tokens={int(mb_tokens.numel())}, mb_cu={mb_cu.tolist()}"
            )

    return microbatches


# ═══════════════════════════════════════════════════════════════════════════════
# Async Solver: double-buffered overlap of solver (CPU) with training (GPU)
#
# Timeline:
#   iter 0 (warmup): launch solver(B0) in subprocess, buffer B0, skip training
#   iter N (N>=1):   join solver(B_{N-1}) → broadcast → launch solver(B_N)
#                    → build microbatches from B_{N-1} → train B_{N-1}
#                    (solver(B_N) runs on CPU in parallel with GPU training)
#
# Cost: 1 warmup iteration (no training). Across epoch boundaries the buffered
# batch carries over, so no data is lost except the very last batch of the
# final epoch.
# ═══════════════════════════════════════════════════════════════════════════════

# Module-level reference so the forked subprocess can access the optimizer
# without pickling (Linux fork inherits parent memory).
_async_optimizer = None


def _groups_to_micro_res(all_groups):
    """Convert solver group objects to pickle-safe list-of-list-of-tuples."""
    all_micro_res = []
    for micro_groups in all_groups:
        micro_res = []
        for strat, group_seqs in micro_groups:
            seq_ids = [s.id for s in group_seqs]
            micro_res.append((
                strat.attn_type, strat.parallel_size,
                strat.sp_size, strat.cp_size,
                strat.placement, seq_ids,
            ))
        all_micro_res.append(micro_res)
    return all_micro_res


def _async_solver_worker(
    seq_lens, result_queue, forced_strategy, world_size, solver_cfg, log_context=None
):
    """
    Solver subprocess entry point (runs on CPU only).
    Uses fork-inherited module-level _async_optimizer.
    """
    from galvatron.models.varlen_llama_hf.adacpsp_solver import (
        Sequence, ParallelStrategy,
    )

    start = time_module.time()
    seqs = [Sequence(seq=sl, id=i) for i, sl in enumerate(seq_lens)]

    if forced_strategy is not None:
        all_groups = _build_forced_groups(seqs, world_size, forced_strategy)
    else:
        all_groups, _ = _solve_global_batch_with_config(
            _async_optimizer, seqs, solver_cfg, log_context=log_context
        )

    if len(all_groups) == 0:
        fallback = ParallelStrategy("ulysses", world_size)
        all_groups = [[(fallback, seqs)]]

    result_queue.put(_groups_to_micro_res(all_groups))
    elapsed = time_module.time() - start
    if log_context is None:
        print(f"[AdaCPSP] Async solver completed in {elapsed:.3f}s")
    else:
        print(f"{log_context} Async solver completed in {elapsed:.3f}s")


class _AsyncSolverState:
    """
    Double-buffer state machine for overlapping AdaCPSP solver with training.

    The solver runs in a forked subprocess on CPU while the GPU executes the
    previous iteration's forward/backward pass.  All distributed collective
    operations (broadcast, new_group) happen in the main process so every rank
    stays synchronised.
    """

    def __init__(self, optimizer, forced_strategy, solver_cfg, world_size, rank):
        self._optimizer = optimizer
        self._forced_strategy = forced_strategy
        self._solver_cfg = solver_cfg
        self._world_size = world_size
        self._rank = rank
        self._process = None
        self._result_queue = None
        self._prev_batch = None
        self._buffered_step_ctx = None
        self._is_first = True

    @property
    def is_warmup(self):
        return self._is_first

    # ── public API ────────────────────────────────────────────────────────

    def warmup(self, batch, step_ctx):
        """Iter 0: launch solver for this batch, buffer data, no training."""
        self._launch_solver(batch, step_ctx)
        self._prev_batch = batch
        self._buffered_step_ctx = step_ctx
        self._is_first = False
        if self._rank == 0:
            warmup_prefix = _format_adacpsp_prefix(step_ctx, "warmup", "rank0")
            print(f"{warmup_prefix} Solver launched, training skipped")

    def step(self, current_batch, args, device, step_ctx):
        """
        Iter >= 1.  Returns microbatches built from *prev_batch*.
        Meanwhile solver(current_batch) starts running in background.
        """
        all_micro_res = self._collect_result(self._buffered_step_ctx)
        self._launch_solver(current_batch, step_ctx)
        microbatches = self._build_microbatches(all_micro_res, args, device, step_ctx)
        self._prev_batch = current_batch
        self._buffered_step_ctx = step_ctx
        return microbatches

    def cleanup(self):
        """Join any lingering subprocess at the very end of training."""
        if self._process is not None and self._process.is_alive():
            self._process.join(timeout=10)
            if self._process.is_alive():
                self._process.kill()
                self._process.join()
            self._process = None

    # ── private helpers ───────────────────────────────────────────────────

    def _launch_solver(self, batch, step_ctx):
        """Extract seq_lens and fork solver subprocess (rank 0 only)."""
        _, cu_seqlens, batch_meta = _unpack_packed_batch(batch)
        seq_lens = _sequence_lengths_from_cu(cu_seqlens)

        if self._rank == 0:
            solve_prefix = _format_adacpsp_prefix(step_ctx, "solve", "rank0")
            _log_global_batch_layout(solve_prefix, cu_seqlens, batch_meta)
            self._result_queue = mp.Queue(maxsize=1)
            self._process = mp.Process(
                target=_async_solver_worker,
                args=(seq_lens, self._result_queue,
                      self._forced_strategy, self._world_size, self._solver_cfg, solve_prefix),
            )
            self._process.start()

    def _collect_result(self, buffered_step_ctx):
        """Join solver subprocess (rank 0), broadcast result to all ranks."""
        all_micro_res = None
        if self._rank == 0:
            collect_prefix = _format_adacpsp_prefix(buffered_step_ctx, "collect", "rank0")
            self._process.join(timeout=600)
            if self._process.is_alive():
                print(f"{collect_prefix} WARNING: solver timed out (600s), killing")
                self._process.kill()
                self._process.join()

            if self._process.exitcode != 0:
                print(f"{collect_prefix} WARNING: solver exited with code "
                      f"{self._process.exitcode}, falling back to sync")
                all_micro_res = self._sync_fallback(buffered_step_ctx)
            else:
                try:
                    all_micro_res = self._result_queue.get_nowait()
                except Exception:
                    print(f"{collect_prefix} WARNING: result queue empty, falling back to sync")
                    all_micro_res = self._sync_fallback(buffered_step_ctx)

        bcast_buf = [all_micro_res]
        torch.distributed.broadcast_object_list(bcast_buf, src=0)
        return bcast_buf[0]

    def _sync_fallback(self, buffered_step_ctx):
        """Synchronous solve on rank 0 when the async subprocess fails."""
        from galvatron.models.varlen_llama_hf.adacpsp_solver import (
            Sequence, ParallelStrategy,
        )
        _, cu_seqlens, _ = _unpack_packed_batch(self._prev_batch)
        seq_lens = _sequence_lengths_from_cu(cu_seqlens)
        seqs = [Sequence(seq=sl, id=i) for i, sl in enumerate(seq_lens)]
        solve_prefix = _format_adacpsp_prefix(buffered_step_ctx, "solve", "rank0")

        if self._forced_strategy is not None:
            all_groups = _build_forced_groups(
                seqs, self._world_size, self._forced_strategy)
        else:
            all_groups, _ = _solve_global_batch_with_config(
                self._optimizer, seqs, self._solver_cfg, log_context=solve_prefix
            )

        if len(all_groups) == 0:
            fallback = ParallelStrategy("ulysses", self._world_size)
            all_groups = [[(fallback, seqs)]]

        return _groups_to_micro_res(all_groups)

    def _build_microbatches(self, all_micro_res, args, device, step_ctx):
        """Build microbatch tensors from prev_batch + solver result."""
        from galvatron.models.varlen_llama_hf.adacpsp_group_manager import (
            convert_microbatch_res,
        )

        packed_tokens, cu_seqlens, batch_meta = _unpack_packed_batch(self._prev_batch)
        dispatch_prefix = _format_adacpsp_prefix(step_ctx, "dispatch", "rank0")

        args.adacpsp_strategies = []
        args.adacpsp_sp_groups = []
        args.adacpsp_cp_groups = []

        microbatches = []
        for mb_idx, micro_res in enumerate(all_micro_res):
            if self._rank == 0:
                group_descs = _describe_microbatch_groups(micro_res, cu_seqlens, batch_meta, self._world_size)
                _log_microbatch_layout(dispatch_prefix, mb_idx, group_descs)

            (my_seq_ids, my_sp_group, my_cp_group,
             my_attn_type, my_sp_size, my_cp_size,
             my_placement) = convert_microbatch_res(micro_res)

            args.adacpsp_strategies.append({
                "sp_size": my_sp_size,
                "cp_size": my_cp_size,
                "attn_type": my_attn_type,
                "placement": my_placement,
            })
            args.adacpsp_sp_groups.append(my_sp_group)
            args.adacpsp_cp_groups.append(my_cp_group)

            if len(my_seq_ids) == 0:
                mb_tokens = torch.zeros(1, dtype=torch.long, device=device)
                mb_cu = torch.zeros(2, dtype=torch.int64, device=device)
                mb_cu[1] = 1
            else:
                parts = []
                offsets = [0]
                for sid in my_seq_ids:
                    start = cu_seqlens[sid].item()
                    end = cu_seqlens[sid + 1].item()
                    parts.append(packed_tokens[start:end])
                    offsets.append(offsets[-1] + (end - start))
                mb_tokens = torch.cat(parts)
                mb_cu = torch.tensor(offsets, dtype=torch.int64, device=device)

            microbatches.append([[mb_tokens, mb_cu]])

        if self._rank == 0:
            for mb_idx, strat in enumerate(args.adacpsp_strategies):
                pl_str = (f", placement={strat['placement']}"
                          if strat['attn_type'] == 'usp' else "")
                mb_tokens, mb_cu = microbatches[mb_idx][0]
                print(
                    f"{dispatch_prefix} Local rank assignment MB{mb_idx}: "
                    f"type={strat['attn_type']}, sp={strat['sp_size']}, cp={strat['cp_size']}{pl_str}, "
                    f"tokens={int(mb_tokens.numel())}, mb_cu={mb_cu.tolist()}"
                )

        return microbatches


def train(args):
    local_rank = args.local_rank
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    world_size = torch.distributed.get_world_size()
    max_len = args.seq_length
    
    # Get model configuration
    config = config_from_meta(args.model_size)
    config = set_model_config(config, args, True)
    config.max_position_embeddings = max_len
    args.seq_length = max_len
    
    if local_rank == 0:
        print(config)
        _print_args("arguments", args)
    
    # Validation
    if args.use_packing:
        assert args.use_flash_attn, "packing is only supported by flash attention"
    
    # ═══════════════════════════════════════════════════════
    # AdaCPSP Setup
    # ═══════════════════════════════════════════════════════
    adacpsp_optimizer = None

    if args.use_adaCPSP:
        # tp=1: no tensor parallelism, full weights on every GPU
        # sp=1, cp=1 during construction: no SP/CP groups initially
        # dp=world_size: FSDP covers all GPUs
        # force_all_modules (via args.use_adaCPSP) ensures all attention
        # modules are created in Attention.__init__
        args.global_tp_deg = 1
        args.global_cp_deg = 1
        # Don't enable Ulysses/SP during construction;
        # they'll be enabled dynamically per microbatch
        args.use_ulysses = False
        args.sequence_parallel = False
        # AdaCPSP always uses packing (varlen sequences)
        args.use_packing = True

        if rank == 0:
            print(f"[AdaCPSP] Model construction: tp=1, sp=1, cp=1, dp={world_size}")
            print(f"[AdaCPSP] force_all_attn_modules=True (from args.use_adaCPSP)")

    # Construct hybrid parallel model
    model = llama_model_hp(config, args)

    torch.distributed.barrier()
    if rank == 0:
        print("[SYNC] All ranks finished model construction")

    param_size_B = sum(p.numel() for p in model.parameters()) / 1e9
    if rank == 0:
        print(f"Model size: {param_size_B:.4f}B parameters")

    # ═══════════════════════════════════════════════════════
    # AdaCPSP: Create Optimizer (no CommunicationGroupManager needed;
    # convert_microbatch_res creates groups lazily)
    # ═══════════════════════════════════════════════════════
    if args.use_adaCPSP:
        from galvatron.models.varlen_llama_hf.adacpsp_solver import AdaCPSPOptimizer, AdaCPSPCostModel
        
        # Create cost model — try to load from profiling data
        script_dir = os.path.dirname(os.path.abspath(__file__))
        configs_dir = os.path.join(script_dir, "configs")
        
        costmodel = None
        
        # Try unified profile JSON from configs dir (newest first)
        if os.path.isdir(configs_dir):
            import glob as _glob, json as _json
            # Find best profile with attention segments / unified comm profile / validation data
            attn_json = None
            legacy_comm_json = None
            comm_profile_json = None
            validation_json = None
            for pf in sorted(_glob.glob(os.path.join(configs_dir, "comm_profile_*.json")), reverse=True):
                try:
                    with open(pf) as _f:
                        _d = _json.load(_f)
                    if "alltoall" in _d and "p2p_ring" in _d:
                        comm_profile_json = pf
                        break
                except Exception:
                    pass
            for pf in sorted(_glob.glob(os.path.join(configs_dir, "profile_validate_*.json")), reverse=True):
                try:
                    with open(pf) as _f:
                        _d = _json.load(_f)
                    if attn_json is None and "attention" in _d and "segments" in _d.get("attention", {}):
                        attn_json = pf
                    if validation_json is None and "comm_validation" in _d:
                        validation_json = pf
                    if legacy_comm_json is None and "communication" in _d and "linear_fits" in _d.get("communication", {}):
                        legacy_comm_json = pf
                except Exception:
                    pass

            if attn_json and comm_profile_json:
                costmodel = AdaCPSPCostModel.from_attention_and_comm_profiles(
                    attention_json=attn_json,
                    comm_profile_json=comm_profile_json,
                    cluster_size=world_size,
                    validation_json=validation_json,
                    gpus_per_node=torch.cuda.device_count(),
                )
                if rank == 0:
                    print(
                        f"[AdaCPSP] Loaded topology-aware comm profile: "
                        f"attn={attn_json}, comm={comm_profile_json}, validation={validation_json}"
                    )

            elif attn_json or legacy_comm_json:
                piecewise = None
                alltoall_linear = {}
                p2p_linear = {}
                
                for pf in [attn_json, legacy_comm_json]:
                    if pf is None:
                        continue
                    with open(pf) as _f:
                        _d = _json.load(_f)
                    if piecewise is None and "attention" in _d and "segments" in _d.get("attention", {}):
                        piecewise = _d["attention"]["segments"]
                    if not alltoall_linear and "communication" in _d and "linear_fits" in _d.get("communication", {}):
                        for key, fit in _d["communication"]["linear_fits"].items():
                            gs = int(key.split("gs")[1])
                            entry = {"alpha": fit["alpha_ms_per_MB"], "beta": fit["beta_ms"]}
                            if key.startswith("alltoall"):
                                alltoall_linear[gs] = entry
                            elif key.startswith("p2p"):
                                p2p_linear[gs] = entry
                
                costmodel = AdaCPSPCostModel(
                    cluster_size=world_size,
                    hidden_size=config.hidden_size,
                    layer_num=config.num_hidden_layers,
                    piecewise_compute_coeffs=piecewise,
                    alltoall_linear_fit=alltoall_linear if alltoall_linear else None,
                    p2p_linear_fit=p2p_linear if p2p_linear else None,
                )
                if rank == 0:
                    print(f"[AdaCPSP] Loaded legacy profiling data: attn={attn_json}, comm={legacy_comm_json}")
        
        # Fallback: try legacy profiling files
        if costmodel is None:
            profile_dir = os.path.join(script_dir, "profiling_results")
            alltoall_file = os.path.join(profile_dir, "alltoall_bandwidth.json")
            p2p_file = os.path.join(profile_dir, "p2p_ring_bandwidth.json")
            attn_file = os.path.join(profile_dir, "attention_piecewise_fit.json")

            if os.path.exists(alltoall_file) and os.path.exists(p2p_file) and os.path.exists(attn_file):
                costmodel = AdaCPSPCostModel.from_profile_files(
                    attention_json=attn_file,
                    alltoall_json=alltoall_file,
                    p2p_json=p2p_file,
                    cluster_size=world_size,
                )
                if rank == 0:
                    print("[AdaCPSP] Loaded legacy profiling data for cost model")
        
        # Final fallback: default cost model
        if costmodel is None:
            costmodel = AdaCPSPCostModel(
            cluster_size=world_size,
                hidden_size=config.hidden_size,
                layer_num=config.num_hidden_layers,
            )
            if rank == 0:
                print("[AdaCPSP] Using default cost model (no profiling data found)")

        # Determine memory limit
        override_mem = getattr(args, 'memory_limit_gb', 0)
        if override_mem and override_mem > 0:
            memory_limit_gb = override_mem
        else:
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            memory_limit_gb = gpu_mem_gb * 0.9

        # Determine allowed attention types
        allowed_attn_types = getattr(args, 'adaCPSP_attn_types', ["ulysses", "ring", "usp"])

        adacpsp_optimizer = AdaCPSPOptimizer(
            costmodel=costmodel,
            cluster_size=world_size,
            memory_limit_gb=memory_limit_gb,
            hide_output=(rank != 0),
            allowed_attn_types=allowed_attn_types,
            # No min_parallel_size constraint now: tp=1, so any sp/cp size works
        )
        
        if rank == 0:
            solver_strategies = adacpsp_optimizer.get_strategy_pool()
            print(f"[AdaCPSP] Allowed attn types: {allowed_attn_types}")
            print(f"[AdaCPSP] Solver strategies: {solver_strategies}")
            print(f"[AdaCPSP] Memory limit: {memory_limit_gb:.1f} GB")
    
    optimizer, opt_param_scheduler = get_optimizer_and_param_scheduler(model, args)
    path = os.path.dirname(os.path.abspath(__file__))
    profiler = get_runtime_profiler(
        args,
        path,
        config,
        start_iter=getattr(args, "profile_start_iter", 0),
        end_iter=getattr(args, "profile_end_iter", 20),
    )
    if args.use_adaCPSP:
        profiler.set_time_log_rank(0)
        profiler.set_memory_profiler(
            rank,
            profiler.profile_ranks,
            max_profile_iter=max(1, getattr(args, "profile_end_iter", 20) - 1),
        )
        args.profile_data_batch_size = args.global_train_batch_size
        args.profile_scheduler_batch_size = args.global_batch_size
        if rank == 0:
            batch_prefix = "[AdaCPSP][phase=batch_size][rank_scope=rank0]"
            print(
                f"{batch_prefix} Batch size semantics: "
                f"data_batch_size={args.profile_data_batch_size}, "
                f"scheduler_batch_size={args.profile_scheduler_batch_size}"
            )
    profiler.profile_memory(0, "After creating model")

    # Create dataset and dataloader
    if local_rank == 0:
        print("Creating Dataset...")
    
    # For AdaCPSP: dataloader gives ALL ranks the same data
    # For non-AdaCPSP: use the dp group for distributed loading
    if args.use_adaCPSP:
        dataloader_group = None
    else:
        dataloader_group = model.dp_groups_whole[0].group
    
    # Parse forced strategy (for heterogeneous group testing)
    forced_strategy = None
    if hasattr(args, 'adaCPSP_forced_strategy') and args.adaCPSP_forced_strategy:
        forced_strategy = _parse_forced_strategy(args.adaCPSP_forced_strategy)
        if rank == 0:
            print(f"[AdaCPSP] Forced strategy: {forced_strategy}")

    # Placement override: "auto" (solver decides), "head_first", "context_first"
    force_placement = getattr(args, "force_placement", "auto")
    if force_placement != "auto" and adacpsp_optimizer is not None:
        adacpsp_optimizer.force_placement = force_placement
        if rank == 0:
            print(f"[AdaCPSP] Forced placement: {force_placement}")

    # ═══════════════════════════════════════════════════════
    # Async solver state (double-buffering)
    # ═══════════════════════════════════════════════════════
    async_state = None
    use_async = (args.use_adaCPSP
                 and not getattr(args, 'adaCPSP_sync_solver', False))
    if use_async:
        global _async_optimizer
        _async_optimizer = adacpsp_optimizer
        solver_cfg = _get_adacpsp_solver_config(args)
        async_state = _AsyncSolverState(
            optimizer=adacpsp_optimizer,
            forced_strategy=forced_strategy,
            solver_cfg=solver_cfg,
            world_size=world_size,
            rank=rank,
        )
        if rank == 0:
            print(
                "[AdaCPSP] Solver config: "
                f"method={solver_cfg['method']}, mode={solver_cfg['solve_mode']}, "
                f"chunk_alg={solver_cfg['chunk_alg']}, bucket_num={solver_cfg['bucket_num']}, "
                f"mb_option_num={solver_cfg['mb_option_num']}"
            )
            print("[AdaCPSP] Async solver enabled (double-buffering)")
    elif args.use_adaCPSP and rank == 0:
        solver_cfg = _get_adacpsp_solver_config(args)
        print(
            "[AdaCPSP] Solver config: "
            f"method={solver_cfg['method']}, mode={solver_cfg['solve_mode']}, "
            f"chunk_alg={solver_cfg['chunk_alg']}, bucket_num={solver_cfg['bucket_num']}, "
            f"mb_option_num={solver_cfg['mb_option_num']}"
        )
        print("[AdaCPSP] Sync solver mode (no overlap)")

    trainloader = distributed_dataloader(
        dataset=DataLoaderForVarlenLlama(args, device),
        global_bsz=args.global_train_batch_size,
        shuffle=False,
        args=args,
        group=dataloader_group,
    )

    if local_rank == 0:
        print("Start training...")

    # Training loop
    for ep in range(args.epochs):
        if not args.check_loss and not args.profile:
            trainloader = tqdm(trainloader) if rank == 0 else trainloader

        for iter, batch in enumerate(trainloader):
            step_ctx = _build_step_context(iter, use_async)

            # ── Async double-buffer: warmup iter (launch solver, skip train) ──
            if async_state is not None and async_state.is_warmup:
                async_state.warmup(batch, step_ctx)
                continue

            # ── Prepare microbatches ──
            if not args.use_packing:
                batch = [batch]
            elif args.use_adaCPSP:
                if async_state is not None:
                    batch = async_state.step(batch, args, device, step_ctx)
                else:
                    batch = _adacpsp_solve_and_assign(
                        batch, adacpsp_optimizer, forced_strategy,
                        args, rank, world_size, device, step_ctx,
                    )

            profiler.set_step_context(**step_ctx)
            profiler.profile_time_start(iter)
            profiler.profile_memory(iter, "Before Forward")

            loss = model.forward_backward(batch, iter, profiler)
            profiler.profile_memory(iter, "After Backward")

            total_norm = clip_grad_norm(model, args.clip_grad)

            optimizer.step()
            opt_param_scheduler.step(increment=args.global_batch_size)
            profiler.profile_memory(iter, "After optimizer_step")

            optimizer.zero_grad()

            profiler.post_profile_memory(iter)
            for param_group in optimizer.param_groups:
                learning_rate = param_group["lr"]
            profiler.profile_time_end(iter, loss, learning_rate, total_norm)

            if local_rank == 0:
                print_loss(args, loss, ep, step_ctx["train_step_id"] if step_ctx["train_step_id"] is not None else iter)
            torch.distributed.barrier()

    # Clean up lingering solver subprocess
    if async_state is not None:
        async_state.cleanup()


if __name__ == '__main__':
    args = initialize_galvatron(model_args, mode='train_dist')
    set_seed()
    train(args)
