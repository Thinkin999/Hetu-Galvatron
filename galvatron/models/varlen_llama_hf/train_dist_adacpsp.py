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


def _adacpsp_solve_and_assign(batch, adacpsp_optimizer, forced_strategy,
                              args, rank, world_size, device):
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

    packed_tokens, cu_seqlens = batch
    num_seqs = cu_seqlens.shape[0] - 1

    # Reconstruct per-sequence lengths (needed by solver)
    seq_lens = [(cu_seqlens[i + 1] - cu_seqlens[i]).item() for i in range(num_seqs)]

    # ─── Rank 0 solves ───
    all_micro_res = None
    if rank == 0:
        seqs = [Sequence(seq=sl, id=i) for i, sl in enumerate(seq_lens)]

        if forced_strategy is not None:
            all_groups = _build_forced_groups(seqs, world_size, forced_strategy)
        else:
            all_groups, _ = adacpsp_optimizer.solve_globalbatch(seqs)

        if len(all_groups) == 0:
            print("[AdaCPSP] Solver failed, fallback to Ulysses×" + str(world_size))
            fallback = ParallelStrategy("ulysses", world_size)
            all_groups = [[(fallback, seqs)]]

        all_micro_res = []
        for micro_groups in all_groups:
            micro_res = []
            for strat, group_seqs in micro_groups:
                seq_ids = [s.id for s in group_seqs]
                micro_res.append((
                    strat.attn_type, strat.parallel_size,
                    strat.sp_size, strat.cp_size, seq_ids,
                ))
            all_micro_res.append(micro_res)

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
         my_attn_type, my_sp_size, my_cp_size) = convert_microbatch_res(micro_res)

        args.adacpsp_strategies.append({
            "sp_size": my_sp_size,
            "cp_size": my_cp_size,
            "attn_type": my_attn_type,
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
            print(f"  [AdaCPSP] MB{mb_idx}: type={strat['attn_type']}, "
                  f"sp={strat['sp_size']}, cp={strat['cp_size']}")

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
            # Find best profile with attention segments
            attn_json = None
            comm_json = None
            for pf in sorted(_glob.glob(os.path.join(configs_dir, "profile_validate_*.json")), reverse=True):
                try:
                    with open(pf) as _f:
                        _d = _json.load(_f)
                    if attn_json is None and "attention" in _d and "segments" in _d.get("attention", {}):
                        attn_json = pf
                    if comm_json is None and "communication" in _d and "linear_fits" in _d.get("communication", {}):
                        comm_json = pf
                except Exception:
                    pass
            
            if attn_json or comm_json:
                piecewise = None
                alltoall_linear = {}
                p2p_linear = {}
                
                for pf in [attn_json, comm_json]:
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
                    print(f"[AdaCPSP] Loaded profiling data: attn={attn_json}, comm={comm_json}")
        
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
            profiler.profile_time_start(iter)
            profiler.profile_memory(iter, "Before Forward")

            if not args.use_packing:
                batch = [batch]
            elif args.use_adaCPSP:
                batch = _adacpsp_solve_and_assign(
                    batch, adacpsp_optimizer, forced_strategy,
                    args, rank, world_size, device,
                )

            loss = model.forward_backward(batch, iter, profiler)
            profiler.profile_memory(iter, "After Backward")

            total_norm = clip_grad_norm(model, args.clip_grad)

            # Optimizer step
            optimizer.step()
            opt_param_scheduler.step(increment=args.global_batch_size)
            profiler.profile_memory(iter, "After optimizer_step")

            optimizer.zero_grad()

            profiler.post_profile_memory(iter)
            for param_group in optimizer.param_groups:
                learning_rate = param_group["lr"]
            profiler.profile_time_end(iter, loss, learning_rate, total_norm)

            if local_rank == 0:
                print_loss(args, loss, ep, iter)
            torch.distributed.barrier()    


if __name__ == '__main__':
    args = initialize_galvatron(model_args, mode='train_dist')
    set_seed()
    train(args)
