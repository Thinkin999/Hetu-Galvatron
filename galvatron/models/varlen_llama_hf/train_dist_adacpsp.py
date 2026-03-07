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
    profiler = get_runtime_profiler(args, path, config, start_iter=0)
    profiler.profile_memory(0, "After creating model")

    # Create dataset and dataloader
    if local_rank == 0:
        print("Creating Dataset...")
    
    # For AdaCPSP: dataloader gives ALL ranks the same data
    # For non-AdaCPSP: use the dp group for distributed loading
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
        adaCPSP_optimizer_=adacpsp_optimizer,
        adaCPSP_forced_strategy_=forced_strategy,
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

            # Handle batch format
            if not args.use_packing:
                batch = [batch]
            
            # Forward and backward
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
