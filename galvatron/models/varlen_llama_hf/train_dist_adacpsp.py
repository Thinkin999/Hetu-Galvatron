import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True  # 抑制 dynamo 编译错误，回退到 eager 模式

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
    # AdaCPSP Setup: ensure model is constructed with all
    # attention modules (Ulysses dist_attn + Ring zigzag_ring)
    # ═══════════════════════════════════════════════════════
    adacpsp_optimizer = None
    group_manager = None

    if args.use_adaCPSP:
        # Force tp_deg >= 2 (→ sp >= 2 when use_ulysses) and cp_deg >= 2
        # during model construction so that both DistributedAttention
        # and ZigzagRingFlashAttention modules are created
        original_tp = args.global_tp_deg
        original_cp = args.global_cp_deg

        if args.global_tp_deg < 2 or args.global_cp_deg < 2:
            args.global_tp_deg = max(args.global_tp_deg, 2)
            args.global_cp_deg = max(args.global_cp_deg, 2)
            while args.global_tp_deg * args.global_cp_deg > world_size:
                if args.global_cp_deg > 2:
                    args.global_cp_deg //= 2
                elif args.global_tp_deg > 2:
                    args.global_tp_deg //= 2
                else:
                    break
            # vocab_tp/vocab_cp must match
            args.vocab_tp = args.global_tp_deg
            args.vocab_cp = args.global_cp_deg

            if rank == 0:
                print(f"[AdaCPSP] Overriding construction groups: "
                      f"tp={original_tp}→{args.global_tp_deg}, cp={original_cp}→{args.global_cp_deg}")

        # Enable both Ulysses and sequence_parallel for model construction
        args.use_ulysses = True
        args.sequence_parallel = True

    # Construct hybrid parallel model
    model = llama_model_hp(config, args)

    param_size_B = sum(p.numel() for p in model.parameters()) / 1e9
    if rank == 0:
        print(f"Model size: {param_size_B:.4f}B parameters")

    # ═══════════════════════════════════════════════════════
    # AdaCPSP: Create CommunicationGroupManager and Optimizer
    # ═══════════════════════════════════════════════════════
    if args.use_adaCPSP:
        from galvatron.models.varlen_llama_hf.adacpsp_group_manager import CommunicationGroupManager
        from galvatron.models.varlen_llama_hf.adacpsp_solver import AdaCPSPOptimizer, AdaCPSPCostModel

        # Create group manager with all possible strategies
        group_manager = CommunicationGroupManager(world_size)
        args.adacpsp_group_manager = group_manager

        # Create cost model and optimizer
        costmodel = AdaCPSPCostModel(
            cluster_size=world_size,
            hidden_size=config.hidden_size,
            layer_num=config.num_hidden_layers,
        )

        # Try to load profiling data
        profile_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiling_results")
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
                print("[AdaCPSP] Loaded profiling data for cost model")
        else:
            if rank == 0:
                print("[AdaCPSP] Using default cost model (no profiling data found)")

        # Determine memory limit (use 90% of GPU memory)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        memory_limit_gb = gpu_mem_gb * 0.9

        adacpsp_optimizer = AdaCPSPOptimizer(
            costmodel=costmodel,
            cluster_size=world_size,
            memory_limit_gb=memory_limit_gb,
            hide_output=(rank != 0),
            min_parallel_size=args.global_tp_deg,  # sp_size must always = tp_deg
        )

        if rank == 0:
            strategies = group_manager.get_all_strategies()
            solver_strategies = adacpsp_optimizer.get_strategy_pool()
            print(f"[AdaCPSP] Group manager strategies (sp,cp): {strategies}")
            print(f"[AdaCPSP] Solver strategies: {solver_strategies}")
            print(f"[AdaCPSP] Fixed sp_size (=tp_deg): {args.global_tp_deg}")
            print(f"[AdaCPSP] Memory limit: {memory_limit_gb:.1f} GB")

    optimizer, opt_param_scheduler = get_optimizer_and_param_scheduler(model, args)
    path = os.path.dirname(os.path.abspath(__file__))
    profiler = get_runtime_profiler(args, path, config, start_iter=0)
    profiler.profile_memory(0, "After creating model")

    # Create dataset and dataloader
    if local_rank == 0:
        print("Creating Dataset...")

    dataloader_group = model.dp_groups_whole[0].group

    trainloader = distributed_dataloader(
        dataset=DataLoaderForVarlenLlama(args, device),
        global_bsz=args.global_train_batch_size,
        shuffle=False,
        args=args,
        group=dataloader_group,
        adaCPSP_optimizer_=adacpsp_optimizer,
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
