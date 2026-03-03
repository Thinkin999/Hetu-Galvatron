import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True  # 抑制 dynamo 编译错误，回退到 eager 模式

from torch import nn
import torch.distributed
from torch.optim import Adam
from transformers import LlamaConfig, LlamaForCausalLM
from tqdm import tqdm
import os
from galvatron.core import (
    RuntimeProfiler,
    clip_grad_norm,
    get_optimizer_and_param_scheduler,
    initialize_galvatron,
)
from galvatron.models.varlen_llama_hf.LlamaModel_hybrid_parallel import get_llama_config, get_runtime_profiler, llama_model_hp
from galvatron.models.varlen_llama_hf.meta_configs import model_layer_configs, model_name
from galvatron.utils import print_loss, set_seed, print_param_num
from galvatron.core import (
    RuntimeProfiler,
    construct_hybrid_parallel_model_api,
    get_hybrid_parallel_configs_api,
    init_empty_weights,
)
from megatron.training.arguments import _print_args

# Import LLaMA specific modules  
from galvatron.models.varlen_llama_hf.LlamaModel_hybrid_parallel import get_hybrid_parallel_configs, construct_hybrid_parallel_model
from galvatron.models.varlen_llama_hf.meta_configs import config_from_meta, set_model_config, model_name, model_layer_configs
from galvatron.models.varlen_llama_hf.arguments import model_args

# Import AdaCPSP modules
from galvatron.utils import distributed_dataloader, print_loss
#from galvatron.models.varlen_llama_hf.adacpsp_solver import AdaCPSPCostModel, AdaCPSPOptimizer
from galvatron.models.varlen_llama_hf.varlen_dataloder import DataLoaderForVarlenLlama


# Cost model parameters for different model sizes
ADACPSP_PARAM_DICT = {
    'llama-7b': {
        'act_per_token': 4.48,
        'cpt_alpha1': 5.128e-6,
        'cpt_alpha2': 183.9576e-3,
        'cpt_beta1': 629.3563,
        'hidden_size': 4096,
        'num_attention_heads': 32,
        'num_kv_heads': 32,
        'layer_num': 32,
    },
    'llama-13b': {
        'act_per_token': 4.22,
        'cpt_alpha1': 9.2852e-6,
        'cpt_alpha2': 306.0189e-3,
        'cpt_beta1': 1132.5632,
        'hidden_size': 5120,
        'num_attention_heads': 40,
        'num_kv_heads': 40,
        'layer_num': 40,
    },
    'llama-70b': {
        'act_per_token': 3.42,
        'cpt_alpha1': 15.4262e-6,
        'cpt_alpha2': 803.5742e-3,
        'cpt_beta1': 2789.3644,
        'hidden_size': 8192,
        'num_attention_heads': 64,
        'num_kv_heads': 8,  # GQA
        'layer_num': 80,
    },
}


def train(args):
    local_rank = args.local_rank
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    world_size = torch.distributed.get_world_size()
    max_len = args.seq_length
    
    # Get model configuration
    config = config_from_meta(args.model_size)
    config = set_model_config(config, args, True)  # 设置 True 以调用 overwrite_megatron_args，设置 padded_vocab_size
    config.max_position_embeddings = max_len
    args.seq_length = max_len
    
    if local_rank == 0:
        print(config)
        _print_args("arguments", args)
    
    # Get hybrid parallel configs
    hybrid_parallel_configs = get_hybrid_parallel_configs(model_config=config, training_args=args)
    
    # Create model
    if local_rank == 0:
        print("Creating Model...")
    
    if args.initialize_on_meta:
        with init_empty_weights(True):
            llama_model = LlamaForCausalLM(config)
    else:
        llama_model = LlamaForCausalLM(config)
    
    param_size_B = sum(p.numel() for p in llama_model.parameters()) / 1e9
    if rank == 0:
        print(f"Model size: {param_size_B:.4f}B parameters")
    
    memory_limit_gb = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024**3)
    exit(0)
    # Validation
    if args.use_packing:
        assert args.use_flash_attn, "packing is only supported by flash attention"
    
    # Initialize AdaCPSP optimizer if enabled
    adacpsp_optimizer = None
    if args.use_adaCPSP:
        assert args.use_ulysses or args.use_cp, "AdaCPSP requires use_ulysses or use_cp to be enabled"
        assert args.use_packing, "AdaCPSP requires use_packing"
        
        # Get cost model parameters
        model_params = ADACPSP_PARAM_DICT.get(args.model_size, ADACPSP_PARAM_DICT['llama-7b'])
        
        # Adjust act_per_token for specific configurations
        if args.model_size == 'llama-7b' and world_size == 16:
            model_params['act_per_token'] = 2.668764648
        
        # AlltoAll bandwidth dict
        alltoall_bandwidth_dict = {
            1: 1e10, 2: 119.54, 4: 104.07, 8: 96.5, 
            16: 10.33, 32: 5.94, 64: 4.87
        }
        
        # P2P bandwidth for CP (can be tuned based on cluster)
        p2p_bandwidth_gbs = getattr(args, 'p2p_bandwidth_gbs', 200)
        
        # Create cost model
        adacpsp_cost_model = AdaCPSPCostModel(
            cluster_size=world_size,
            hidden_size=model_params['hidden_size'],
            num_attention_heads=model_params['num_attention_heads'],
            num_kv_heads=model_params['num_kv_heads'],
            layer_num=model_params['layer_num'],
            param_size_B=param_size_B,
            zero_stage=3 if args.sdp == 1 else (3 if args.default_dp_type == "zero3" else 2),
            mixed_precision=True,
            act_per_token=model_params['act_per_token'],
            cpt_alpha1=model_params['cpt_alpha1'],
            cpt_alpha2=model_params['cpt_alpha2'],
            cpt_beta1=model_params['cpt_beta1'],
            alltoall_bandwidth_dict_gbs=alltoall_bandwidth_dict,
            p2p_bandwidth_gbs=p2p_bandwidth_gbs,
        )
        
        # Memory limit for optimizer
        memory_limit = getattr(args, 'memory_limit_gb', 28)
        
        # Create optimizer
        adacpsp_optimizer = AdaCPSPOptimizer(
            cluster_size=world_size,
            memory_limit_gb=memory_limit,
            costmodel=adacpsp_cost_model,
            max_sp_size=getattr(args, 'max_sp_size', None),
            max_cp_size=getattr(args, 'max_cp_size', None),
            hide_output=(rank != 0),
        )
        
        if rank == 0:
            print(f"AdaCPSP initialized: strategy={args.adacpsp_strategy}, memory_limit={memory_limit}GB")
    
    # Construct hybrid parallel model
    model = construct_hybrid_parallel_model(
        model=llama_model,
        model_config=config,
        training_args=args,
        hybrid_parallel_configs=hybrid_parallel_configs
    )
    optimizer, opt_param_scheduler = get_optimizer_and_param_scheduler(model, args)
    path = os.path.dirname(os.path.abspath(__file__))
    profiler = get_runtime_profiler(args, path, config, start_iter=0)
    profiler.profile_memory(0, "After creating model")
    # Create dataset and dataloader
    if local_rank == 0:
        print("Creating Dataset...")
    
    # Determine dataloader group
    if args.use_adaCPSP:
        # Each rank samples independently when using AdaCPSP
        dataloader_group = torch.distributed.new_group(ranks=[torch.distributed.get_rank()])
    else:
        dataloader_group = model.dp_groups_whole[0].group
    
    trainloader = distributed_dataloader(
        dataset=DataLoaderForVarlenLlama(args, device),
        global_bsz=args.global_train_batch_size,
        shuffle=False,
        args=args,
        group=dataloader_group,
        adaCPSP_optimizer_=None if not args.use_adaCPSP else adacpsp_optimizer,
    )
    
    # Create optimizer
    # optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.adam_weight_decay)
    # profiler = args.profiler
    
    if local_rank == 0:
        print("Start training...")
    
    # Training loop
    for ep in range(args.epochs):
        if not args.check_loss and not args.profile:
            trainloader = tqdm(trainloader) if rank == 0 else trainloader
        
        for iter, batch in enumerate(trainloader):
            # Skip first iteration (async solving warmup)
            if iter == 0:
                continue
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
