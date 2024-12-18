import torch
from torch import nn
from torch.optim import Adam
from flash_attn.models.gpt import GPTLMHeadModel
from tqdm import tqdm
import os
from galvatron.utils import set_seed, distributed_dataloader, print_loss
from galvatron.core import initialize_galvatron, GalvatronProfiler
from galvatron.models.llama_fa.LlamaModel_hybrid_parallel import get_hybrid_parallel_configs, construct_hybrid_parallel_model
from galvatron.models.llama_fa.dataloader import DataLoaderForLlama, test_collate_fn
from galvatron.models.llama_fa.meta_configs import config_from_meta, llama_config_to_gpt2_config, set_model_config, model_name, model_layer_configs
from galvatron.models.llama_fa.arguments import model_args


def train(args):
    local_rank = args.local_rank
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    world_size = torch.distributed.get_world_size()

    llama_config = config_from_meta(args.model_size)
    config = llama_config_to_gpt2_config(llama_config, args)
    config = set_model_config(config, args)
    
    if local_rank == 0:
        print(config)
    
    hybrid_parallel_configs = get_hybrid_parallel_configs(model_config=config, training_args=args)
    if local_rank == 0:
        print("Creating Model...")
    llama_model = GPTLMHeadModel(config, device='meta' if args.initialize_on_meta else 'cpu')
    model = construct_hybrid_parallel_model(
        model=llama_model, 
        model_config=config, 
        training_args=args, 
        hybrid_parallel_configs=hybrid_parallel_configs
    )
    
    # from galvatron.models.llama_fa import llama_model_hp
    # model = llama_model_hp(config, args)
    
    if local_rank == 0:
        print("Creating Dataset...")
    trainloader = distributed_dataloader(
        dataset=DataLoaderForLlama(args, device),
        global_bsz=args.global_train_batch_size,
        shuffle=True,
        args=args,
        group = model.dp_groups_whole[0].group,
        collate_fn = test_collate_fn
    )
    
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.adam_weight_decay)

    path = os.path.dirname(os.path.abspath(__file__))
    profiler = GalvatronProfiler(args)
    profiler.set_profiler_dist(path, model_layer_configs(config), model_name(config))
    
    profiler.profile_memory(0, "After creating model")
    if local_rank == 0:
        print("Start training...")
    
    if args.profile_forward:
        torch.set_grad_enabled(False)
        
    for ep in range(args.epochs):
        if not args.check_loss and not args.profile:
            trainloader = tqdm(trainloader)
        for iter, batch in enumerate(trainloader):
            tokens, kwargs, loss_func = batch
            profiler.profile_time_start(iter)
            profiler.profile_memory(iter, "Before Forward")

            input_ids = tokens
            batch = [input_ids]
            
            loss = model.forward_backward(batch, iter, profiler, 
                                      loss_func=loss_func,
                                      **kwargs)
            
            profiler.profile_memory(iter, "After Backward")
            
            optimizer.step()
            
            profiler.profile_memory(iter, "After optimizer_step")
            
            optimizer.zero_grad()
            
            print_loss(args, loss, ep, iter)

            profiler.post_profile_memory(iter)
            profiler.profile_time_end(iter)

            torch.distributed.barrier()

if __name__ == '__main__':
    args = initialize_galvatron(model_args, mode='train_dist')
    set_seed()
    train(args)