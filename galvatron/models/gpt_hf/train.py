import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import GPT2Config, GPT2LMHeadModel
from dataloader import DataLoaderForGPT
from tqdm import tqdm
from galvatron.utils import set_seed, print_loss
from galvatron.core import initialize_galvatron, GalvatronProfiler
from galvatron.models.gpt_hf.meta_configs import config_from_meta, set_model_config
from galvatron.models.gpt_hf.arguments import model_args

def model_forward(model, input_ids):
    lm_logits = model(input_ids=input_ids).logits
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    from torch.nn import CrossEntropyLoss
    loss_fn = CrossEntropyLoss()
    loss = loss_fn(shift_logits.view(-1, model.config.vocab_size), shift_labels.view(-1).long())
    return loss

def train(args):
    cuda_condition = torch.cuda.is_available()
    device = torch.device("cuda:%d"%args.gpu_id if cuda_condition else "cpu")

    config = config_from_meta(args.model_size)
    config = set_model_config(config, args, True)

    print("Creating Model...")
    model = GPT2LMHeadModel(config)
    model.to(device)
    
    print("Creating Dataloader...")
    dataset = DataLoaderForGPT(args, device)
    trainloader = DataLoader(
        dataset=dataset,
        batch_size=args.global_train_batch_size,
        shuffle=False
    )
    
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.adam_weight_decay)

    profiler = GalvatronProfiler(args)
    profiler.set_profiler_single()

    profiler.profile_memory(0, "After creating model")
    print("Start training...")
    for ep in range(args.epochs):
        if not args.check_loss and not args.profile:
            trainloader = tqdm(trainloader)
        for iter, batch in enumerate(trainloader):
            profiler.profile_time_start(iter)
            
            input_ids = batch
            
            profiler.profile_memory(iter, "Before Forward")

            loss = model_forward(model, input_ids)

            profiler.profile_memory(iter, "After Forward")

            loss.backward()

            profiler.profile_memory(iter, "After Backward")
            
            optimizer.step()

            profiler.profile_memory(iter, "After optimizer_step")
            
            optimizer.zero_grad()

            print_loss(args, loss, ep, iter)
            
            profiler.post_profile_memory(iter)
            profiler.profile_time_end(iter)

if __name__ == '__main__':
    args = initialize_galvatron(model_args, mode='train')
    set_seed()
    train(args)