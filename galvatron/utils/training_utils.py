import torch
import numpy as np
import random 
import torch.distributed as dist
import torch.distributed
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
#from galvatron.flexsp_solver import Sequence, SeqBucket
import multiprocessing as mp

def get_args():
    from galvatron.core import get_args as _get_args
    return _get_args()

global_group_set = [] #indicate wheter a group is created
group_pool = {}#通信的进程池
is_first_iter = True
solve_process = None
mp_manager = mp.Manager()
solved_globalbatch_gps =mp_manager.list()#已经解决的microbatch的分配 []
flexSP_optimizer = None
adaCPSP_optimizer = None
prev_batch = None

def solve_target(seqs, shared_globalbatch_gps):
    args = get_args()
    flexSP_optimizer.token_info(seqs)
    import time
    ignore_strategies = [32] if 'wikipedia' in args.dataset and args.seq_length <= 192000 else []
    if flexSP_optimizer.strategy == 'flexSP':
        start = time.time()
        print('Running Solver...')
        chunk_alg = 'sort_consec'
        globalbatch_groups, globalbatch_results = flexSP_optimizer.solve_flexSP_globalbatch_mp_gbmb(seqs, chunk_alg = chunk_alg, mb_option_num = 4)
        end = time.time()
        print('Solver Time Cost: %.4f'%(end-start))
    elif flexSP_optimizer.strategy == 'adaptive_bfd':
        print('Running Baseline BFD (adaptive sp size)...')
        globalbatch_groups, globalbatch_results = \
            flexSP_optimizer.homo_sp_baseline_ffd_bfd_globalbatch(seqs, 'bfd', sp_select_rule='adaptive', ignore_strategies=ignore_strategies)
    elif flexSP_optimizer.strategy == 'fix_sp_bfd':
        print(f"Running Baseline BFD (fix sp size={flexSP_optimizer.fix_sp_size})...")
        globalbatch_groups, globalbatch_results = \
            flexSP_optimizer.homo_sp_baseline_ffd_bfd_globalbatch(seqs, 'bfd', sp_select_rule='fix_sp', ignore_strategies=ignore_strategies)
    globalbatch_time = sum([results['M'] for results in globalbatch_results])
    mb_num = len(globalbatch_groups)
    print(f'Globalbatch Final Results: microbatch size = {mb_num}, Time = {globalbatch_time:.2f}')
    for idx, (groups, results) in enumerate(zip(globalbatch_groups, globalbatch_results)):
        print(f"============= Microbatch {idx}, Time: {results['M']:.2f} =============")
        for sp_size, group in groups:
            flexSP_optimizer.print_group_seqs_info(group, sp_size)
    for mbsz in globalbatch_groups:
        for _ in range(len(mbsz)):
            mbsz[_] = (mbsz[_][0], [seq.id for seq in mbsz[_][1]])
    shared_globalbatch_gps.extend(globalbatch_groups)

def convert_microbatch_res(micro_res):
    global global_group_set, group_pool
    cum_cnt = 0
    sp_group = None
    batch_indices = []
    for res_tuple in micro_res:
        sp_size, seq_id_list  = res_tuple
        rank_start = cum_cnt
        rank_end = cum_cnt + sp_size
        ranks = list(range(rank_start, rank_end))
        if tuple(ranks) not in global_group_set:
            sp_group_ = torch.distributed.new_group(ranks)
            global_group_set.append(tuple(ranks))
            if torch.distributed.get_rank() in ranks:
                group_pool[tuple(ranks)] = sp_group_
        cum_cnt += sp_size
        if torch.distributed.get_rank() in ranks:
            sp_group = group_pool[tuple(ranks)]
            batch_indices = seq_id_list
    return batch_indices, sp_group

def set_seed():
    seed = 123
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    

def collate_fn(batch):#这里的batch应该是一个list
    global is_first_iter, solve_process, solved_globalbatch_gps, prev_batch
    max_len = max([len(seq) for seq in batch])#global batch里面最长的sequence length
    world_size = torch.distributed.get_world_size()
    max_len = ((max_len - 1) // world_size + 1) * world_size#变成world size倍
    args = get_args()
    max_len = min(max_len, get_args().seq_length)#用max len和seq length的最小值
    if not get_args().use_packing:
        padded_batch = torch.zeros((len(batch), max_len), dtype=torch.long,device=batch[0].device)
        for i, seq in enumerate(batch):
            padded_batch[i, :len(seq)] = seq
        return padded_batch#padding在一起的风格
    else:
        if adaCPSP_optimizer:
            # ═══════════════════════════════════════════════
            # AdaCPSP: solver determines microbatch strategy
            # ═══════════════════════════════════════════════
            from galvatron.models.varlen_llama_hf.adacpsp_solver import Sequence
            
            rank = dist.get_rank()
            args = get_args()
            args.adacpsp_strategies = []
            
            if rank == 0:
                # Create Sequence objects for solver
                seqs = [Sequence(seq=s.shape[0], id=i) for i, s in enumerate(batch)]
                
                # Run solver
                all_groups, all_results = adaCPSP_optimizer.solve_globalbatch(seqs)
                
                if len(all_groups) == 0:
                    # Solver failed → fallback: single microbatch, Ulysses×N
                    world_size = torch.distributed.get_world_size()
                    print("[AdaCPSP] Solver failed, fallback to Ulysses×" + str(world_size))
                    all_groups = [[(type('S', (), {'attn_type': 'ulysses', 'parallel_size': world_size})(), seqs)]]
                
                # Encode microbatch plans for broadcasting
                # Each microbatch: [sp_size, cp_size, num_seqs, seq_id_0, seq_id_1, ...]
                # Per-microbatch homogeneous: first group's strategy used for the whole microbatch
                encoded_mbs = []
                for mb_groups in all_groups:
                    all_seq_ids = []
                    strategy = None
                    for strat, group_seqs in mb_groups:
                        if strategy is None:
                            strategy = strat
                        for seq in group_seqs:
                            all_seq_ids.append(seq.id)
                    
                    # Encode as (total_parallel_size, 1) for broadcasting
                    # The receiver side will re-derive (sp, cp) using fixed tp_deg
                    total_parallel = strategy.parallel_size
                    encoded_mbs.append([total_parallel, 1] + all_seq_ids)
                
                # Broadcast number of microbatches
                num_mb = torch.LongTensor([len(encoded_mbs)]).cuda()
                dist.broadcast(num_mb, 0)
                
                for enc in encoded_mbs:
                    enc_t = torch.LongTensor(enc).cuda()
                    length_t = torch.LongTensor([len(enc_t)]).cuda()
                    dist.broadcast(length_t, 0)
                    dist.broadcast(enc_t, 0)
            else:
                num_mb = torch.LongTensor([0]).cuda()
                dist.broadcast(num_mb, 0)
                
                encoded_mbs = []
                for _ in range(num_mb.item()):
                    length_t = torch.LongTensor([0]).cuda()
                    dist.broadcast(length_t, 0)
                    enc_t = torch.zeros(length_t.item(), dtype=torch.long).cuda()
                    dist.broadcast(enc_t, 0)
                    encoded_mbs.append(enc_t.cpu().tolist())
            
            # Decode and build microbatches
            # Return format: list of [[packed_tokens, cu_seqlens]] per microbatch
            # Strategy info stored in args.adacpsp_strategies
            #
            # CRITICAL: sp_size (Ulysses) must always equal the model's construction-time
            # tp_deg because QKV weights are physically sharded by TP degree.
            # Only cp_size (Ring) can vary dynamically.
            # The solver's parallel_size is the TOTAL GPU count per group.
            # We derive: sp_size = tp_deg (fixed), cp_size = total / tp_deg.
            fixed_sp = args.global_tp_deg  # = tp_deg from model construction
            
            microbatches = []
            for mb_idx, enc in enumerate(encoded_mbs):
                if isinstance(enc, torch.Tensor):
                    enc = enc.cpu().tolist()
                raw_sp_size = int(enc[0])
                raw_cp_size = int(enc[1])
                seq_ids = [int(x) for x in enc[2:]]
                
                # Derive correct (sp, cp) from total parallel size
                total_parallel = raw_sp_size * raw_cp_size
                sp_size = fixed_sp
                cp_size = max(1, total_parallel // fixed_sp)
                
                strat_info = {
                    "sp_size": sp_size,
                    "cp_size": cp_size,
                    "attn_type": "combined" if cp_size > 1 else "ulysses",
                }
                # Store strategy for this microbatch
                args.adacpsp_strategies.append(strat_info)
                
                # Build packed tokens + cu_seqlens
                sequences = [batch[sid] for sid in seq_ids]
                cu_seqlens = torch.empty(len(sequences) + 1, dtype=torch.int64,
                                        device=batch[0].device)
                cu_seqlens[0] = 0
                for j in range(len(sequences)):
                    cu_seqlens[j + 1] = cu_seqlens[j] + len(sequences[j])
                packed_tokens = torch.cat(sequences)
                
                microbatches.append([[packed_tokens, cu_seqlens]])
            
            return microbatches
        else:
            cu_seqlens = torch.empty(len(batch)+1, dtype=torch.int64)
            cu_seqlens[0] = 0
            for _ in range(1, len(cu_seqlens)):
                cu_seqlens[_] = cu_seqlens[_-1] + len(batch[_ - 1])
            batch = torch.concat(batch)
            return [batch, cu_seqlens]

def distributed_dataloader(dataset, global_bsz, shuffle = True, args = None, group = None, adaCPSP_optimizer_ = None):
    rank = torch.distributed.get_rank(group)
    world_size = torch.distributed.get_world_size(group)
    global adaCPSP_optimizer
    adaCPSP_optimizer = adaCPSP_optimizer_
    # pp_deg = args.pp_deg if args is not None and 'pp_deg' in args else 1
    # data_num_replicas = world_size // pp_deg
    train_batch_size_input = global_bsz // world_size
    trainloader = DataLoader(dataset=dataset,
                            batch_size=train_batch_size_input,
                            sampler=DistributedSampler(dataset,shuffle=shuffle,num_replicas=world_size,rank=rank), 
                            collate_fn=collate_fn)
    return trainloader

def print_loss(args, loss, ep, iter):
    if args.check_loss or args.profile:
        if loss is None:
            return
        if isinstance(loss, (list, tuple)): # Average loss of each microbatch
            if len(loss) == 0:
                return
            if isinstance(loss[0], torch.Tensor):
                loss = np.mean([l.item() for l in loss])
            else:
                loss = np.mean(loss)
        else:
            loss = loss.item() if isinstance(loss, torch.Tensor) else loss
        print('[Epoch %d] (Iteration %d): Loss = %.3f'% (ep,iter,loss))
