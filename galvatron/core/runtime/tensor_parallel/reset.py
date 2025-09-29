import torch
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker, get_data_parallel_rng_tracker_name, get_expert_parallel_rng_tracker_name, get_tensor_parallel_rng_tracker_name
from megatron.training import get_args
from galvatron.core.runtime.moe.router import TopKRouter

# from torch.nn.init import xavier_uniform_ as init_method
from .utils import init_method_normal

# TODO: reset expert param / fine-grained correctly

def colummn_row_reset_parameters(self):
    args = get_args()
    if getattr(self, "is_expert", False):
        with get_cuda_rng_tracker().fork(get_expert_parallel_rng_tracker_name(self.tp_and_ep_group)):
            init_method = init_method_normal(args.init_method_std)
            init_method(self.weight)
    else:
        with get_cuda_rng_tracker().fork(get_tensor_parallel_rng_tracker_name(self.tp_group)):
            init_method = init_method_normal(args.init_method_std)
            init_method(self.weight)
    if hasattr(self, "bias") and self.bias != None:
        with torch.no_grad():
            self.bias.zero_()

def router_reset_parameters(self):
    args = get_args()
    with get_cuda_rng_tracker().fork(get_data_parallel_rng_tracker_name()):
        init_method = init_method_normal(args.init_method_std)
        init_method(self.weight)

def init_reset_parameter():
    ColumnParallelLinear.reset_parameters = colummn_row_reset_parameters
    RowParallelLinear.reset_parameters = colummn_row_reset_parameters
    VocabParallelEmbedding.reset_parameters = colummn_row_reset_parameters
    TopKRouter.reset_parameters = router_reset_parameters
