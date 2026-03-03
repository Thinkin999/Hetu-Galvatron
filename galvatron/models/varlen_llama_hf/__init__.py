import torch
from megatron.legacy.model.rms_norm import RMSNorm as LlamaRMSNorm

from .LlamaModel_hybrid_parallel import construct_hybrid_parallel_model, get_hybrid_parallel_configs, llama_model_hp

# AdaCPSP components
from .adacpsp_solver import (
    Sequence,
    AdaCPSPCostModel,
    AdaCPSPOptimizer,
    AdaCPSPConfig,
    CommunicationGroupManager,
)
from .adacpsp_dataloader import (
    distributed_dataloader as adacpsp_distributed_dataloader,
    collate_fn as adacpsp_collate_fn,
    print_loss as adacpsp_print_loss,
    set_seed as adacpsp_set_seed,
)


def rms_reset_parameters(self):
    with torch.no_grad():
        torch.nn.init.ones_(self.weight)


LlamaRMSNorm.reset_parameters = rms_reset_parameters
