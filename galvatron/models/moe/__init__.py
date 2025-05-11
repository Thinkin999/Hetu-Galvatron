import torch
from megatron.legacy.model.rms_norm import RMSNorm as MoERMSNorm

from .MoEModel_hybrid_parallel import construct_hybrid_parallel_model, get_hybrid_parallel_configs, moe_model_hp


def rms_reset_parameters(self):
    with torch.no_grad():
        torch.nn.init.ones_(self.weight)


MoERMSNorm.reset_parameters = rms_reset_parameters
