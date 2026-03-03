from typing import List, Optional, Union

import torch
from megatron.training import get_args


def listify_model(model: Union[torch.nn.Module, List[torch.nn.Module]]) -> List[torch.nn.Module]:
    if isinstance(model, list):
        return model
    return [model]


def chunk_batch(inputs, chunks):
    if inputs is None:
        return inputs

    batches = [[] for _ in range(chunks)]
    # Actual number of chunks produced
    num_chunks = -1
    if not get_args().use_packing or len(inputs) <= 1:
        for input in inputs:
            if torch.is_tensor(input):
                # Chunk only tensors.
                tensors = input.chunk(chunks)

                # Validate number of chunks equal across all inputs.
                if num_chunks != -1 and num_chunks != len(tensors):
                    raise RuntimeError(f'Found different number of chunks produced for inputs: {num_chunks} and {len(tensors)}')
                num_chunks = len(tensors)

                for i, tensor in enumerate(tensors):
                    batches[i].append(tensor)
            else:
                # Replicate non-tensors or tensors wrapped with 'NoChunk'.
                for i in range(chunks):
                    batches[i].append(input)
                num_chunks = chunks
    else:
        # inputs: [input_ids, cu_seqlens]
        input_ids, cu_seqlens = inputs
        assert isinstance(input_ids, torch.Tensor) and len(input_ids.shape) == 1, "packing use flattened input!"
        assert isinstance(cu_seqlens, torch.Tensor) and len(cu_seqlens.shape) == 1
        tot_bsz  = len(cu_seqlens) - 1
        micro_bsz = tot_bsz // chunks
        for i in range(chunks):
            local_cu_seqlens = cu_seqlens[micro_bsz * i : (micro_bsz * (i+1) + 1)].clone()
            local_input_ids = input_ids[local_cu_seqlens[0] : local_cu_seqlens[-1]]
            local_cu_seqlens = local_cu_seqlens - local_cu_seqlens[0]
            batches[i].append(local_input_ids),batches[i].append(local_cu_seqlens)
        num_chunks = chunks
    # Truncate to actual number of chunks
    batches = batches[:num_chunks]
    return batches


def chunk_dict(kwargs, chunks):
    batches = [{} for _ in range(chunks)]
    num_chunks = -1
    for k, v in kwargs.items():
        if torch.is_tensor(v) and not (k.endswith("_mask") and v.shape[0] == 1) and not k.startswith("rotary"):
            tensors = v.chunk(chunks)
            if num_chunks != -1 and num_chunks != len(tensors):
                raise RuntimeError(
                    f"Found different number of chunks produced for inputs: {num_chunks} and {len(tensors)}"
                )
            num_chunks = len(tensors)
            for i, tensor in enumerate(tensors):
                batches[i][k] = tensor
        else:
            for i in range(chunks):
                batches[i][k] = v

    if num_chunks >= 0:
        batches = batches[:num_chunks]
    return batches
