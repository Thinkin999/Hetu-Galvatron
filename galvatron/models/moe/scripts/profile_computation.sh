# export CUDA_DEVICE_MAX_CONNECTIONS=1 # to enable CP computation/communication streams to overlap
export TORCH_NCCL_AVOID_RECORD_STREAMS=1 # to avoid max_reserved_memory and max_allocated_memory over-sized
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export NVTE_BATCH_MHA_P2P_COMM=1 # to force TransformerEngine to use batched send/recv for CP
export NCCL_DEBUG=WARN

export CUDA_VISIBLE_DEVICES=1

export NUM_NODES=1
export NUM_GPUS_PER_NODE=1
export MASTER_ADDR=localhost
export MASTER_PORT=29520
export NODE_RANK=0

export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=2

# export NCCL_NVLS_ENABLE=1
export GLOO_SOCKET_IFNAME=eth0
# export CUDA_DEVICE_MAX_CONNECTIONS=1

LAUNCHER="python3 -m torch.distributed.launch"
LAUNCHER="${LAUNCHER} --nnodes ${NUM_NODES}"
LAUNCHER="${LAUNCHER} --nproc_per_node ${NUM_GPUS_PER_NODE}"

export PROFILE_LAUNCHER="$LAUNCHER"
export PROFILE_TRAINER="train_dist_random.py"

MODEL_ARGS="
    --model_size mixtral-8x7b \
    --set_model_config_manually 0 \
    --set_experts_manually 0 \
    --vocab_size 32000 \
    --hidden_size 4096 \
    --num_attention_heads 32 \
    --seq_length 5120"

PROFILE_ARGS="
    --profile_mode batch \
    --profile_type computation \
    --profile_seq_length_list 8192 \
    --profile_min_batch_size 1 \
    --profile_max_batch_size 12 \
    --profile_batch_size_step 1 \
    --layernum_min 2 \
    --layernum_max 4 \
    --mixed_precision bf16 \
    --use-flash-attn \
    --sequence_parallel \
    --profile_flow_control scripts_only \
    --profile_unit mlp"

# PROFILE_ARGS="
#     --profile_mode sequence \
#     --profile_type computation \
#     --profile_batch_size 1 \
#     --profile_min_seq_length 4096 \
#     --profile_max_seq_length 32768 \
#     --profile_seq_length_step 4096 \
#     --layernum_min 1 \
#     --layernum_max 2 \
#     --mixed_precision bf16 \
#     --use-flash-attn"

# models in flash_attn cannot use fp32 without flash_attn
# PROFILE_ARGS="
#     --profile_mode static \
#     --profile_type computation \
#     --profile_batch_size 4 \
#     --layernum_min 12 \
#     --layernum_max 24 \
#     --mixed_precision fp32"

# PROFILE_UNIT="
#     --profile_unit attention"

python3 profiler.py ${MODEL_ARGS} ${PROFILE_ARGS}