export NUM_NODES=1
export NUM_GPUS_PER_NODE=8
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
# export NCCL_SOCKET_IFNAME=ib0
export NODE_RANK=$RANK

LAUNCHER="python3 -m torch.distributed.launch"
LAUNCHER="${LAUNCHER} --nnodes ${NUM_NODES}"
LAUNCHER="${LAUNCHER} --nproc_per_node ${NUM_GPUS_PER_NODE}"

export PROFILE_LAUNCHER="$LAUNCHER"
export PROFILE_TRAINER="train_dist_test.py"

MODEL_ARGS_BASE="
    --model_type bert \
    --model_size bert-base \
    --set_model_config_manually 0 \
    --vocab_size 30522 \
    --hidden_size 768 \
    --num_attention_heads 12 \
    --num_hidden_layers 12 \
    --intermediate_size 3072 \
    --max_position_embeddings 512 \
    --seq_length 512"

MODEL_ARGS_LARGE="
    --model_type bert \
    --model_size bert-large \
    --set_model_config_manually 0 \
    --vocab_size 30522 \
    --hidden_size 1024 \
    --num_attention_heads 16 \
    --num_hidden_layers 24 \
    --intermediate_size 4096 \
    --max_position_embeddings 512 \
    --seq_length 512"

PROFILE_ARGS_BF16="
    --profile_mode sequence \
    --profile_type memory \
    --profile_batch_size 8 \
    --profile_min_seq_length 512 \
    --profile_max_seq_length 8192 \
    --layernum_min 1 \
    --layernum_max 2 \
    --max_tp_deg 8 \
    --profile_dp_type zero3 \
    --mixed_precision bf16 \
    --sequence_parallel \
    --use-flash-attn"

# python3 profiler.py ${MODEL_ARGS_BASE} ${PROFILE_ARGS_BF16}
python3 profiler.py ${MODEL_ARGS_LARGE} ${PROFILE_ARGS_BF16}