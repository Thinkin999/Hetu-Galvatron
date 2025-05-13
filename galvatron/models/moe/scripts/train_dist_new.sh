# export CUDA_DEVICE_MAX_CONNECTIONS=1 # to enable CP computation/communication streams to overlap
export TORCH_NCCL_AVOID_RECORD_STREAMS=1 # to avoid max_reserved_memory and max_allocated_memory over-sized
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export NVTE_BATCH_MHA_P2P_COMM=1 # to force TransformerEngine to use batched send/recv for CP
export NCCL_DEBUG=WARN

ports=(`echo $METIS_WORKER_0_PORT | tr ',' ' '`)
port=${ports[0]}
export NUM_NODES=1
export NUM_GPUS_PER_NODE=$ARNOLD_WORKER_GPU
export MASTER_ADDR=$METIS_WORKER_0_HOST
export MASTER_PORT=$port
export NODE_RANK=`expr $ARNOLD_ID - 0`

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
LAUNCHER="${LAUNCHER} --master_addr ${MASTER_ADDR}"
LAUNCHER="${LAUNCHER} --master_port ${MASTER_PORT}"
LAUNCHER="${LAUNCHER} --node_rank ${NODE_RANK}"

TRAINER="train_dist.py"
DATA_PATH=/mnt/bn/wyj-data-lq/lxy/dataset/wikitext/mixtral_text_document
TOKENIZER_MODEL=/mnt/bn/wyj-data-lq/lxy/Mixtral-8x7B-v0.1

MODEL_ARGS="
    --model_size mixtral-8x7b \
    --set_model_config_manually 0 \
    --set_layernum_manually 1 \
    --set_seqlen_manually 1 \
    --set_experts_manually 0 \
    --vocab_size 32000 \
    --hidden_size 4096 \
    --num_hidden_layers 8 \
    --num_attention_heads 32 \
    --seq_length 2048"

TRAIN_ARGS="
    --global_train_batch_size 64 \
    --train-iters 20 \
    --eval-iters 1 \
    --lr 1.25e-6 \
    --lr-decay-style cosine \
    --min-lr 1.25e-7 \
    --lr-warmup-fraction 0.1 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1.0e-5 \
    --init-method-std 0.01 \
    --adam_weight_decay 0.01 \
    --dropout_prob 0.1 \
    --check_loss 0 \
    --profile 1 \
    --no_async_grad_reduce \
    --save_profiled_memory 0"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 949,50,1 \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL}
"

CKPT_ARGS="
    --load /home/pkuhetu/lxy/checkpoints/llama2-7b-chat-hf-split
"

# CKPT_ARGS="
#     --save /home/pkuhetu/lxy/checkpoints/galvatron_save_llama
#     --save-interval 10
# "

# CKPT_ARGS="
#     --load /home/pkuhetu/lxy/checkpoints/galvatron_save_llama \
#     --load_iteration 10 \
#     --distributed_checkpoint
# "

PARALLEL_ARGS="
    --pp_deg 1 \
    --global_tp_deg 8 \
    --global_tp_consec 1 \
    --global_ep_deg 8 \
    --global_tp_of_ep_deg 1 \
    --sdp 1 \
    --global_checkpoint 1 \
    --vocab_tp 8 \
    --chunks 8 \
    --pipeline_type pipedream_flush \
    --default_dp_type zero2 \
    --mixed_precision bf16 \
    --sequence-parallel \
    --use-flash-attn \
    --initialize_on_meta 1"
    # --galvatron_config_path ./configs/galvatron_config_hidden4096_head32_1nodes_8gpus_per_node_36GB_bf16_[tpconsec_off].json"

${LAUNCHER} ${TRAINER} ${MODEL_ARGS} ${TRAIN_ARGS} ${PARALLEL_ARGS} ${DATA_ARGS} # ${CKPT_ARGS}