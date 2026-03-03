export NUM_NODES=1
export NUM_GPUS_PER_NODE=8
export MASTER_ADDR=localhost
export MASTER_PORT=29020
export NODE_RANK=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
#export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_HCA=mlx5_2,mlx5_5
LAUNCHER="python3 -m torch.distributed.launch"
LAUNCHER="${LAUNCHER} --nnodes ${NUM_NODES}"
LAUNCHER="${LAUNCHER} --nproc_per_node ${NUM_GPUS_PER_NODE}"
LAUNCHER="${LAUNCHER} --master_addr ${MASTER_ADDR}"
LAUNCHER="${LAUNCHER} --master_port ${MASTER_PORT}"
LAUNCHER="${LAUNCHER} --node_rank ${NODE_RANK}"

TRAINER="train_dist_adacpsp.py"
#TRAINER="train_dist_profiler_use.py"
DATA_PATH=/home/pkuhetu/lqs/megatron_data/my_qwen_text 
VOCAB_FILE=/home/pkuhetu/lqs/megatron_data/qwen2.5_tokenizer/vocab.json
TOKENIZER_MODEL=/home/pkuhetu/lqs/megatron_data/qwen2.5_tokenizer

MODEL_ARGS="
    --model_size qwen2.5-3b \
    --set_model_config_manually 0 \
    --set_layernum_manually 0 \
    --set_seqlen_manually 1 \
    --vocab_size 32000 \
    --hidden_size 1536 \
    --num_hidden_layers 1 \
    --num_attention_heads 32 \
    --seq_length 524288"

TRAIN_ARGS="
    --global_train_batch_size 8 \
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
    --save_profiled_memory 0"
   #--vocab_tp 2 \
DATA_ARGS="
    --data-path /home/pkuhetu/lqs/megatron_data/my_qwen_text/my-qwen2.5_text_text_sentence \
    --split 949,50,1 \
    --tokenizer-type NullTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --num-workers 0 \
    --no-create-attention-mask-in-dataloader
"

# CKPT_ARGS="
#     --load /home/pkuhetu/lxy/checkpoints/llama2-7b-chat-hf-split
# "

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
    --global_tp_deg 1 \
    --global_tp_consec 1 \
    --sdp 1 \
    --global_checkpoint 0 \
    --vocab_tp 1 \
    --vocab_cp 8 \
    --chunks 2 \
    --global_cp_deg 8 \
    --pipeline_type pipedream_flush \
    --default_dp_type zero2 \
    --mixed_precision bf16 \
    --sequence-parallel \
    --use-ulysses \
    --use-flash-attn \
    --initialize_on_meta 1 \
    --use-packing "
#     --use-adaCPSP \
#     --adaCPSP-strategy adaptive \
#     --memory-limit-gb 28 \
# "
    #--galvatron_config_path /home/pkuhetu/lqs/galvatron_lxy/Hetu-Galvatron/galvatron/models/llama_hf/configs/test.json"
#--sequence-parallel \
${LAUNCHER} ${TRAINER} ${MODEL_ARGS} ${TRAIN_ARGS} ${PARALLEL_ARGS} ${DATA_ARGS} ${CKPT_ARGS}