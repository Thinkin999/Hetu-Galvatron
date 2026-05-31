#!/usr/bin/env bash
# n1: validate sp>1 COMMUNICATION strategies (ulysses8 a2a, ring8 p2p, usp2x4)
# on single-node 8-GPU. One sequence per step (GBS=1) spread across all 8 GPUs
# via the group's sp/cp. Compares measured [FBPROF] to cost-model total_time.
# IMPORTANT: mnist must be fully dead (contention inflates fb).

set -uo pipefail
ENVBIN=/mnt/bn/wyj-data0-hl/lqs/envs/galvatron-adacpsp-py39-torch21-cu121/bin
MODEL_DIR=/mnt/bn/wyj-data0-hl/lqs/src/Hetu-Galvatron/galvatron/models/varlen_llama_hf
cd "$MODEL_DIR"
export PYTHONPATH="$MODEL_DIR/../../site_package:$MODEL_DIR/../../.."
unset MASTER_ADDR MASTER_PORT RANK WORLD_SIZE LOCAL_RANK 2>/dev/null || true

STRATS="${STRATS:-ulysses:8 ring:8 usp:2x4}"
SEQ_SET="${SEQ_SET:-4096 8192}"
GBS="${GBS:-8}"   # must be multiple of world/(pp*tp*cp); 8 works for all strategies
ITERS="${ITERS:-12}"
RUN=/tmp/n1_comm
mkdir -p "$RUN"
SUMMARY="${RUN}/summary.tsv"
echo -e "strategy\tseq\tgbs\tfb_ms_median\tbefore_mb\tafter_bwd_mb" > "$SUMMARY"

for strat in $STRATS; do
  for S in $SEQ_SET; do
    label=$(echo "$strat" | tr ':' '_')
    log="${RUN}/${label}_seq${S}.log"
    echo "[$(date +%H:%M:%S)] cell strat=$strat seq=$S"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $ENVBIN/torchrun --standalone --nnodes 1 --nproc_per_node 8 \
      train_dist_adacpsp.py --model_size qwen2.5-7b \
      --set_model_config_manually 0 --set_layernum_manually 0 --set_seqlen_manually 1 \
      --vocab_size 152064 --hidden_size 3584 --num_hidden_layers 28 --num_attention_heads 28 \
      --seq_length "$S" --global_train_batch_size "$GBS" --train-iters "$ITERS" \
      --lr 1e-4 --adam_weight_decay 0.01 --dropout_prob 0.0 --check_loss 0 \
      --profile 1 --profile_start_iter 5 --profile_end_iter $((ITERS-1)) --save_profiled_memory 0 \
      --pp_deg 1 --global_tp_deg 1 --global_tp_consec 1 --sdp 0 --global_checkpoint 0 \
      --vocab_tp 1 --chunks 1 --global_cp_deg 1 --pipeline_type pipedream_flush \
      --default_dp_type zero2 --mixed_precision bf16 --use-flash-attn --initialize_on_meta 1 \
      --use-packing --use-adaCPSP --adaCPSP-sync-solver --adaCPSP-forced-strategy "$strat" \
      --dataset fix_length > "$log" 2>&1
    fbmed=$(grep -oE "FBPROF\] iter=[0-9]+ fb_ms=[0-9.]+" "$log" | awk -F'iter=| fb_ms=' '$2>=1{print $3}' | sort -n | awk '{a[NR]=$1} END{if(NR>0) print a[int((NR+1)/2)]; else print "NA"}')
    bmb=$(grep -A1 "Before Forward" "$log" | grep -oE "Max memory: [0-9.]+" | head -1 | grep -oE "[0-9.]+")
    abmb=$(grep -A1 "After Backward" "$log" | grep -oE "Max memory: [0-9.]+" | head -1 | grep -oE "[0-9.]+")
    echo -e "${strat}\t${S}\t${GBS}\t${fbmed:-NA}\t${bmb:-NA}\t${abmb:-NA}" >> "$SUMMARY"
    echo "   fb_med=${fbmed:-NA}ms before=${bmb:-NA} after_bwd=${abmb:-NA}"
  done
done

echo "==== SUMMARY ===="
cat "$SUMMARY"
