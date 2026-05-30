#!/usr/bin/env bash
# Single-GPU galvatron layer-diff sweep (REAL runtime, not bare HF).
# Validated that single-GPU galvatron == multi-GPU (comm hidden up to 8 GPU
# NVLink), so single-GPU clean profiling is representative.
#
# Sweeps L x seq x ckpt; runs forced ulysses:1 (sp=1, no attention comm) so the
# measured fb = compute + linear + (hidden) FSDP comm. Parses [FBPROF] (median
# steady fb_ms) and [Profile] memory lines from each run.
#
# IMPORTANT: ensure no mnist/other GPU job is running (contention inflates fb!).

set -uo pipefail
ENVBIN=/mnt/bn/wyj-data0-hl/lqs/envs/galvatron-adacpsp-py39-torch21-cu121/bin
MODEL_DIR=/mnt/bn/wyj-data0-hl/lqs/src/Hetu-Galvatron/galvatron/models/varlen_llama_hf
cd "$MODEL_DIR"
export PYTHONPATH="$MODEL_DIR/../../site_package:$MODEL_DIR/../../.."
unset MASTER_ADDR MASTER_PORT RANK WORLD_SIZE LOCAL_RANK 2>/dev/null || true

LAYER_SET="${LAYER_SET:-1 2 4}"
SEQ_SET="${SEQ_SET:-4096 8192 16384}"
CKPT="${CKPT:-0}"
ITERS="${ITERS:-12}"
RUN=/tmp/galv_layerdiff
mkdir -p "$RUN"
SUMMARY="${RUN}/summary_ckpt${CKPT}.tsv"
echo -e "layers\tseq\tckpt\tfb_ms_median\tbefore_mb\tafter_fwd_mb\tafter_bwd_mb" > "$SUMMARY"

for L in $LAYER_SET; do
  for S in $SEQ_SET; do
    log="${RUN}/L${L}_seq${S}_ckpt${CKPT}.log"
    echo "[$(date +%H:%M:%S)] cell L=$L seq=$S ckpt=$CKPT"
    CUDA_VISIBLE_DEVICES=0 $ENVBIN/torchrun --standalone --nnodes 1 --nproc_per_node 1 \
      train_dist_adacpsp.py --model_size qwen2.5-7b \
      --set_model_config_manually 0 --set_layernum_manually 1 --set_seqlen_manually 1 \
      --vocab_size 152064 --hidden_size 3584 --num_hidden_layers "$L" --num_attention_heads 28 \
      --seq_length "$S" --global_train_batch_size 1 --train-iters "$ITERS" \
      --lr 1e-4 --adam_weight_decay 0.01 --dropout_prob 0.0 --check_loss 0 \
      --profile 1 --profile_start_iter 5 --profile_end_iter $((ITERS-1)) --save_profiled_memory 0 \
      --pp_deg 1 --global_tp_deg 1 --global_tp_consec 1 --sdp 0 --global_checkpoint "$CKPT" \
      --vocab_tp 1 --chunks 1 --global_cp_deg 1 --pipeline_type pipedream_flush \
      --default_dp_type zero2 --mixed_precision bf16 --use-flash-attn --initialize_on_meta 1 \
      --use-packing --use-adaCPSP --adaCPSP-sync-solver --adaCPSP-forced-strategy "ulysses:1" \
      --dataset fix_length > "$log" 2>&1
    # parse median of steady fb (iter>=1)
    fbmed=$(grep -oE "FBPROF\] iter=[0-9]+ fb_ms=[0-9.]+" "$log" | awk -F'iter=| fb_ms=' '$2>=1{print $3}' | sort -n | awk '{a[NR]=$1} END{if(NR>0) print a[int((NR+1)/2)]; else print "NA"}')
    bmb=$(grep -A1 "Before Forward" "$log" | grep -oE "Max memory: [0-9.]+" | head -1 | grep -oE "[0-9.]+")
    afmb=$(grep -A1 "After Forward" "$log" | grep -oE "Max memory: [0-9.]+" | head -1 | grep -oE "[0-9.]+")
    abmb=$(grep -A1 "After Backward" "$log" | grep -oE "Max memory: [0-9.]+" | head -1 | grep -oE "[0-9.]+")
    echo -e "${L}\t${S}\t${CKPT}\t${fbmed:-NA}\t${bmb:-NA}\t${afmb:-NA}\t${abmb:-NA}" >> "$SUMMARY"
    echo "   fb_med=${fbmed:-NA}ms  before=${bmb:-NA}  after_fwd=${afmb:-NA}  after_bwd=${abmb:-NA}"
  done
done

echo "==== SUMMARY (${SUMMARY}) ===="
cat "$SUMMARY"
