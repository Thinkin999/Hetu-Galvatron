
INPUT_PATH=/mnt/bn/wyj-data-lq/lxy/Mixtral-8x7B-v0.1
OUTPUT_PATH=/mnt/bn/wyj-data-lq/lxy/checkpoint/mixtral-split

CHECKPOINT_ARGS="
    --input_checkpoint $INPUT_PATH \
    --output_dir $OUTPUT_PATH
"

python checkpoint_convert_h2g.py --model_type llama ${CHECKPOINT_ARGS}