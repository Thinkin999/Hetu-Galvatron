(

    # 1. 清理环境变量 (加分号确保隔离)

    unset ARNOLD_WORKER_NUM ARNOLD_WORKER_GPU ARNOLD_ID METIS_WORKER_0_HOST METIS_WORKER_0_PORT;

    unset RUN_MODE MODELS SEQ_LENGTHS_K GBS_LIST STRATEGIES GPU_CONFIGS;

    unset NNODES NODE_RANK NPROC_PER_NODE MASTER_ADDR MASTER_PORT;

    unset NUM_ITERS WARMUP_ITERS TIMEOUT_SECONDS MEMORY_LIMIT_GB DEFAULT_DP_TYPE NUM_WORKERS DATASET;

    unset LIVE_LOG_TO_STDOUT LIVE_LOG_SCOPE AUTO_MEMORY_LIMIT EXTRA_TRAIN_ARGS;

    unset RESULT_ROOT LOCAL_RESULT_ROOT DATASET_MOUNT_DIR CUDA_VISIBLE_DEVICES;



    # 2. 设置基础路径 (注意路径两端建议加双引号)

    ROOT="/mnt/bn/wyj-data0-hl/lqs"



    # 3. 切换目录并激活环境 (给变量加双引号防止路径溢出)

    cd "$ROOT/src/Hetu-Galvatron" || exit 1

    source "$ROOT/tools/miniforge3/etc/profile.d/conda.sh"

    conda activate "$ROOT/envs/galvatron-adacpsp-py39-torch21-cu121"



    # 4. 获取随机可用端口

    FREE_PORT=$(python - <<'PY'

import socket

with socket.socket() as s:

    s.bind(("", 0))

    print(s.getsockname()[1])

PY

)

    echo "Using MASTER_PORT=$FREE_PORT"



    # 5. 设置运行参数

    export RUN_MODE=quick

    export NNODES=1

    export NODE_RANK=0

    export NPROC_PER_NODE=8

    export MASTER_ADDR=127.0.0.1

    export MASTER_PORT="$FREE_PORT"

    export MODELS="qwen2.5-7b"

    export SEQ_LENGTHS_K="16"

    export GBS_LIST="16"

    export STRATEGIES="adacpsp"

    export NUM_ITERS="5"

    export WARMUP_ITERS="2"

    export TIMEOUT_SECONDS="-1"

    export NUM_WORKERS="0"

    export AUTO_MEMORY_LIMIT="1"



    # 6. 执行脚本

    bash byted_experiments/entrypoint.sh

)