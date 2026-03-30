#!/bin/bash
###############################################################################
# Recommended Merlin Seed entrypoint.
#
# Usage on platform:
#   bash byted_experiments/entrypoint.sh
#
# Optional:
#   RUN_MODE=quick bash byted_experiments/entrypoint.sh
###############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_MODE="${RUN_MODE:-full}"

echo "=========================================="
echo "ByteDance Merlin Seed entrypoint"
echo "=========================================="
echo "ARNOLD_WORKER_GPU=${ARNOLD_WORKER_GPU:-<unset>}"
echo "ARNOLD_WORKER_NUM=${ARNOLD_WORKER_NUM:-<unset>}"
echo "ARNOLD_ID=${ARNOLD_ID:-<unset>}"
echo "METIS_WORKER_0_HOST=${METIS_WORKER_0_HOST:-<unset>}"
echo "METIS_WORKER_0_PORT=${METIS_WORKER_0_PORT:-<unset>}"
echo "RUN_MODE=${RUN_MODE}"
echo "=========================================="

case "${RUN_MODE}" in
    quick)
        bash "${SCRIPT_DIR}/run_quick_test.sh"
        ;;
    full)
        bash "${SCRIPT_DIR}/run_all_experiments.sh"
        ;;
    *)
        echo "ERROR: unsupported RUN_MODE=${RUN_MODE}"
        echo "Supported values: quick, full"
        exit 1
        ;;
esac
