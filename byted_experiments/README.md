# ByteDance Merlin Seed Experiments

This directory mirrors the workflow under `experiments/`, but the launch path
is adapted for Merlin Seed / Arnold style multi-node jobs.

## Files

- `entrypoint.sh`: recommended platform entry command
- `run_all_experiments.sh`: sweep models / sequence lengths / GBS / strategies
- `run_quick_test.sh`: small sanity run
- `analyze_results.py`: reuses the analyzer from `experiments/`

## Expected platform environment variables

The scripts read these variables when running on Merlin Seed:

- `ARNOLD_WORKER_GPU`
- `ARNOLD_WORKER_NUM`
- `ARNOLD_ID`
- `METIS_WORKER_0_HOST`
- `METIS_WORKER_0_PORT`

## Optional overrides

- `MODELS`
- `SEQ_LENGTHS_K`
- `GBS_LIST`
- `STRATEGIES`
- `DATASET`
- `MEMORY_LIMIT_GB`
- `RESULT_ROOT`
- `DATASET_MOUNT_DIR`
- `EXTRA_TRAIN_ARGS`

## Result logs

Experiment logs now default to the mounted workspace path instead of HDFS:

- default root: `/mnt/bn/wyj-data0-hl/lqs/logs/Hetu-Galvatron/byted_experiments`
- one timestamped subdirectory per run
- override with `RESULT_ROOT=/your/path`

## Dataset placement

`train_dist_adacpsp.py` currently searches datasets under repo-relative
`varlen_datasets/` or a hard-coded local path. To make the ByteDance platform
workflow easier, `run_all_experiments.sh` can create a symlink:

- if `DATASET_MOUNT_DIR=/path/to/datasets` is set
- then `${PROJECT_DIR}/varlen_datasets -> ${DATASET_MOUNT_DIR}`

Place dataset text files there, e.g. `${DATASET_MOUNT_DIR}/wikipedia.txt`.

## Recommended entry command

```bash
bash byted_experiments/entrypoint.sh
```

## Notes

- One Merlin Seed job should correspond to exactly one GPU-count configuration.
- The default sweep assumes the current job uses all allocated GPUs.
- Do not pass multiple values in `GPU_CONFIGS` here.
- If you want to compare different total GPU counts on Merlin Seed, submit
  separate jobs with different resource requests.
