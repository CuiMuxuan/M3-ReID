#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-temp_311}"
DATASET_DIR="${BUPT_DIR:-/data/datasets/BUPTCampus}"
WORKERS="${WORKERS:-8}"
GPU="${GPU:-0}"
LOG_DIR="${LOG_DIR:-run_logs}"

mkdir -p "${LOG_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

python train_m3reid.py \
  --dataset BUPTCampus \
  --dataset_dir "${DATASET_DIR}" \
  --t 10 \
  --img_h 288 --img_w 144 \
  --p_num 4 --k_num 8 \
  --test_batch_size 32 \
  --workers "${WORKERS}" \
  --persistent_workers \
  --prefetch_factor 4 \
  --pin_memory \
  --non_blocking \
  --cudnn_benchmark \
  --lr 0.0001 --wd 0.0005 \
  --accum_steps 1 \
  --fp16 \
  --use_m3plus \
  --triplet_weight 0.5 \
  --triplet_frame_weight 0.25 \
  --epochs 200 \
  --log_interval 20 \
  --test_interval 5 \
  --save_interval 10 \
  --desc M3Plus_t10_v100_bs32 \
  --gpu "${GPU}" \
  2>&1 | tee "${LOG_DIR}/m3plus_t10_buptcampus_v100_$(date +%Y%m%d_%H%M%S).log"
