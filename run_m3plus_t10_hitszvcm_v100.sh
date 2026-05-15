#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-temp_311}"
DATASET_DIR="${HITSZ_DIR:-/data/datasets/HITSZ-VCM}"
WORKERS="${WORKERS:-8}"
GPU="${GPU:-0}"
LOG_DIR="${LOG_DIR:-run_logs}"
P_NUM="${P_NUM:-8}"
K_NUM="${K_NUM:-2}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-32}"
ACCUM_STEPS="${ACCUM_STEPS:-2}"
M3PLUS_AUG_STRENGTH="${M3PLUS_AUG_STRENGTH:-mild}"
TRIPLET_WEIGHT="${TRIPLET_WEIGHT:-0.5}"
TRIPLET_FRAME_WEIGHT="${TRIPLET_FRAME_WEIGHT:-0.1}"
DESC="${DESC:-M3Plus_t10_v100_bs$((P_NUM * K_NUM))_acc${ACCUM_STEPS}}"

mkdir -p "${LOG_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

python train_m3reid.py \
  --dataset HITSZVCM \
  --dataset_dir "${DATASET_DIR}" \
  --t 10 \
  --img_h 288 --img_w 144 \
  --p_num "${P_NUM}" --k_num "${K_NUM}" \
  --test_batch_size "${TEST_BATCH_SIZE}" \
  --workers "${WORKERS}" \
  --persistent_workers \
  --prefetch_factor 4 \
  --pin_memory \
  --non_blocking \
  --cudnn_benchmark \
  --lr 0.0002 --wd 0.0005 \
  --accum_steps "${ACCUM_STEPS}" \
  --fp16 \
  --use_m3plus \
  --m3plus_aug_strength "${M3PLUS_AUG_STRENGTH}" \
  --triplet_weight "${TRIPLET_WEIGHT}" \
  --triplet_frame_weight "${TRIPLET_FRAME_WEIGHT}" \
  --epochs 200 \
  --log_interval 20 \
  --test_interval 5 \
  --save_interval 10 \
  --desc "${DESC}" \
  --gpu "${GPU}" \
  2>&1 | tee "${LOG_DIR}/m3plus_t10_hitszvcm_v100_$(date +%Y%m%d_%H%M%S).log"
