#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-base}"
DATASET_DIR="${BUPT_DIR:-/root/work/BUPTCampus}"
WORKERS="${WORKERS:-2}"
GPU="${GPU:-0}"
LOG_DIR="${LOG_DIR:-run_logs}"
P_NUM="${P_NUM:-4}"
K_NUM="${K_NUM:-4}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-32}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"
MVL_NUM_HEADS="${MVL_NUM_HEADS:-2}"
PART_NUM="${PART_NUM:-4}"
PART_DIM="${PART_DIM:-2048}"
FEATURE_DROPOUT="${FEATURE_DROPOUT:-0.0}"
TRIPLET_WEIGHT="${TRIPLET_WEIGHT:-0.35}"
TRIPLET_FRAME_WEIGHT="${TRIPLET_FRAME_WEIGHT:-0.10}"
ID_LABEL_SMOOTHING="${ID_LABEL_SMOOTHING:-0.05}"
OPTIMIZER="${OPTIMIZER:-adam}"
GRAD_CHECKPOINT_HEAD="${GRAD_CHECKPOINT_HEAD:-0}"
EPOCHS="${EPOCHS:-130}"
EVAL_START_EPOCH="${EVAL_START_EPOCH:-60}"
TEST_INTERVAL="${TEST_INTERVAL:-5}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"
DESC="${DESC:-SchemeA_partonly_t10_buptcampus_v100_bs$((P_NUM * K_NUM))}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-0}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
TORCH_SHARING_STRATEGY="${TORCH_SHARING_STRATEGY:-file_system}"

EXTRA_ARGS=()
if [ "${GRAD_CHECKPOINT_HEAD}" = "1" ]; then
  EXTRA_ARGS+=(--grad_checkpoint_head)
fi
if [ "${PERSISTENT_WORKERS}" = "1" ]; then
  EXTRA_ARGS+=(--persistent_workers)
else
  EXTRA_ARGS+=(--no-persistent_workers)
fi

mkdir -p "${LOG_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

python train_m3reid.py \
  --dataset BUPTCampus \
  --dataset_dir "${DATASET_DIR}" \
  --t 10 \
  --img_h 288 --img_w 144 \
  --p_num "${P_NUM}" --k_num "${K_NUM}" \
  --test_batch_size "${TEST_BATCH_SIZE}" \
  --workers "${WORKERS}" \
  --prefetch_factor "${PREFETCH_FACTOR}" \
  --pin_memory \
  --non_blocking \
  --torch_sharing_strategy "${TORCH_SHARING_STRATEGY}" \
  --cudnn_benchmark \
  --lr 0.0001 --wd 0.0005 \
  --optimizer "${OPTIMIZER}" \
  --accum_steps "${ACCUM_STEPS}" \
  --fp16 \
  --eval_fp16 \
  --use_m3plus \
  --m3plus_mode part_only \
  --m3plus_aug_strength none \
  --mvl_num_heads "${MVL_NUM_HEADS}" \
  --part_num "${PART_NUM}" \
  --part_dim "${PART_DIM}" \
  --feature_dropout "${FEATURE_DROPOUT}" \
  --sample_method identity_cross_modality \
  --triplet_weight "${TRIPLET_WEIGHT}" \
  --triplet_frame_weight "${TRIPLET_FRAME_WEIGHT}" \
  --id_label_smoothing "${ID_LABEL_SMOOTHING}" \
  --epochs "${EPOCHS}" \
  --log_interval 20 \
  --eval_start_epoch "${EVAL_START_EPOCH}" \
  --test_interval "${TEST_INTERVAL}" \
  --early_stop_patience "${EARLY_STOP_PATIENCE}" \
  --save_interval 10 \
  --desc "${DESC}" \
  --gpu "${GPU}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${LOG_DIR}/scheme_a_t10_buptcampus_v100_$(date +%Y%m%d_%H%M%S).log"
