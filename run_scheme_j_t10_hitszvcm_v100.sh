#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-base}"
DATASET_DIR="${HITSZ_DIR:-/root/work/HITSZ-VCM}"
BASELINE_CKPT="${BASELINE_CKPT:-/root/work/M3-ReID/ckptlog/HITSZVCM/Time-2026-05-18_01-27-50_SchemeD_dualfusion_t10_hitszvcm_v100_bs16/modelckpt/model_best.pth}"
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
FUSION_ALPHA="${FUSION_ALPHA:-0.01}"
PART_LR_MULT="${PART_LR_MULT:-4.0}"
FREEZE_BASE_EPOCHS="${FREEZE_BASE_EPOCHS:-0}"
LR="${LR:-0.000003}"
TRIPLET_WEIGHT="${TRIPLET_WEIGHT:-0.06}"
TRIPLET_FRAME_WEIGHT="${TRIPLET_FRAME_WEIGHT:-0.10}"
PART_ID_WEIGHT="${PART_ID_WEIGHT:-0.15}"
PART_TRIPLET_WEIGHT="${PART_TRIPLET_WEIGHT:-0.08}"
PART_MMA_WEIGHT="${PART_MMA_WEIGHT:-0.01}"
COSFACE_WEIGHT="${COSFACE_WEIGHT:-0.20}"
COSFACE_FRAME_WEIGHT="${COSFACE_FRAME_WEIGHT:-0.25}"
PART_COSFACE_WEIGHT="${PART_COSFACE_WEIGHT:-0.03}"
COSFACE_MARGIN="${COSFACE_MARGIN:-0.18}"
COSFACE_SCALE="${COSFACE_SCALE:-32.0}"
COSFACE_START_EPOCH="${COSFACE_START_EPOCH:-1}"
ID_LABEL_SMOOTHING="${ID_LABEL_SMOOTHING:-0.0}"
OPTIMIZER="${OPTIMIZER:-adam}"
GRAD_CHECKPOINT_HEAD="${GRAD_CHECKPOINT_HEAD:-0}"
EPOCHS="${EPOCHS:-55}"
EVAL_START_EPOCH="${EVAL_START_EPOCH:-5}"
TEST_INTERVAL="${TEST_INTERVAL:-5}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-8}"
LR_MILESTONES="${LR_MILESTONES:-30,45}"
DESC="${DESC:-SchemeJ_cosface_dualfusion_t10_hitszvcm_v100_bs$((P_NUM * K_NUM))}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-0}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
TORCH_SHARING_STRATEGY="${TORCH_SHARING_STRATEGY:-file_system}"

if [ ! -f "${BASELINE_CKPT}" ]; then
  echo "BASELINE_CKPT does not exist: ${BASELINE_CKPT}"
  exit 1
fi

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
  --dataset HITSZVCM \
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
  --lr "${LR}" --wd 0.0005 \
  --optimizer "${OPTIMIZER}" \
  --accum_steps "${ACCUM_STEPS}" \
  --lr_milestones "${LR_MILESTONES}" \
  --fp16 \
  --eval_fp16 \
  --resume "${BASELINE_CKPT}" \
  --use_m3plus \
  --m3plus_mode dual_fusion \
  --m3plus_aug_strength none \
  --mvl_num_heads "${MVL_NUM_HEADS}" \
  --part_num "${PART_NUM}" \
  --part_dim "${PART_DIM}" \
  --feature_dropout "${FEATURE_DROPOUT}" \
  --fusion_alpha "${FUSION_ALPHA}" \
  --part_lr_mult "${PART_LR_MULT}" \
  --freeze_base_epochs "${FREEZE_BASE_EPOCHS}" \
  --sample_method identity_cross_modality \
  --triplet_weight "${TRIPLET_WEIGHT}" \
  --triplet_frame_weight "${TRIPLET_FRAME_WEIGHT}" \
  --part_id_weight "${PART_ID_WEIGHT}" \
  --part_triplet_weight "${PART_TRIPLET_WEIGHT}" \
  --part_mma_weight "${PART_MMA_WEIGHT}" \
  --cosface_weight "${COSFACE_WEIGHT}" \
  --cosface_frame_weight "${COSFACE_FRAME_WEIGHT}" \
  --part_cosface_weight "${PART_COSFACE_WEIGHT}" \
  --cosface_margin "${COSFACE_MARGIN}" \
  --cosface_scale "${COSFACE_SCALE}" \
  --cosface_start_epoch "${COSFACE_START_EPOCH}" \
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
  2>&1 | tee "${LOG_DIR}/scheme_j_t10_hitszvcm_v100_$(date +%Y%m%d_%H%M%S).log"
