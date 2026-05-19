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
TEMPORAL_DIM="${TEMPORAL_DIM:-256}"
TEMPORAL_DROPOUT="${TEMPORAL_DROPOUT:-0.05}"
FEATURE_DROPOUT="${FEATURE_DROPOUT:-0.0}"
FUSION_ALPHA="${FUSION_ALPHA:-0.01}"
PART_LR_MULT="${PART_LR_MULT:-8.0}"
TEMPORAL_LR_MULT="${TEMPORAL_LR_MULT:-40.0}"
FREEZE_BASE_EPOCHS="${FREEZE_BASE_EPOCHS:-8}"
LR="${LR:-0.000005}"
TRIPLET_WEIGHT="${TRIPLET_WEIGHT:-0.08}"
TRIPLET_FRAME_WEIGHT="${TRIPLET_FRAME_WEIGHT:-0.10}"
PART_ID_WEIGHT="${PART_ID_WEIGHT:-0.25}"
PART_TRIPLET_WEIGHT="${PART_TRIPLET_WEIGHT:-0.12}"
PART_MMA_WEIGHT="${PART_MMA_WEIGHT:-0.02}"
ID_LABEL_SMOOTHING="${ID_LABEL_SMOOTHING:-0.03}"
OPTIMIZER="${OPTIMIZER:-adam}"
GRAD_CHECKPOINT_HEAD="${GRAD_CHECKPOINT_HEAD:-0}"
EPOCHS="${EPOCHS:-70}"
EVAL_START_EPOCH="${EVAL_START_EPOCH:-5}"
TEST_INTERVAL="${TEST_INTERVAL:-5}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"
LR_MILESTONES="${LR_MILESTONES:-35,55}"
DESC="${DESC:-SchemeI_temporal_dualfusion_t10_hitszvcm_v100_bs$((P_NUM * K_NUM))}"
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
  --m3plus_mode temporal_dual_fusion \
  --m3plus_aug_strength none \
  --mvl_num_heads "${MVL_NUM_HEADS}" \
  --part_num "${PART_NUM}" \
  --part_dim "${PART_DIM}" \
  --temporal_dim "${TEMPORAL_DIM}" \
  --temporal_dropout "${TEMPORAL_DROPOUT}" \
  --temporal_lr_mult "${TEMPORAL_LR_MULT}" \
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
  2>&1 | tee "${LOG_DIR}/scheme_i_t10_hitszvcm_v100_$(date +%Y%m%d_%H%M%S).log"
