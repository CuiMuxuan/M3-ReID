#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-base}"
DATASET_DIR="${BUPT_DIR:-/root/work/BUPTCampus}"
BASELINE_CKPT="${BASELINE_CKPT:-/root/work/M3-ReID/ckptlog/BUPTCampus/Time-2026-05-24_14-05-46_SchemeW5NoBoT_dual_calibrated_gate075_gw0005_alpha0.08_cal0.02_fz12_t10_buptcampus_v100_bs16_from_nobot/modelckpt/model_best.pth}"
WORKERS="${WORKERS:-2}"
GPU="${GPU:-0}"
LOG_DIR="${LOG_DIR:-run_logs}"
P_NUM="${P_NUM:-4}"
K_NUM="${K_NUM:-4}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-16}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"
MVL_NUM_HEADS="${MVL_NUM_HEADS:-2}"
PART_NUM="${PART_NUM:-4}"
PART_DIM="${PART_DIM:-2048}"
FEATURE_DROPOUT="${FEATURE_DROPOUT:-0.0}"
FUSION_ALPHA="${FUSION_ALPHA:-0.08}"
PROJECTION_ALPHA="${PROJECTION_ALPHA:-0.010}"
CALIBRATION_ALPHA="${CALIBRATION_ALPHA:-0.025}"
PART_LR_MULT="${PART_LR_MULT:-4.0}"
FREEZE_BASE_EPOCHS="${FREEZE_BASE_EPOCHS:-12}"
LR="${LR:-0.0000008}"
TRIPLET_WEIGHT="${TRIPLET_WEIGHT:-0.03}"
TRIPLET_FRAME_WEIGHT="${TRIPLET_FRAME_WEIGHT:-0.06}"
PART_ID_WEIGHT="${PART_ID_WEIGHT:-0.08}"
PART_TRIPLET_WEIGHT="${PART_TRIPLET_WEIGHT:-0.04}"
PART_MMA_WEIGHT="${PART_MMA_WEIGHT:-0.005}"
PROJECTION_ID_WEIGHT="${PROJECTION_ID_WEIGHT:-0.012}"
PROJECTION_TRIPLET_WEIGHT="${PROJECTION_TRIPLET_WEIGHT:-0.006}"
PROJECTION_MMA_WEIGHT="${PROJECTION_MMA_WEIGHT:-0.0005}"
PROJECTION_GATE_WEIGHT="${PROJECTION_GATE_WEIGHT:-0.004}"
PROJECTION_GATE_IR_TARGET="${PROJECTION_GATE_IR_TARGET:-0.65}"
PROJECTION_GATE_RGB_TARGET="${PROJECTION_GATE_RGB_TARGET:-0.35}"
CALIBRATION_ID_WEIGHT="${CALIBRATION_ID_WEIGHT:-0.02}"
CALIBRATION_TRIPLET_WEIGHT="${CALIBRATION_TRIPLET_WEIGHT:-0.01}"
CALIBRATION_MMA_WEIGHT="${CALIBRATION_MMA_WEIGHT:-0.001}"
CALIBRATION_GATE_WEIGHT="${CALIBRATION_GATE_WEIGHT:-0.005}"
CALIBRATION_GATE_TARGET="${CALIBRATION_GATE_TARGET:-0.75}"
ID_LABEL_SMOOTHING="${ID_LABEL_SMOOTHING:-0.0}"
OPTIMIZER="${OPTIMIZER:-adam}"
GRAD_CHECKPOINT_HEAD="${GRAD_CHECKPOINT_HEAD:-0}"
EPOCHS="${EPOCHS:-20}"
EVAL_START_EPOCH="${EVAL_START_EPOCH:-5}"
TEST_INTERVAL="${TEST_INTERVAL:-5}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-3}"
LR_MILESTONES="${LR_MILESTONES:-30,35}"
DESC="${DESC:-SchemeAANoBoT_quad_calibrated_fromW5_pa${PROJECTION_ALPHA}_fa${FUSION_ALPHA}_ca${CALIBRATION_ALPHA}_t10_buptcampus_v100_bs$((P_NUM * K_NUM))}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-0}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
TORCH_SHARING_STRATEGY="${TORCH_SHARING_STRATEGY:-file_system}"
DROP_CLASSIFIER_ON_RESUME="${DROP_CLASSIFIER_ON_RESUME:-0}"

if [ ! -f "${BASELINE_CKPT}" ]; then
  echo "BASELINE_CKPT does not exist: ${BASELINE_CKPT}"
  exit 1
fi

EXTRA_ARGS=()
if [ "${GRAD_CHECKPOINT_HEAD}" = "1" ]; then
  EXTRA_ARGS+=(--grad_checkpoint_head)
fi
if [ "${DROP_CLASSIFIER_ON_RESUME}" = "1" ]; then
  EXTRA_ARGS+=(--drop_classifier_on_resume)
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
  --lr "${LR}" --wd 0.0005 \
  --optimizer "${OPTIMIZER}" \
  --accum_steps "${ACCUM_STEPS}" \
  --lr_milestones "${LR_MILESTONES}" \
  --fp16 \
  --eval_fp16 \
  --resume "${BASELINE_CKPT}" \
  --use_m3plus \
  --m3plus_mode quad_calibrated_fusion \
  --m3plus_aug_strength none \
  --mvl_num_heads "${MVL_NUM_HEADS}" \
  --part_num "${PART_NUM}" \
  --part_dim "${PART_DIM}" \
  --feature_dropout "${FEATURE_DROPOUT}" \
  --fusion_alpha "${FUSION_ALPHA}" \
  --projection_alpha "${PROJECTION_ALPHA}" \
  --calibration_alpha "${CALIBRATION_ALPHA}" \
  --part_lr_mult "${PART_LR_MULT}" \
  --freeze_base_epochs "${FREEZE_BASE_EPOCHS}" \
  --sample_method identity_cross_modality \
  --triplet_weight "${TRIPLET_WEIGHT}" \
  --triplet_frame_weight "${TRIPLET_FRAME_WEIGHT}" \
  --part_id_weight "${PART_ID_WEIGHT}" \
  --part_triplet_weight "${PART_TRIPLET_WEIGHT}" \
  --part_mma_weight "${PART_MMA_WEIGHT}" \
  --projection_id_weight "${PROJECTION_ID_WEIGHT}" \
  --projection_triplet_weight "${PROJECTION_TRIPLET_WEIGHT}" \
  --projection_mma_weight "${PROJECTION_MMA_WEIGHT}" \
  --projection_gate_weight "${PROJECTION_GATE_WEIGHT}" \
  --projection_gate_ir_target "${PROJECTION_GATE_IR_TARGET}" \
  --projection_gate_rgb_target "${PROJECTION_GATE_RGB_TARGET}" \
  --calibration_id_weight "${CALIBRATION_ID_WEIGHT}" \
  --calibration_triplet_weight "${CALIBRATION_TRIPLET_WEIGHT}" \
  --calibration_mma_weight "${CALIBRATION_MMA_WEIGHT}" \
  --calibration_gate_weight "${CALIBRATION_GATE_WEIGHT}" \
  --calibration_gate_target "${CALIBRATION_GATE_TARGET}" \
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
  2>&1 | tee "${LOG_DIR}/scheme_aa_t10_buptcampus_v100_$(date +%Y%m%d_%H%M%S).log"
