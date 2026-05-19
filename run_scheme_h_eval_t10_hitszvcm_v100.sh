#!/usr/bin/env bash
set -euo pipefail

if [ -z "${MODEL_CKPT:-}" ]; then
  echo "MODEL_CKPT is required, e.g. MODEL_CKPT=/root/work/M3-ReID/ckptlog/HITSZVCM/.../modelckpt/model_best.pth"
  exit 1
fi
if [ ! -f "${MODEL_CKPT}" ]; then
  echo "MODEL_CKPT does not exist: ${MODEL_CKPT}"
  exit 1
fi

CONDA_ENV="${CONDA_ENV:-base}"
DATASET_DIR="${HITSZ_DIR:-/root/work/HITSZ-VCM}"
WORKERS="${WORKERS:-2}"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
PART_NUM="${PART_NUM:-4}"
PART_DIM="${PART_DIM:-2048}"
MVL_NUM_HEADS="${MVL_NUM_HEADS:-2}"
FEATURE_DROPOUT="${FEATURE_DROPOUT:-0.0}"
M3PLUS_MODE="${M3PLUS_MODE:-dual_fusion}"
TEMPORAL_DIM="${TEMPORAL_DIM:-256}"
TEMPORAL_DROPOUT="${TEMPORAL_DROPOUT:-0.0}"
FUSION_ALPHA="${FUSION_ALPHA:-0.05}"
ADAPTIVE_GATE_MIN="${ADAPTIVE_GATE_MIN:-0.0}"
ADAPTIVE_GATE_MAX="${ADAPTIVE_GATE_MAX:-0.10}"
MAX_EVAL_CLIPS="${MAX_EVAL_CLIPS:-0}"
EVAL_FP16="${EVAL_FP16:-1}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-0}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
DESC="${DESC:-SchemeH_multiclip_alpha${FUSION_ALPHA}_t10_hitszvcm_v100}"

EXTRA_ARGS=()
if [ "${EVAL_FP16}" = "1" ]; then
  EXTRA_ARGS+=(--eval_fp16)
fi
if [ "${PERSISTENT_WORKERS}" = "1" ]; then
  EXTRA_ARGS+=(--persistent_workers)
else
  EXTRA_ARGS+=(--no-persistent_workers)
fi
if [ "${MAX_EVAL_CLIPS}" != "0" ]; then
  EXTRA_ARGS+=(--max_eval_clips "${MAX_EVAL_CLIPS}")
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

python test_m3reid.py \
  --dataset HITSZVCM \
  --dataset_dir "${DATASET_DIR}" \
  --t 10 \
  --img_h 288 --img_w 144 \
  --batch_size "${BATCH_SIZE}" \
  --workers "${WORKERS}" \
  --prefetch_factor "${PREFETCH_FACTOR}" \
  --pin_memory \
  --non_blocking \
  --cudnn_benchmark \
  --resume "${MODEL_CKPT}" \
  --use_m3plus \
  --m3plus_mode "${M3PLUS_MODE}" \
  --mvl_num_heads "${MVL_NUM_HEADS}" \
  --part_num "${PART_NUM}" \
  --part_dim "${PART_DIM}" \
  --feature_dropout "${FEATURE_DROPOUT}" \
  --temporal_dim "${TEMPORAL_DIM}" \
  --temporal_dropout "${TEMPORAL_DROPOUT}" \
  --fusion_alpha "${FUSION_ALPHA}" \
  --adaptive_gate_min "${ADAPTIVE_GATE_MIN}" \
  --adaptive_gate_max "${ADAPTIVE_GATE_MAX}" \
  --eval_sample_mode all \
  --desc "${DESC}" \
  --gpu "${GPU}" \
  "${EXTRA_ARGS[@]}"
