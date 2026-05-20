#!/usr/bin/env bash
set -euo pipefail

if [ -z "${MODEL_CKPT:-}" ]; then
  echo "MODEL_CKPT is required, e.g. MODEL_CKPT=/root/work/M3-ReID/ckptlog/BUPTCampus/.../modelckpt/model_best.pth"
  exit 1
fi
if [ ! -f "${MODEL_CKPT}" ]; then
  echo "MODEL_CKPT does not exist: ${MODEL_CKPT}"
  exit 1
fi

CONDA_ENV="${CONDA_ENV:-base}"
DATASET_DIR="${BUPT_DIR:-/root/work/BUPTCampus}"
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
PART_MATCH_WEIGHT="${PART_MATCH_WEIGHT:-0.0}"
PART_MATCH_WEIGHT_I2V="${PART_MATCH_WEIGHT_I2V:-}"
PART_MATCH_WEIGHT_V2I="${PART_MATCH_WEIGHT_V2I:-}"
PART_MATCH_NEIGHBOR_RADIUS="${PART_MATCH_NEIGHBOR_RADIUS:-1}"
PART_MATCH_SYMMETRIC="${PART_MATCH_SYMMETRIC:-1}"
PART_RERANK_TOPK="${PART_RERANK_TOPK:-0}"
PART_RERANK_TOPK_I2V="${PART_RERANK_TOPK_I2V:-}"
PART_RERANK_TOPK_V2I="${PART_RERANK_TOPK_V2I:-}"
PART_RERANK_WEIGHT="${PART_RERANK_WEIGHT:-0.0}"
PART_RERANK_WEIGHT_I2V="${PART_RERANK_WEIGHT_I2V:-}"
PART_RERANK_WEIGHT_V2I="${PART_RERANK_WEIGHT_V2I:-}"
PART_RERANK_NORM="${PART_RERANK_NORM:-zscore}"
RECIPROCAL_TOPK="${RECIPROCAL_TOPK:-0}"
RECIPROCAL_TOPK_I2V="${RECIPROCAL_TOPK_I2V:-}"
RECIPROCAL_TOPK_V2I="${RECIPROCAL_TOPK_V2I:-}"
RECIPROCAL_WEIGHT="${RECIPROCAL_WEIGHT:-0.0}"
RECIPROCAL_WEIGHT_I2V="${RECIPROCAL_WEIGHT_I2V:-}"
RECIPROCAL_WEIGHT_V2I="${RECIPROCAL_WEIGHT_V2I:-}"
MAX_EVAL_CLIPS="${MAX_EVAL_CLIPS:-0}"
EVAL_FP16="${EVAL_FP16:-1}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-0}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
DESC="${DESC:-SchemeH_multiclip_alpha${FUSION_ALPHA}_t10_buptcampus_v100}"

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
if [ "${PART_MATCH_SYMMETRIC}" = "1" ]; then
  EXTRA_ARGS+=(--part_match_symmetric)
else
  EXTRA_ARGS+=(--no-part_match_symmetric)
fi
if [ -n "${PART_MATCH_WEIGHT_I2V}" ]; then
  EXTRA_ARGS+=(--part_match_weight_i2v "${PART_MATCH_WEIGHT_I2V}")
fi
if [ -n "${PART_MATCH_WEIGHT_V2I}" ]; then
  EXTRA_ARGS+=(--part_match_weight_v2i "${PART_MATCH_WEIGHT_V2I}")
fi
if [ -n "${PART_RERANK_WEIGHT_I2V}" ]; then
  EXTRA_ARGS+=(--part_rerank_weight_i2v "${PART_RERANK_WEIGHT_I2V}")
fi
if [ -n "${PART_RERANK_WEIGHT_V2I}" ]; then
  EXTRA_ARGS+=(--part_rerank_weight_v2i "${PART_RERANK_WEIGHT_V2I}")
fi
if [ -n "${PART_RERANK_TOPK_I2V}" ]; then
  EXTRA_ARGS+=(--part_rerank_topk_i2v "${PART_RERANK_TOPK_I2V}")
fi
if [ -n "${PART_RERANK_TOPK_V2I}" ]; then
  EXTRA_ARGS+=(--part_rerank_topk_v2i "${PART_RERANK_TOPK_V2I}")
fi
if [ -n "${RECIPROCAL_TOPK_I2V}" ]; then
  EXTRA_ARGS+=(--reciprocal_topk_i2v "${RECIPROCAL_TOPK_I2V}")
fi
if [ -n "${RECIPROCAL_TOPK_V2I}" ]; then
  EXTRA_ARGS+=(--reciprocal_topk_v2i "${RECIPROCAL_TOPK_V2I}")
fi
if [ -n "${RECIPROCAL_WEIGHT_I2V}" ]; then
  EXTRA_ARGS+=(--reciprocal_weight_i2v "${RECIPROCAL_WEIGHT_I2V}")
fi
if [ -n "${RECIPROCAL_WEIGHT_V2I}" ]; then
  EXTRA_ARGS+=(--reciprocal_weight_v2i "${RECIPROCAL_WEIGHT_V2I}")
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

python test_m3reid.py \
  --dataset BUPTCampus \
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
  --part_match_weight "${PART_MATCH_WEIGHT}" \
  --part_match_neighbor_radius "${PART_MATCH_NEIGHBOR_RADIUS}" \
  --part_rerank_topk "${PART_RERANK_TOPK}" \
  --part_rerank_weight "${PART_RERANK_WEIGHT}" \
  --part_rerank_norm "${PART_RERANK_NORM}" \
  --reciprocal_topk "${RECIPROCAL_TOPK}" \
  --reciprocal_weight "${RECIPROCAL_WEIGHT}" \
  --eval_sample_mode all \
  --desc "${DESC}" \
  --gpu "${GPU}" \
  "${EXTRA_ARGS[@]}"

