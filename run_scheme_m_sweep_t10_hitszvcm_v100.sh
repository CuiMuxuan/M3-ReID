#!/usr/bin/env bash
set -euo pipefail

if [ -z "${MODEL_CKPT:-}" ]; then
  echo "MODEL_CKPT is required, e.g. MODEL_CKPT=/root/work/M3-ReID/ckptlog/HITSZVCM/.../modelckpt/model_best.pth"
  exit 1
fi

SWEEP_CONFIGS="${SWEEP_CONFIGS:-10:0.015:0.015 20:0.010:0.015 20:0.020:0.015 20:0.015:0.020 30:0.015:0.015}"

for config in ${SWEEP_CONFIGS}; do
  IFS=: read -r topk i2v_weight v2i_weight <<< "${config}"
  export PART_RERANK_TOPK="${topk}"
  export PART_RERANK_WEIGHT_I2V="${i2v_weight}"
  export PART_RERANK_WEIGHT_V2I="${v2i_weight}"
  export DESC="SchemeM_sweep_top${topk}_i2v${i2v_weight}_v2i${v2i_weight}_t10_hitszvcm_v100"
  bash ./run_scheme_m_eval_t10_hitszvcm_v100.sh
done
