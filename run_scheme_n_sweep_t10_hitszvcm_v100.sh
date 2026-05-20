#!/usr/bin/env bash
set -euo pipefail

if [ -z "${MODEL_CKPT:-}" ]; then
  echo "MODEL_CKPT is required, e.g. MODEL_CKPT=/root/work/M3-ReID/ckptlog/HITSZVCM/.../modelckpt/model_best.pth"
  exit 1
fi

SWEEP_CONFIGS="${SWEEP_CONFIGS:-20:0.022:0.015 20:0.025:0.015 20:0.020:0.018 20:0.022:0.018}"

for config in ${SWEEP_CONFIGS}; do
  IFS=: read -r reciprocal_topk i2v_weight v2i_weight <<< "${config}"
  export RECIPROCAL_TOPK="${reciprocal_topk}"
  export RECIPROCAL_WEIGHT_I2V="${i2v_weight}"
  export RECIPROCAL_WEIGHT_V2I="${v2i_weight}"
  export DESC="SchemeN_sweep_rk${reciprocal_topk}_i2v${i2v_weight}_v2i${v2i_weight}_t10_hitszvcm_v100"
  bash ./run_scheme_n_eval_t10_hitszvcm_v100.sh
done
