#!/usr/bin/env bash
set -euo pipefail

if [ -z "${MODEL_CKPT:-}" ]; then
  echo "MODEL_CKPT is required, e.g. MODEL_CKPT=/root/work/M3-ReID/ckptlog/BUPTCampus/.../modelckpt/model_best.pth"
  exit 1
fi

SWEEP_CONFIGS="${SWEEP_CONFIGS:-0.01:20:0.025:0.022 0.05:20:0.025:0.022 0.10:20:0.025:0.022 0.20:20:0.025:0.022}"

for config in ${SWEEP_CONFIGS}; do
  IFS=: read -r fusion_alpha reciprocal_topk i2v_weight v2i_weight <<< "${config}"
  export FUSION_ALPHA="${fusion_alpha}"
  export RECIPROCAL_TOPK="${reciprocal_topk}"
  export RECIPROCAL_WEIGHT_I2V="${i2v_weight}"
  export RECIPROCAL_WEIGHT_V2I="${v2i_weight}"
  export DESC="SchemeN_sweep_alpha${fusion_alpha}_rk${reciprocal_topk}_i2v${i2v_weight}_v2i${v2i_weight}_t10_buptcampus_v100"
  bash ./run_scheme_n_eval_t10_buptcampus_v100.sh
done
