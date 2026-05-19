#!/usr/bin/env bash
set -euo pipefail

export M3PLUS_MODE="${M3PLUS_MODE:-dual_fusion}"
export FUSION_ALPHA="${FUSION_ALPHA:-0.01}"
export PART_MATCH_WEIGHT="${PART_MATCH_WEIGHT:-0.0}"
export PART_MATCH_WEIGHT_I2V="${PART_MATCH_WEIGHT_I2V:-0.02}"
export PART_MATCH_WEIGHT_V2I="${PART_MATCH_WEIGHT_V2I:-0.0}"
export PART_MATCH_NEIGHBOR_RADIUS="${PART_MATCH_NEIGHBOR_RADIUS:-1}"
export PART_MATCH_SYMMETRIC="${PART_MATCH_SYMMETRIC:-1}"
export DESC="${DESC:-SchemeL_partmatch_dir_schemeD_alpha${FUSION_ALPHA}_i2v${PART_MATCH_WEIGHT_I2V}_v2i${PART_MATCH_WEIGHT_V2I}_t10_hitszvcm_v100}"

bash ./run_scheme_h_eval_t10_hitszvcm_v100.sh
