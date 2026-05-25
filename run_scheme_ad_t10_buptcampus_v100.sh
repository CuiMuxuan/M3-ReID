#!/usr/bin/env bash
set -euo pipefail

export M3PLUS_MODE="${M3PLUS_MODE:-detached_direction_aware_quad_calibrated_fusion}"
export DESC="${DESC:-SchemeADNoBoT_detached_direction_quad_fromAA_fa0.08_pa0.015_ca0.025_rw0.05_pgw004_lr3e7_t10_buptcampus_v100_bs16}"

bash ./run_scheme_ac_t10_buptcampus_v100.sh
