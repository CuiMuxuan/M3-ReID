# M3Plus Long-Term Execution Plan

## Goal

Use the 10-frame task as the main track and improve both BUPTCampus and HITSZ-VCM over the provided M3-ReID baseline logs by at least 2 percentage points.

Primary acceptance metric: Rank-1 in both retrieval directions.
Secondary acceptance metric: mAP in both retrieval directions.

## Baseline From Provided Logs

| Dataset | Frames | Best Epoch | Direction | Rank-1 | mAP | Target Rank-1 | Target mAP |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| BUPTCampus | 10 | 176 | i2v | 73.51 | 68.39 | 75.51 | 70.39 |
| BUPTCampus | 10 | 176 | v2i | 73.89 | 66.65 | 75.89 | 68.65 |
| HITSZ-VCM | 10 | 185 | i2v | 73.58 | 60.75 | 75.58 | 62.75 |
| HITSZ-VCM | 10 | 185 | v2i | 77.00 | 63.27 | 79.00 | 65.27 |

The 10-frame track is selected because it is stronger than 6 frames on BUPTCampus and gives a better HITSZ-VCM v2i result in the logs.

## Implemented Method

The implementation is exposed through `--use_m3plus` while keeping the original baseline path available when the flag is omitted.

Changed modules:

| Area | File | Change |
| --- | --- | --- |
| Model | `models/modules/enhancement.py` | Added multi-scale residual fusion, lightweight channel-spatial attention, and part-guided local aggregation. |
| Model | `models/model_m3reid.py` | Added optional M3Plus feature path and concatenated local part descriptor. |
| Loss | `losses/metric_loss.py` | Added cross-modality batch-hard triplet loss. |
| Loss | `losses/mma_loss.py` | Made invalid-batch fallback return a tensor for stable mixed precision training. |
| Data | `data/transform.py` | Added weak low-light and block occlusion augmentations. |
| Entry | `train_m3reid.py` | Added `--t`, `--use_m3plus`, triplet hyperparameters, cross-modality sampler default, and M3Plus augmentation path. |
| Entry | `test_m3reid.py` | Added `--t` and `--use_m3plus` for matching checkpoints. |

Core idea:

1. Multi-scale residual fusion reuses layer3 detail cues before final MVL pooling.
2. Lightweight channel-spatial attention suppresses background and modality-specific noise.
3. Part-guided local aggregation provides horizontal body-part descriptors, a common strong ReID improvement.
4. Cross-modality batch-hard triplet loss directly optimizes hard visible-infrared positive/negative pairs.
5. Weak low-light and modest occlusion augmentation improve robustness without making data augmentation the only contribution.

## Environment

Use the requested Conda environment:

```bash
conda activate temp_311
pip install -r requirements.txt
```

Verified locally:

```text
Python 3.11.5
torch 2.11.0+cu130
torchvision 0.26.0+cu130
CUDA available: True
```

No extra runtime dependency was added beyond the existing `requirements.txt`.

## V100 32GB Linux Server Runbook

The remote Ubuntu 22.04 / V100 32GB path should use the Linux scripts instead of the local Windows smoke scripts. They keep the 10-frame M3Plus method, use a safe physical training batch of `p_num * k_num = 16`, keep the effective batch at 32 through `--accum_steps 2`, enable AMP, and tune the DataLoader for server I/O.

Default server dataset paths:

```bash
/data/datasets/BUPTCampus
/data/datasets/HITSZ-VCM
```

First server run:

```bash
git pull
chmod +x run_m3plus_t10_buptcampus_v100.sh run_m3plus_t10_hitszvcm_v100.sh
./run_m3plus_t10_buptcampus_v100.sh
./run_m3plus_t10_hitszvcm_v100.sh
```

Override paths or worker count when needed:

```bash
BUPT_DIR=/your/path/BUPTCampus WORKERS=12 ./run_m3plus_t10_buptcampus_v100.sh
HITSZ_DIR=/your/path/HITSZ-VCM WORKERS=12 ./run_m3plus_t10_hitszvcm_v100.sh
```

The V100 observed OOM at physical batch 24/32, while batch 16 used about 26.9GB / 32GB. Keep physical batch 16 for stable full runs. After the first HITSZ-VCM bs16 run plateaued below baseline, the HITSZ server script now uses `P_NUM=8 K_NUM=2 ACCUM_STEPS=2` by default: it keeps the same memory footprint but gives the batch-hard metric loss more identities and hard negatives. The scripts also default to `M3PLUS_AUG_STRENGTH=mild` and `TRIPLET_FRAME_WEIGHT=0.1` to reduce over-regularization from the extra weak-light/occlusion and frame-level hard mining.

To probe a little more GPU memory without changing code, try physical batch 18 once:

```bash
HITSZ_DIR=/root/work/HITSZ-VCM P_NUM=3 K_NUM=6 ACCUM_STEPS=2 ./run_m3plus_t10_hitszvcm_v100.sh
```

If it OOMs or fragments memory after evaluation, return to the default batch 16. Avoid `K_NUM=5` because the cross-modality identity sampler is cleaner with even samples per identity.

First failed HITSZ-VCM bs16 note:

```text
Run: M3Plus_t10_v100_bs16_nofile
Setting: P_NUM=4, K_NUM=4, ACCUM_STEPS=1, M3Plus standard augmentation
Best average Rank-1: Epoch 105, i2v 70.46 / 58.73 mAP, v2i 73.94 / 60.72 mAP
Conclusion: below the 10-frame baseline and target; do not continue this exact configuration.
```

For long sessions, run inside `tmux` or `screen`; each script also writes a console log under `run_logs/`, while the training script writes checkpoints and TensorBoard files under `ckptlog/`.

## Main Training Commands

BUPTCampus 10-frame M3Plus:

```bash
powershell -ExecutionPolicy Bypass -File .\run_m3plus_t10_buptcampus.ps1
```

HITSZ-VCM 10-frame M3Plus:

```bash
powershell -ExecutionPolicy Bypass -File .\run_m3plus_t10_hitszvcm.ps1
```

Evaluation must use the same `--t 10 --use_m3plus` switches:

```bash
conda run -n temp_311 python test_m3reid.py ^
  --dataset BUPTCampus ^
  --dataset_dir C:\baidunetdiskdownload\BUPTCampus ^
  --t 10 ^
  --batch_size 4 ^
  --use_m3plus ^
  --resume ckptlog/BUPTCampus/Time-XXXX_M3Plus_t10/modelckpt/model_epoch-XXX.pth ^
  --desc M3Plus_t10_eval ^
  --gpu 0
```

## Execution Schedule

1. Smoke test each dataset for 1 epoch with `--save_interval 1 --test_interval 1` on a short debug run, confirming no loader, CUDA, or checkpoint shape errors.
2. Run full BUPTCampus 10-frame training for 200 epochs.
3. Select the best checkpoint by average of i2v/v2i Rank-1. If Rank-1 ties, use average mAP.
4. Run full HITSZ-VCM 10-frame training with the same M3Plus settings.
5. Repeat each full run for seeds `0, 1, 2` only after the first seed crosses the target or is within 0.5 points.
6. Report mean and best metrics against the baseline table above.

Smoke-test command pattern:

```bash
conda run -n temp_311 python train_m3reid.py ^
  --dataset BUPTCampus ^
  --dataset_dir C:\baidunetdiskdownload\BUPTCampus ^
  --t 10 ^
  --epochs 1 ^
  --max_train_batches 2 ^
  --p_num 2 --k_num 2 ^
  --accum_steps 8 ^
  --test_batch_size 4 ^
  --test_interval 1 ^
  --save_interval 1 ^
  --fp16 ^
  --use_m3plus ^
  --desc M3Plus_t10_smoke ^
  --gpu 0
```

## If First Full Runs Miss The Target

Try these changes in order, one at a time:

| Priority | Change | Command Adjustment | Expected Effect |
| ---: | --- | --- | --- |
| 1 | Stronger hard metric term | `--triplet_weight 0.75` | Better Rank-1 from harder cross-modal separation. |
| 2 | More frame-level hard mining | `--triplet_frame_weight 0.5` | Helps short/local motion cues, especially BUPTCampus. |
| 3 | Slightly smoother ID loss | `--id_label_smoothing 0.15` | Reduces overfitting visible/infrared style artifacts. |
| 4 | More identities per batch if memory allows | `--p_num 6 --k_num 8` | More hard negatives and positives per step. |
| 5 | Conservative augmentation | keep `--use_m3plus`, reduce occlusion probabilities in `data/transform.py` by 0.1 | Use if mAP drops while Rank-1 rises. |

## Result Logging Template

Create one row per final run:

| Dataset | Frames | Seed | Checkpoint | i2v R1 | i2v mAP | v2i R1 | v2i mAP | Target Passed |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| BUPTCampus | 10 | 0 | `model_epoch-XXX.pth` |  |  |  |  |  |
| HITSZ-VCM | 10 | 0 | `model_epoch-XXX.pth` |  |  |  |  |  |
