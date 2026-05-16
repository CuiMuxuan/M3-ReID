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

## Experiment Matrix

We will run one controlled scheme at a time. Each scheme must report the same four numbers: i2v Rank-1, i2v mAP, v2i Rank-1, and v2i mAP. A scheme is considered successful only if both BUPTCampus and HITSZ-VCM reach the target table above on the same 10-frame protocol.

| Scheme | Status | Core Change | Training Delta | Why It Is Tested |
| --- | --- | --- | --- | --- |
| A | ready to run | Baseline MVL + local part branch only + cross-modality batch-hard triplet | `--use_m3plus --m3plus_mode part_only --m3plus_aug_strength none --triplet_weight 0.35 --triplet_frame_weight 0.10 --id_label_smoothing 0.05` | Isolate the most common ReID gain source, local part descriptors, without the full M3Plus fusion and extra augmentation noise. |
| B | queued | Two-stage transfer: baseline checkpoint warm start, then Scheme A/full M3Plus fine-tune | resume best baseline or best Scheme A, lower LR, 30-60 epoch fine-tune | Reduce scratch-training instability and preserve the original strong global representation. |
| C | queued | Loss-level change: supervised contrastive or Circle-style metric head | keep model close to Scheme A, replace or down-weight current triplet | Test whether stronger metric geometry gives Rank-1 gains without adding more inference cost. |
| D | queued | Gated local/global fusion | replace plain concatenation with a small gate/projection for global and part features | Prevent the part branch from overwhelming MVL features; likely useful if Scheme A mAP drops or one direction regresses. |
| E | queued | Full M3Plus with conservative augmentation | `--m3plus_mode full`, mild or no weak-light/occlusion | Revisit multi-scale and attention only after the local branch baseline is understood. |

Scheme decision rule:

| Outcome After One Dataset | Decision |
| --- | --- |
| Best average Rank-1 is below baseline by more than 1 point by epoch 90 | stop this scheme on that dataset and move to next scheme. |
| Best average Rank-1 is within 0.5 point of target | continue to epoch 130 and run the other dataset. |
| One direction improves while the other regresses | keep the checkpoint, then test lower triplet weight or Scheme D. |
| Both directions pass target | repeat with seed 1 before claiming the final result. |

## Scheme A: Part-Only Local Branch

Goal: isolate local part descriptors plus hard cross-modality metric learning. This is intentionally not the previous full M3Plus setting.

Implementation switches:

```text
--use_m3plus
--m3plus_mode part_only
--m3plus_aug_strength none
--sample_method identity_cross_modality
--part_num 4
--part_dim 2048
--triplet_weight 0.35
--triplet_frame_weight 0.10
--id_label_smoothing 0.05
```

New server scripts:

```bash
./run_scheme_a_t10_hitszvcm_v100.sh
./run_scheme_a_t10_buptcampus_v100.sh
```

Server tmux commands. Run one dataset at a time on a single V100:

```bash
cd /root/work/M3-ReID
git pull
chmod +x run_scheme_a_t10_hitszvcm_v100.sh run_scheme_a_t10_buptcampus_v100.sh
tmux new -d -s m3_hitsz_a 'cd /root/work/M3-ReID && CONDA_ENV=base HITSZ_DIR=/root/work/HITSZ-VCM ./run_scheme_a_t10_hitszvcm_v100.sh'
```

After HITSZ-VCM finishes, start BUPTCampus:

```bash
cd /root/work/M3-ReID
tmux new -d -s m3_bupt_a 'cd /root/work/M3-ReID && CONDA_ENV=base BUPT_DIR=/root/work/BUPTCampus ./run_scheme_a_t10_buptcampus_v100.sh'
```

Result row to fill after each run:

| Scheme | Dataset | Seed | Best Epoch | i2v R1 | i2v mAP | v2i R1 | v2i mAP | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A | HITSZ-VCM | 0 |  |  |  |  |  |  |
| A | BUPTCampus | 0 |  |  |  |  |  |  |

## Implemented Method

The implementation is exposed through `--use_m3plus` while keeping the original baseline path available when the flag is omitted. `--m3plus_mode full` keeps the full enhanced path, while `--m3plus_mode part_only` enables Scheme A.

Changed modules:

| Area | File | Change |
| --- | --- | --- |
| Model | `models/modules/enhancement.py` | Added multi-scale residual fusion, lightweight channel-spatial attention, and part-guided local aggregation. |
| Model | `models/model_m3reid.py` | Added optional M3Plus feature path, `full` / `part_only` mode selection, and concatenated local part descriptor. |
| Loss | `losses/metric_loss.py` | Added cross-modality batch-hard triplet loss. |
| Loss | `losses/mma_loss.py` | Made invalid-batch fallback return a tensor for stable mixed precision training. |
| Data | `data/transform.py` | Added weak low-light and block occlusion augmentations. |
| Entry | `train_m3reid.py` | Added `--t`, `--use_m3plus`, `--m3plus_mode`, triplet hyperparameters, cross-modality sampler default, and M3Plus augmentation path. |
| Entry | `test_m3reid.py` | Added `--t`, `--use_m3plus`, and `--m3plus_mode` for matching checkpoints. |

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

The V100 observed OOM at physical batch 24/32, while physical batch 16 is stable. Keep physical batch 16 for full runs.

The `P_NUM=8 K_NUM=2 ACCUM_STEPS=2`, AdamW, compact part branch retry was worse on HITSZ-VCM:

```text
Run: M3Plus_t10_v100_bs16_acc2
Best average Rank-1: Epoch 60
i2v: r1 62.24, mAP 46.57
v2i: r1 64.46, mAP 47.72
Conclusion: do not continue this configuration.
```

The current default scripts return to the stronger quality setting while keeping DataLoader stability fixes:


```text
P_NUM=4
K_NUM=4
ACCUM_STEPS=1
PART_DIM=2048
FEATURE_DROPOUT=0.0
OPTIMIZER=adam
M3PLUS_AUG_STRENGTH=standard
TRIPLET_FRAME_WEIGHT=0.25
EPOCHS=130
EVAL_START_EPOCH=80        # skip expensive early validation
TEST_INTERVAL=5            # validate every 5 epochs after epoch 80
EARLY_STOP_PATIENCE=8
--eval_fp16                # faster validation feature extraction
```

AMP is enabled for both training and evaluation. The main time saving is skipping early validation; the training recipe itself stays close to the best observed run.

Optional memory experiments:

```bash
# Save activation memory in the M3Plus attention/local head only. Slower, but may allow P_NUM=9 K_NUM=2 or larger TEST_BATCH_SIZE.
GRAD_CHECKPOINT_HEAD=1 HITSZ_DIR=/root/work/HITSZ-VCM ./run_m3plus_t10_hitszvcm_v100.sh

# Requires: pip install bitsandbytes
OPTIMIZER=adam8bit HITSZ_DIR=/root/work/HITSZ-VCM ./run_m3plus_t10_hitszvcm_v100.sh
```

Do not checkpoint the ResNet backbone by default because it contains BatchNorm layers; naïve checkpointing can update BN running statistics during recomputation and hurt ReID accuracy. If time is more important than maximum accuracy, run a compact ablation with `MVL_NUM_HEADS=1 PART_DIM=512`; if accuracy drops, return to the default.

DataLoader stability note for the V100 server:

```text
WORKERS=2
PERSISTENT_WORKERS=0
PREFETCH_FACTOR=2
TORCH_SHARING_STRATEGY=file_system
```

This is the default after the Python 3.12 multiprocessing `ConnectionRefusedError` seen with 8 workers and persistent workers. If the run is stable and GPU utilization is too low, try `WORKERS=4 PERSISTENT_WORKERS=0` first; only re-enable persistent workers after a full epoch has completed cleanly.

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
