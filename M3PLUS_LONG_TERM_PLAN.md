# M3Plus Long-Term Execution Plan

> Historical HITSZ notes remain in this file. The active execution line has moved to BUPTCampus; use the BUPTCampus update near the end for the current best result, viable paths, and failed routes.

> Superseded constraint: the project now accepts model/module innovation only. Do not use BoT-style training tricks from "Bag of Tricks and A Strong Baseline for Deep Person Re-identification" as gain sources, including warmup LR, random erasing, label smoothing, last-stride changes, BNNeck as a new contribution, center loss, or image-size/batch-size tuning. Older rows in this file that mention those tricks are historical notes, not active instructions.

## Goal

Active goal as of 2026-05-25: use model/module innovation only, with direct single-clip similarity evaluation, and improve BUPTCampus over the no-BoT baseline by at least 1 Rank-1 point where possible.

Primary acceptance metric: Rank-1 in both retrieval directions.
Secondary acceptance metric: mAP in both retrieval directions.

The older +2 target and HITSZ reference table below are historical context. Do not use re-ranking, multi-clip inference, BoT-style tricks, or evaluation-score changes as model-only evidence.

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
| A | historical | Baseline MVL + local part branch only + cross-modality batch-hard triplet | `--use_m3plus --m3plus_mode part_only --m3plus_aug_strength none --triplet_weight 0.35 --triplet_frame_weight 0.10 --id_label_smoothing 0.0` | Historical row only; BoT-style smoothing is not allowed. |
| B | queued | Baseline checkpoint warm start with a zero-init local residual branch | resume best baseline, lower LR, 30-60 epoch fine-tune | Preserve the original strong global representation while adding local part cues without changing the baseline embedding shape. |
| C | queued | Loss-level change: supervised contrastive or Circle-style metric head | keep model close to Scheme A, replace or down-weight current triplet | Test whether stronger metric geometry gives Rank-1 gains without adding more inference cost. |
| D | ready to run | Baseline-preserving dual global/local fusion | warm-start baseline, freeze global branch for 5 epochs, train local part branch with its own ID + triplet losses, then fine-tune with weighted late fusion | Prevent the part branch from overwhelming MVL features while still giving it direct learning signal; main route after Scheme B stayed below baseline. |
| E | queued | Full M3Plus with conservative augmentation | `--m3plus_mode full`, mild or no weak-light/occlusion | Revisit multi-scale and attention only after the local branch baseline is understood. |

Scheme decision rule:

| Outcome After One Dataset | Decision |
| --- | --- |
| Best average Rank-1 is below baseline by more than 1 point by epoch 90 | stop this scheme on that dataset and move to next scheme. |
| Best average Rank-1 is within 0.5 point of target | continue to epoch 130 and run the other dataset. |
| One direction improves while the other regresses | keep the checkpoint, then test lower triplet weight or Scheme D. |
| Both directions pass target | repeat with seed 1 before claiming the final result. |

## Server Baseline Rerun

Before Scheme B, rerun the original M3-ReID baseline on the same V100 server and dataset mount. This confirms whether the current server, CUDA/PyTorch stack, and uploaded datasets reproduce the provided baseline logs.

Baseline scripts:

```bash
./run_baseline_t10_hitszvcm_v100.sh
./run_baseline_t10_buptcampus_v100.sh
```

They intentionally omit `--use_m3plus`, keep the original README batch shape `P_NUM=4 K_NUM=8`, and run for `EPOCHS=200` by default because the provided 10-frame baselines peak late:

```text
HITSZ-VCM best epoch: 185
BUPTCampus best epoch: 176
```

If baseline physical batch 32 OOMs on V100, retry with `K_NUM=4`. Mark that result as a server sanity check, not as a strict reproduction of the provided baseline logs.

Run one dataset at a time:

```bash
cd /root/work/M3-ReID
git pull
chmod +x run_baseline_t10_hitszvcm_v100.sh run_baseline_t10_buptcampus_v100.sh
tmux new -d -s m3_hitsz_base 'cd /root/work/M3-ReID && CONDA_ENV=base HITSZ_DIR=/root/work/HITSZ-VCM ./run_baseline_t10_hitszvcm_v100.sh'
```

After HITSZ-VCM finishes:

```bash
cd /root/work/M3-ReID
tmux new -d -s m3_bupt_base 'cd /root/work/M3-ReID && CONDA_ENV=base BUPT_DIR=/root/work/BUPTCampus ./run_baseline_t10_buptcampus_v100.sh'
```

Baseline rerun result rows:

| Dataset | Seed | Best Epoch | i2v R1 | i2v mAP | v2i R1 | v2i mAP | Note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| HITSZ-VCM | 0 | 105 | 73.60 | 61.05 | 76.72 | 63.37 | V100 batch 16 rerun, log `baseline_t10_hitszvcm_v100_20260517_074230.log`; use this as the current Scheme B reference. |
| BUPTCampus | 0 |  |  |  |  |  |  |

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
--id_label_smoothing 0.0
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

## Scheme B: Baseline Warm-Start Local Residual Fine-Tune

Scheme B starts from the server baseline `model_best.pth`, then adds a local part residual branch with a smaller learning rate and short milestone schedule. The local residual projection is zero-initialized and keeps the embedding dimension unchanged, so the baseline backbone, MVL, and classifiers can load without shape resets. This is the safer next step after Scheme A underperformed from scratch.

Reference checkpoint location on the server:

```bash
/root/work/M3-ReID/checkpoints/baseline_refs/HITSZVCM_t10_bs16_seed0_epoch105_model_best.pth
```

Save the current HITSZ-VCM baseline best checkpoint there:

```bash
cd /root/work/M3-ReID
mkdir -p checkpoints/baseline_refs
BEST_CKPT=$(find ckptlog/HITSZVCM -path '*Baseline_t10_hitszvcm_v100_bs16*/modelckpt/model_best.pth' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
cp "$BEST_CKPT" checkpoints/baseline_refs/HITSZVCM_t10_bs16_seed0_epoch105_model_best.pth
ls -lh checkpoints/baseline_refs/HITSZVCM_t10_bs16_seed0_epoch105_model_best.pth
```

Fine-tune scripts:

```bash
./run_scheme_b_t10_hitszvcm_v100.sh
./run_scheme_b_t10_buptcampus_v100.sh
```

Default Scheme B deltas:

```text
--resume ${BASELINE_CKPT}
--use_m3plus
--m3plus_mode local_residual
--m3plus_aug_strength none
--sample_method identity_cross_modality
--triplet_weight 0.05
--triplet_frame_weight 0.00
--id_label_smoothing 0.00
--lr_milestones 30,50
EPOCHS=60
```

Run HITSZ-VCM after its baseline is complete:

```bash
cd /root/work/M3-ReID
git pull
chmod +x run_scheme_b_t10_hitszvcm_v100.sh
BASELINE_CKPT=/root/work/M3-ReID/checkpoints/baseline_refs/HITSZVCM_t10_bs16_seed0_epoch105_model_best.pth
tmux new -d -s m3_hitsz_b "cd /root/work/M3-ReID && CONDA_ENV=base HITSZ_DIR=/root/work/HITSZ-VCM BASELINE_CKPT=${BASELINE_CKPT} ./run_scheme_b_t10_hitszvcm_v100.sh"
```

Run BUPTCampus after its baseline is complete:

```bash
cd /root/work/M3-ReID
BASELINE_CKPT=/root/work/M3-ReID/ckptlog/BUPTCampus/Time-XXXX_Baseline_t10_buptcampus_v100_bs32/modelckpt/model_best.pth
tmux new -d -s m3_bupt_b "cd /root/work/M3-ReID && CONDA_ENV=base BUPT_DIR=/root/work/BUPTCampus BASELINE_CKPT=${BASELINE_CKPT} ./run_scheme_b_t10_buptcampus_v100.sh"
```

If the first Scheme B evaluation is already 3+ Rank-1 points below the baseline checkpoint, stop and move to Scheme D instead of spending the full 60 epochs.

Scheme B result rows:

| Scheme | Dataset | Seed | Baseline Checkpoint | Best Epoch | i2v R1 | i2v mAP | v2i R1 | v2i mAP | Decision |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| B | HITSZ-VCM | 0 |  |  |  |  |  |  |  |
| B | BUPTCampus | 0 |  |  |  |  |  |  |  |

## Scheme D: Baseline-Preserving Dual Global/Local Fusion

Scheme D keeps the loaded baseline global MVL representation and classifier shape intact, then adds a separately supervised local part branch. Unlike Scheme B, the local branch is not injected into the global embedding during training. Inference uses weighted late fusion:

```text
feature = concat(sqrt(1-alpha) * norm(global), sqrt(alpha) * norm(local))
```

Default alpha is `0.20`, so the baseline global descriptor remains dominant while the local descriptor can improve hard identity ordering.

Implementation switches:

```text
--resume ${BASELINE_CKPT}
--use_m3plus
--m3plus_mode dual_fusion
--m3plus_aug_strength none
--sample_method identity_cross_modality
--fusion_alpha 0.20
--freeze_base_epochs 5
--lr 0.00001
--part_lr_mult 30.0
--triplet_weight 0.05
--triplet_frame_weight 0.10
--part_id_weight 0.50
--part_triplet_weight 0.35
--part_mma_weight 0.05
--id_label_smoothing 0.0
--lr_milestones 40,60
EPOCHS=80
```

New server scripts:

```bash
./run_scheme_d_t10_hitszvcm_v100.sh
./run_scheme_d_t10_buptcampus_v100.sh
```

Run HITSZ-VCM first:

```bash
cd /root/work/M3-ReID
git pull
chmod +x run_scheme_d_t10_hitszvcm_v100.sh run_scheme_d_t10_buptcampus_v100.sh
tmux new -d -s m3_hitsz_d 'cd /root/work/M3-ReID && CONDA_ENV=base HITSZ_DIR=/root/work/HITSZ-VCM ./run_scheme_d_t10_hitszvcm_v100.sh'
```

Run BUPTCampus after its baseline checkpoint is available:

```bash
cd /root/work/M3-ReID
BASELINE_CKPT=/root/work/M3-ReID/ckptlog/BUPTCampus/Time-XXXX_Baseline_t10_buptcampus_v100_bs32/modelckpt/model_best.pth
tmux new -d -s m3_bupt_d "cd /root/work/M3-ReID && CONDA_ENV=base BUPT_DIR=/root/work/BUPTCampus BASELINE_CKPT=${BASELINE_CKPT} ./run_scheme_d_t10_buptcampus_v100.sh"
```

Decision rule for Scheme D:

| Observation | Decision |
| --- | --- |
| Epoch 5 is more than 2 Rank-1 points below the loaded baseline in both directions | run a checkpoint-only eval with `FUSION_ALPHA=0.10`; if still low, inspect checkpoint loading before continuing. |
| Best epoch by 30 is still below baseline average Rank-1 by more than 0.5 | stop this alpha and retry `FUSION_ALPHA=0.10 PART_TRIPLET_WEIGHT=0.20`. |
| One direction improves while the other regresses | keep the checkpoint and test alpha sweep `0.10,0.15,0.20,0.25` with `test_m3reid.py`. |
| Average Rank-1 exceeds baseline by at least 1 point | continue to 80 epochs and run the other dataset. |

Scheme D result rows:

| Scheme | Dataset | Seed | Baseline Checkpoint | Best Epoch | i2v R1 | i2v mAP | v2i R1 | v2i mAP | Decision |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| D | HITSZ-VCM | 0 | HITSZVCM_t10_bs16_seed0_epoch105_model_best.pth |  |  |  |  |  |  |
| D | BUPTCampus | 0 |  |  |  |  |  |  |  |

## Implemented Method

The implementation is exposed through `--use_m3plus` while keeping the original baseline path available when the flag is omitted. `--m3plus_mode full` keeps the full enhanced path, while `--m3plus_mode part_only` enables Scheme A.

Changed modules:

| Area | File | Change |
| --- | --- | --- |
| Model | `models/modules/enhancement.py` | Added multi-scale residual fusion, lightweight channel-spatial attention, and part-guided local aggregation. |
| Model | `models/model_m3reid.py` | Added optional M3Plus feature path, `full` / `part_only` / `local_residual` mode selection, concatenated local descriptor, and baseline-compatible local residual fine-tune path. |
| Loss | `losses/metric_loss.py` | Added cross-modality batch-hard triplet loss. |
| Loss | `losses/mma_loss.py` | Made invalid-batch fallback return a tensor for stable mixed precision training. |
| Data | `data/transform.py` | Added weak low-light and block occlusion augmentations. |
| Entry | `train_m3reid.py` | Added `--t`, `--use_m3plus`, `--m3plus_mode`, `--lr_milestones`, triplet hyperparameters, cross-modality sampler default, checkpoint-load summary, and M3Plus augmentation path. |
| Entry | `test_m3reid.py` | Added `--t`, `--use_m3plus`, and `--m3plus_mode` for matching checkpoints. |

Additional Scheme D changes:

| Area | File | Change |
| --- | --- | --- |
| Model | `models/model_m3reid.py` | Added `dual_fusion` mode, a separately supervised local branch, base-branch freeze helper, and weighted global/local inference feature concatenation. |
| Entry | `train_m3reid.py` | Added Scheme D local loss weights, part LR multiplier, base freeze phase, fusion alpha, and output-dimension-safe validation. |
| Entry | `test_m3reid.py` | Added `dual_fusion` and `--fusion_alpha` support for alpha sweeps. |
| Scripts | `run_scheme_d_t10_hitszvcm_v100.sh`, `run_scheme_d_t10_buptcampus_v100.sh` | Added V100 run scripts for warm-start dual-fusion fine-tuning. |

Core idea:

1. Multi-scale residual fusion reuses layer3 detail cues before final MVL pooling.
2. Lightweight channel-spatial attention suppresses background and modality-specific noise.
3. Part-guided local aggregation provides horizontal body-part descriptors, a common strong ReID improvement.
4. The `local_residual` mode injects local part cues through a zero-initialized residual projection, allowing baseline checkpoints to load almost fully for safer fine-tuning.
5. Cross-modality batch-hard triplet loss directly optimizes hard visible-infrared positive/negative pairs.
6. Weak low-light and modest occlusion augmentation improve robustness without making data augmentation the only contribution.

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
| 3 | Deprecated under BoT exclusion | `--id_label_smoothing 0.0` | Do not use smoothing as a gain source. |
| 4 | More identities per batch if memory allows | `--p_num 6 --k_num 8` | More hard negatives and positives per step. |
| 5 | Conservative augmentation | keep `--use_m3plus`, reduce occlusion probabilities in `data/transform.py` by 0.1 | Use if mAP drops while Rank-1 rises. |

## Result Logging Template

Create one row per final run:

| Dataset | Frames | Seed | Checkpoint | i2v R1 | i2v mAP | v2i R1 | v2i mAP | Target Passed |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| BUPTCampus | 10 | 0 | `model_epoch-XXX.pth` |  |  |  |  |  |
| HITSZ-VCM | 10 | 0 | `model_epoch-XXX.pth` |  |  |  |  |  |

## HITSZ-VCM Current Record

Current baseline and target:

```text
Reference 10-frame baseline: i2v 73.58 / v2i 77.00
V100 rerun reference: i2v 73.60 / v2i 76.72
Model-only +1 target: i2v 74.58 / v2i 78.00
```

Current best model-only result:

```text
SchemeD dual_fusion, single-clip, alpha=0.05: i2v 74.39 / v2i 77.64
Gap to model-only +1 target: i2v -0.19 / v2i -0.36
```

Separate non-model-only results:

```text
SchemeD + SchemeH multi-clip, no re-ranking: i2v 74.67 / v2i 78.15
SchemeD + SchemeH + SchemeM/N re-ranking: i2v 75.70 / v2i 79.00
```

What works:

- `SchemeD` is the strongest HITSZ model-side route so far: baseline-preserving dual global/local fusion with a small late-fusion weight.
- The local part branch is useful only when it is bounded and added conservatively to the pretrained global embedding.
- `SchemeH` multi-clip and `SchemeM/N` reciprocal top-k re-ranking are useful for separate inference/post-processing gains, but they are not model innovation.

What does not work:

- Part-only training and direct local residual injection damage the strong global embedding.
- Prototype/cross-prototype, CosFace, temporal-conv, and adaptive fusion variants did not reliably lift both i2v and v2i Rank-1.
- Broader part-reranking sweeps saturate around the same top-1 ordering errors and should not be treated as model-side progress.

Current judgment:

- HITSZ model-only progress is real but incomplete: `SchemeD` improves over the baseline, but it does not pass the adjusted +1 target under single-clip direct similarity.
- The only result that reaches the older +2 HITSZ target uses re-ranking, so it must remain separate from the model-side claim.
- A future HITSZ model-side route should stay baseline-preserving and target top-1 ordering without changing evaluation scores after embedding extraction.

## BUPTCampus Current Record

Current baseline and target:

```text
No-BoT baseline: i2v 69.03 / v2i 72.41
Model-only +1 target: i2v 70.03 / v2i 73.41
```

Current best model-only result:

```text
SchemeAA single-clip: i2v 70.15 / v2i 72.96
```

What works:

- `SchemeAA` is the only BUPTCampus model-side run that clearly improves over the baseline.
- The conservative `quad_calibrated_fusion` family is stable and trainable.
- `projection_gate` helps when it keeps the projection branch bounded.

What does not work:

- `SchemeS` reliability part fusion.
- `SchemeT` bidirectional calibration.
- `SchemeV` anchor projection and gate variants.
- `SchemeW` dual calibrated variants.
- `SchemeAC`, `SchemeAC2`, `SchemeAC3`, `SchemeAD`, and `SchemeAD2` router-based direction-aware calibration variants.

Current judgment:

- The best active direction is still `SchemeAA` or a nearby non-router reliability/calibration design.
- Router-heavy modality direction modeling is not currently a viable main path on BUPTCampus.
- `SchemeH` multi-clip can still be used as a separate inference-side enhancement, but it does not count as model innovation.
- Next implemented test: `SchemeAE reliability_quad_calibrated_fusion`, a non-router sample-wise reliability gate over the AA four-branch embedding. It starts from the AA checkpoint and keeps single-clip direct similarity evaluation.
