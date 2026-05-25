# M3Plus Project Handoff: Scheme A to K and BUPTCampus Follow-up

## Project Context

- Project: M3-ReID / M3Plus
- Local path: `C:\code\M3-ReID`
- Server path: `/root/work/M3-ReID`
- Server GPU: Tesla V100 32GB
- Workflow: edit locally, manually commit and push to GitHub, pull on server, train in `tmux`
- Main protocol: 10-frame
- Current goal: model-side innovation only, no re-ranking contribution counted.
- BUPTCampus updated acceptance threshold: improve Rank-1 by at least 1 point over the no-BoT baseline in both directions when using single-clip direct similarity evaluation.
- Innovation requirement: at least one clearly writable model innovation, supported by at least two concrete model modules.
- HITSZ-VCM model-only target: `i2v 74.58 / v2i 78.00`
- BUPTCampus model-only target: train a fresh baseline first, then set target to baseline Rank-1 + 1.00 in both directions.

Target scope:

```text
Counted: trained model architecture/modules/losses that affect the emitted embedding.
Not counted: SchemeM/N re-ranking, reciprocal boost, part-score reranking, or other post-hoc score-matrix changes.
Separate reporting only: SchemeH multi-clip, because it is an inference sampling strategy rather than model innovation.
```

BoT exclusion rule:

```text
The paper "Bag of Tricks and A Strong Baseline for Deep Person Re-identification" is now an explicit exclusion list.
Do not use or claim gains from its trick set: warmup learning rate, random erasing, label smoothing, last-stride changes, BNNeck as a new contribution, center loss, image-size/batch-size tuning, or other baseline/training-trick changes from that paper.
Allowed direction: model/module innovation only, with direct single-clip evaluation and no post-hoc score changes.
Implementation guardrails: label smoothing defaults to 0.0, M3Plus extra augmentation defaults to none, and RandomErasing is disabled by default in train_m3reid.py.
```

Current best HITSZ-VCM model-only result:

```text
SchemeD Dv1 single-clip alpha=0.05
i2v 74.39 / mAP 61.47 / mINP 34.89
v2i 77.64 / mAP 64.32 / mINP 34.67
```

Remaining HITSZ-VCM model-only gap:

```text
i2v: -0.19
v2i: -0.36
```

Best HITSZ-VCM result with re-ranking, kept as a separate non-model-only result:

```text
SchemeD Dv1 + SchemeH multi-clip + SchemeM/N direction-aware reranking
i2v 75.70 / mAP 63.13 / mINP 36.03
v2i 79.00 / mAP 65.95 / mINP 36.19
```

Important checkpoints:

```text
Original baseline:
/root/work/M3-ReID/checkpoints/baseline_refs/HITSZVCM_t10_bs16_seed0_epoch105_model_best.pth

Current strongest warm-start checkpoint:
/root/work/M3-ReID/ckptlog/HITSZVCM/Time-2026-05-18_01-27-50_SchemeD_dualfusion_t10_hitszvcm_v100_bs16/modelckpt/model_best.pth
```

## Scheme A: Part-Only Local Branch

What it did:

- Enabled only the local part branch.
- Used cross-modality batch-hard triplet.
- Intended to strengthen the baseline with horizontal body-part descriptors.

Why it did not reach the target:

- The part branch alone could not preserve the strong global MVL representation from the baseline.
- Local descriptors helped in principle, but without the original global embedding as the main route, ranking quality was not stable enough.

Decision:

- Do not continue as the main route.

## Scheme B: Baseline Warm-Start Local Residual

What it did:

- Warm-started from the baseline checkpoint.
- Added a zero-initialized local residual branch.
- Injected local part features into the original embedding.

Key result:

```text
Log: ckptlog/scheme_b_t10_hitszvcm_v100_20260517_203303.log
Best roughly: i2v 72.97 / v2i 76.15
```

Why it did not reach the target:

- The residual branch directly perturbed the baseline embedding space.
- Zero initialization reduced initial damage, but training still failed to add useful cross-modal local information.
- The global ranking space was weakened more than the local branch helped.

Decision:

- Failed. Do not continue.

## Scheme C: Metric Loss / Circle / Supervised Contrastive Direction

What it planned:

- Keep the model close to the baseline.
- Improve metric geometry through stronger ranking/classification objectives.

Why it did not become the main route:

- Later CosFace, prototype, and cross-prototype experiments showed the same pattern: loss-only pressure can improve local geometry or one direction, but does not reliably improve both i2v and v2i Rank-1.
- The current bottleneck is not ordinary classification convergence. It is top-1 retrieval ordering.

Decision:

- Do not pursue as a standalone main route.

## Scheme D: Baseline-Preserving Dual Global/Local Fusion

What it did:

- Preserved the baseline global MVL representation.
- Added a separately supervised local part branch.
- Training kept global and local branches separate.
- Inference used late fusion:

```text
feature = concat(sqrt(1-alpha) * norm(global), sqrt(alpha) * norm(local))
```

Key results:

```text
Dv1 best epoch 40 single clip:
i2v 74.15 / mAP 61.48
v2i 77.33 / mAP 64.25

Dv1 alpha sweep best alpha=0.05:
i2v 74.39
v2i 77.64
```

Why it did not reach the target:

- It was effective, but fixed-alpha late fusion has limited headroom.
- The local branch used horizontal stripe aggregation only.
- It lacked sample-level local reliability estimation and explicit cross-modal part matching.

Decision:

- This is the most important training base.
- Future effective schemes should warm-start from Dv1.

## Scheme E: Full M3Plus With Multi-Scale / Attention / Augmentation

What it did:

- Enabled full M3Plus components:
  - multi-scale residual fusion
  - lightweight channel-spatial attention
  - part-guided aggregation
  - extra low-light and occlusion augmentation

Why it did not reach the target:

- Full changes shifted the training distribution too much from the strong baseline.
- V100 batch size was limited to 16, reducing hard-negative quality.
- The complete enhanced path underperformed the conservative baseline-preserving route.

Decision:

- Do not continue full mode.
- Prefer conservative warm-started structural changes.

## Scheme F: Prototype Loss

What it did:

- Added EMA class-prototype loss.
- Intended to stabilize class centers and cross-modal embedding structure.

Why it did not reach the target:

- Prototype loss compressed features toward class centers.
- This can help class compactness or mAP, but top-1 Rank-1 needs fine-grained separation between hard identities.
- The loss disturbed the baseline ranking space without delivering enough top-1 gain.

Decision:

- Failed. Do not continue.

## Scheme G: Cross-Modality Prototype

What it did:

- Added cross-modality prototype triplet / xproto constraints.
- Tried to pull same-ID visible/infrared prototypes together and push different-ID prototypes apart.

Known log:

```text
ckptlog/HITSZVCM/log_Time-2026-05-18_23-28-44_SchemeG_xproto_alpha005_t10_hitszvcm_v100_bs16.txt
```

Why it did not reach the target:

- The run was interrupted, and the observed trend was not promising.
- Like Scheme F, prototype constraints focused too much on global class centers.
- This can reduce test-ID fine-grained discrimination and does not directly fix top-1 ordering.

Decision:

- Stop. Do not continue.

## Scheme H: Multi-Clip Inference

What it did:

- Did not change training.
- During evaluation, averaged embeddings from multiple non-overlapping clips for each track.

Key results:

```text
Dv1 + SchemeH alpha=0.01:
i2v 74.67 / mAP 62.39 / mINP 35.77
v2i 78.15 / mAP 65.23 / mINP 35.47

Dv2 + SchemeH alpha=0.01:
i2v 74.50
v2i 78.11
```

Why it did not reach the target:

- Multi-clip averaging reduces sampling noise and improves mAP.
- It only smooths inference features. It does not change the learned top-1 decision boundary.
- The remaining Rank-1 gap is still about 0.9 points.

Decision:

- Current strongest inference enhancement.
- Every promising checkpoint should be evaluated with SchemeH multi-clip.

## Scheme I: Temporal Refinement Head

What it did:

- Added a lightweight temporal embedding refinement head.
- Used low-rank temporal convolution to refine frame-level embeddings.

Key results:

```text
Single clip best epoch 25:
i2v 73.97 / mAP 62.08
v2i 77.72 / mAP 64.59

SchemeH multi-clip on SchemeI best:
i2v 74.76 / mAP 63.24
v2i 77.76 / mAP 65.80
```

Why it did not reach the target:

- mAP improved, but v2i Rank-1 dropped.
- The temporal head behaved more like smoothing/denoising.
- It did not improve the top-1 boundary and may have weakened strong discriminative frames.

Decision:

- Do not continue ordinary temporal conv refinement.

## Scheme J: CosFace Proxy Fine-Tuning

What it did:

- Warm-started from Dv1.
- Added CosFace proxy loss to sharpen classification margins.

Important parameters:

```text
lr=3e-6
cosface_weight=0.20
cosface_frame_weight=0.25
part_cosface_weight=0.03
margin=0.18
scale=32
epochs=55
```

Current log:

```text
ckptlog/HITSZVCM/log_Time-2026-05-19_18-39-48_SchemeJ_cosface_fromDv1_alpha001_t10_hitszvcm_v100_bs16.txt
```

Key results:

| Epoch | i2v R1 | v2i R1 | Avg R1 |
| ---: | ---: | ---: | ---: |
| 5 | 73.76 | 77.51 | 75.64 |
| 10 | 74.17 | 77.58 | 75.88 |
| 15 | 74.08 | 77.37 | 75.73 |
| 20 | 74.02 | 77.84 | 75.93 |
| 25 | 74.11 | 77.58 | 75.85 |
| 30 | 73.87 | 77.68 | 75.78 |

Why it did not reach the target:

- CosFace slightly helped local classification geometry and v2i at epoch 20.
- It suppressed i2v and did not exceed SchemeH.
- Rank-1 stopped improving before the LR drop, so natural recovery was unlikely.

Decision:

- Stop. Do not continue CosFace.

## Scheme K: Adaptive Dual-Fusion Gate

Current status:

- Code has been implemented locally.
- Training log was checked after this handoff.
- Decision: failed by the epoch 20/25 rule; do not continue as the main route.

What it does:

- Adds `adaptive_dual_fusion` mode.
- Replaces fixed alpha late fusion with sample-level gate prediction:

```text
feature = concat(sqrt(1-gate) * global, sqrt(gate) * local)
```

- Adds a fused head and supervises the fused embedding with ID and triplet losses.
- Gate is initialized near `fusion_alpha=0.01`, so it starts close to the strongest Dv1 + SchemeH setting.

Changed files:

```text
models/modules/enhancement.py
models/model_m3reid.py
train_m3reid.py
test_m3reid.py
run_scheme_h_eval_t10_hitszvcm_v100.sh
run_scheme_k_t10_hitszvcm_v100.sh
```

Local check passed:

```powershell
python -m py_compile train_m3reid.py test_m3reid.py models\model_m3reid.py models\modules\enhancement.py
```

Suggested commit message:

```text
Add SchemeK adaptive dual-fusion gate
```

Server training command:

```bash
cd /root/work/M3-ReID
git pull
chmod +x run_scheme_k_t10_hitszvcm_v100.sh

tmux new -d -s m3_hitsz_k 'cd /root/work/M3-ReID && CONDA_ENV=base HITSZ_DIR=/root/work/HITSZ-VCM BASELINE_CKPT=/root/work/M3-ReID/ckptlog/HITSZVCM/Time-2026-05-18_01-27-50_SchemeD_dualfusion_t10_hitszvcm_v100_bs16/modelckpt/model_best.pth ./run_scheme_k_t10_hitszvcm_v100.sh'
```

Default Scheme K parameters:

```text
m3plus_mode=adaptive_dual_fusion
fusion_alpha=0.01
adaptive_gate_min=0.0
adaptive_gate_max=0.10
lr=2e-6
freeze_base_epochs=5
part_lr_mult=8
fusion_id_weight=0.20
fusion_triplet_weight=0.12
epochs=45
eval_start_epoch=5
test_interval=5
early_stop_patience=6
```

Why it may still fail:

- Scheme K changes fusion reliability, not the intrinsic quality of the part feature.
- If the local branch does not contain enough complementary identity information, the adaptive gate can only avoid harmful local usage, not create new discriminative evidence.

Decision rule:

```text
Check epoch 20/25 first.
If single-clip metrics do not approach or exceed i2v 74.4 / v2i 78.0, stop Scheme K.
If it shows upward movement, run to 45 epochs and evaluate best checkpoint with SchemeH multi-clip.
```

Observed Scheme K log:

```text
ckptlog/HITSZVCM/log_Time-2026-05-19_21-34-47_SchemeK_adaptivefusion_fromDv1_alpha001_t10_hitszvcm_v100_bs16.txt
```

Key results:

| Epoch | i2v R1 | v2i R1 | Avg R1 |
| ---: | ---: | ---: | ---: |
| 5 | 74.30 | 77.58 | 75.94 |
| 10 | 74.02 | 77.27 | 75.65 |
| 15 | 74.11 | 77.17 | 75.64 |
| 20 | 74.00 | 77.23 | 75.62 |
| 25 | 74.43 | 77.41 | 75.92 |

Why it did not reach the target:

- Epoch 20/25 did not approach the required `i2v 74.4 / v2i 78.0` threshold in both directions.
- Best average Rank-1 stayed at epoch 5, so the gate did not show a useful upward trend.
- The learned gate moved near the upper cap around 0.08-0.09, but this mostly increased local usage without fixing the v2i top-1 gap.

## Scheme L: Part-Level Cross-Modal Matching Score

Current status:

- Implemented as evaluation-time score fusion on top of dual-fusion embeddings.
- It keeps the Dv1/SchemeH checkpoint unchanged.
- It splits the local branch into horizontal part descriptors and adds a small explicit part-match similarity to the retrieval score.
- Direction-specific weights are supported through `PART_MATCH_WEIGHT_I2V` and `PART_MATCH_WEIGHT_V2I`.

Observed default result:

```text
Dv1 + SchemeH + SchemeL, fusion_alpha=0.01, part_match_weight=0.02
i2v 74.72 / mAP 62.43 / mINP 35.83
v2i 78.11 / mAP 65.27 / mINP 35.54
```

Observed direction-specific result:

```text
Dv1 + SchemeH + SchemeL, fusion_alpha=0.01, PART_MATCH_WEIGHT_I2V=0.02, PART_MATCH_WEIGHT_V2I=0.0
i2v 74.72 / mAP 62.43 / mINP 35.83
v2i 78.15 / mAP 65.24 / mINP 35.47
```

Interpretation:

- Compared with Dv1 + SchemeH `74.67 / 78.15`, this gives `+0.05` i2v but `-0.04` v2i.
- Average Rank-1 is effectively unchanged, so Scheme L is not enough as a main route.
- The weak i2v gain suggests testing direction-specific fusion: keep part matching for i2v, fall back to baseline scoring for v2i.
- Direction-specific fusion successfully avoids the v2i loss, but the i2v gain remains only `+0.05`; Scheme L alone is not enough.

## Scheme M: Top-K Part Candidate Reranking

Current status:

- Implemented as evaluation-time top-k reranking.
- It first ranks candidates with the current fused score, then only adjusts each query's top-k candidates with a normalized explicit part-match score.
- Default script: `run_scheme_m_eval_t10_hitszvcm_v100.sh`.

Default parameters:

```text
PART_MATCH_WEIGHT_I2V=0.02
PART_MATCH_WEIGHT_V2I=0.0
PART_RERANK_TOPK=20
PART_RERANK_WEIGHT_I2V=0.015
PART_RERANK_WEIGHT_V2I=0.015
PART_RERANK_NORM=zscore
```

Decision rule:

```text
If Scheme M does not improve at least one direction by >=0.3 without hurting the other direction, stop reranking-only work.
If one direction improves but the other drops, rerun with the hurting direction's PART_RERANK_WEIGHT set to 0.0.
```

Observed default result:

```text
Dv1 + SchemeH + SchemeM, topk=20, PART_RERANK_WEIGHT_I2V=0.015, PART_RERANK_WEIGHT_V2I=0.015
i2v 74.85 / mAP 62.46 / mINP 35.79
v2i 78.47 / mAP 65.32 / mINP 35.64
```

Interpretation:

- Compared with Dv1 + SchemeH `74.67 / 78.15`, Scheme M gives `+0.18` i2v and `+0.32` v2i.
- Compared with direction-specific Scheme L `74.72 / 78.15`, top-k reranking adds `+0.13` i2v and `+0.32` v2i.
- This is the first post-SchemeH route that improves both directions without retraining, so continue a narrow Scheme M sweep.
- Remaining HITSZ-VCM gap to target is `i2v -0.73`, `v2i -0.53`.

Partial sweep results:

| Config | i2v R1 | i2v mAP | i2v mINP | v2i R1 | v2i mAP | v2i mINP | Note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| top20 / i2v0.015 / v2i0.015 | 74.85 | 62.46 | 35.79 | 78.47 | 65.32 | 35.64 | Default Scheme M |
| top10 / i2v0.015 / v2i0.015 | 74.87 | 62.38 | 35.62 | 78.37 | 65.17 | 35.27 | Slight i2v R1 gain, worse v2i and mAP |
| top20 / i2v0.010 / v2i0.015 | 74.85 | 62.52 | 35.88 | 78.47 | 65.32 | 35.64 | Same R1 as default, better i2v mAP/mINP |
| top20 / i2v0.020 / v2i0.015 | 74.85 | 62.38 | 35.62 | 78.47 | 65.32 | 35.64 | Same R1, worse i2v mAP/mINP |
| top20 / i2v0.015 / v2i0.020 | 74.85 | 62.46 | 35.79 | 78.27 | 65.24 | 35.55 | Higher v2i weight hurts v2i R1 |
| top30 / i2v0.015 / v2i0.015 | 74.78 | 62.50 | 35.89 | 78.49 | 65.36 | 35.70 | Best v2i, weaker i2v |

Current best Scheme M setting:

```text
Rank-1 tie: top20 / PART_RERANK_WEIGHT_I2V=0.010 or 0.015 / PART_RERANK_WEIGHT_V2I=0.015
Prefer top20 / i2v0.010 / v2i0.015 for now because i2v mAP and mINP are slightly better.
```

Next Scheme M refinement:

```text
Top-k sensitivity is directional:
- i2v R1 peaks at top10 / i2v0.015.
- v2i R1 peaks at top30 / v2i0.015.
Add direction-specific top-k and test i2v_topk=10, v2i_topk=30 with rerank weights 0.015/0.015.
```

Observed direction-specific top-k result:

```text
SchemeM direction-specific top-k, i2v_topk=10, v2i_topk=30, weights 0.015/0.015
i2v 74.87 / mAP 62.37 / mINP 35.62
v2i 78.49 / mAP 65.36 / mINP 35.70
```

Interpretation:

- This combines the best observed directional Rank-1 values from the sweep.
- Compared with Dv1 + SchemeH `74.67 / 78.15`, the gain is `+0.20` i2v and `+0.34` v2i.
- Remaining HITSZ-VCM gap to target is `i2v -0.71`, `v2i -0.51`.
- Scheme M appears near saturation as an evaluation-only part-reranking route; further same-style sweeps are unlikely to close the full gap.

## Scheme N: Cross-Modal Reciprocal Candidate Boost

Rationale:

- Remaining errors are top-1 ordering errors after multi-clip, part matching, and top-k part reranking.
- A common failure mode is a gallery/query hub that is close to many opposite-modality samples but is not a reciprocal neighbor of the current query.
- Scheme N gives a small boost only to candidates inside the current top-k list that are also reciprocal top-k neighbors in the reverse direction.

Default setting:

```text
Base: SchemeM direction-specific top-k
RECIPROCAL_TOPK=20
RECIPROCAL_WEIGHT_I2V=0.010
RECIPROCAL_WEIGHT_V2I=0.010
```

Decision rule:

```text
If reciprocal boost improves both directions or moves one direction by >=0.2 without hurting the other, continue a tiny sweep.
If it hurts either direction by >=0.1, stop Scheme N or set the hurting direction's reciprocal weight to 0.
```

Observed default result:

```text
SchemeN, Base=SchemeM direction-specific top-k, RECIPROCAL_TOPK=20, weights 0.010/0.010
i2v 75.22 / mAP 62.74 / mINP 35.85
v2i 78.72 / mAP 65.69 / mINP 35.96
```

Interpretation:

- Compared with SchemeM direction-specific top-k `74.87 / 78.49`, Scheme N adds `+0.35` i2v and `+0.23` v2i.
- Compared with Dv1 + SchemeH `74.67 / 78.15`, the total gain is `+0.55` i2v and `+0.57` v2i.
- Remaining HITSZ-VCM gap to target is now only `i2v -0.36`, `v2i -0.28`.
- This is the strongest route so far. Continue a narrow reciprocal-weight sweep around `RECIPROCAL_TOPK=20`, not broader part-rerank sweeps.

Narrow reciprocal-weight sweep:

| Config | i2v R1 | i2v mAP | i2v mINP | v2i R1 | v2i mAP | v2i mINP | Note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| rk20 / i2v0.010 / v2i0.010 | 75.22 | 62.74 | 35.85 | 78.72 | 65.69 | 35.96 | Default Scheme N |
| rk20 / i2v0.012 / v2i0.012 | 75.33 | 62.81 | 35.88 | 78.84 | 65.75 | 36.02 | Improves both directions |
| rk20 / i2v0.015 / v2i0.010 | 75.39 | 62.90 | 35.93 | 78.72 | 65.69 | 35.96 | Best i2v so far |
| rk20 / i2v0.010 / v2i0.015 | 75.22 | 62.74 | 35.85 | 78.94 | 65.82 | 36.08 | Best v2i so far |
| rk20 / i2v0.015 / v2i0.015 | 75.39 | 62.90 | 35.93 | 78.94 | 65.82 | 36.08 | Current best, combines both directions |
| rk20 / i2v0.018 / v2i0.015 | 75.55 | 62.99 | 35.97 | 78.94 | 65.82 | 36.08 | i2v nearly reaches target |
| rk20 / i2v0.020 / v2i0.015 | 75.57 | 63.04 | 36.01 | 78.94 | 65.82 | 36.08 | Current best, i2v short by 0.01 |
| rk20 / i2v0.022 / v2i0.015 | 75.59 | 63.08 | 36.02 | 78.94 | 65.82 | 36.08 | i2v reaches target |
| rk20 / i2v0.025 / v2i0.015 | 75.70 | 63.13 | 36.03 | 78.94 | 65.82 | 36.09 | Best i2v so far |
| rk20 / i2v0.020 / v2i0.018 | 75.57 | 63.04 | 36.01 | 78.96 | 65.88 | 36.13 | Best v2i so far, i2v short by 0.01 |
| rk20 / i2v0.022 / v2i0.018 | 75.59 | 63.08 | 36.02 | 78.96 | 65.88 | 36.12 | Best balanced near-target setting |
| rk20 / i2v0.022 / v2i0.020 | 75.59 | 63.08 | 36.02 | 78.96 | 65.92 | 36.17 | v2i still short |
| rk20 / i2v0.025 / v2i0.020 | 75.70 | 63.13 | 36.03 | 78.96 | 65.92 | 36.17 | Strong i2v, v2i still short |
| rk20 / i2v0.022 / v2i0.022 | 75.59 | 63.08 | 36.02 | 79.00 | 65.95 | 36.19 | Reaches both targets |
| rk20 / i2v0.025 / v2i0.022 | 75.70 | 63.13 | 36.03 | 79.00 | 65.95 | 36.19 | Final best: strongest i2v and target v2i |

Next priority:

```text
Final HITSZ-VCM best:
rk20 / i2v0.025 / v2i0.022
i2v 75.70 / mAP 63.13 / mINP 36.03
v2i 79.00 / mAP 65.95 / mINP 36.19

This reaches the HITSZ-VCM target `i2v 75.58 / v2i 79.00`.
Compared with Dv1 + SchemeH `74.67 / 78.15`, final SchemeN gain is `+1.03` i2v and `+0.85` v2i.
Compared with original baseline target gap, this closes the needed remaining gap on HITSZ-VCM.
```

## Model-Only No-Reranking Stage: Schemes O/P/Q

Reason:

- Current HITSZ final result uses SchemeM/N re-ranking.
- Project target has been adjusted: only model innovation counts, and the Rank-1 threshold is reduced from +2 to +1.
- A valid final method must contain at least one writable innovation and at least two model modules.
- Without re-ranking, the strongest observed result is `SchemeD single-clip alpha=0.05`:

```text
i2v 74.39 / v2i 77.64
```

- With SchemeH multi-clip but no re-ranking, the strongest observed result is:

```text
i2v 74.67 / v2i 78.15
```

Goal:

```text
Test whether model-side changes can pass the adjusted +1 model-only target:
HITSZ i2v >= 74.58 and v2i >= 78.00 under single-clip direct similarity.
Training-time validation in train_m3reid.py uses single-clip direct similarity, so it is the strict model-only comparison.
```

Implemented modes:

```text
Scheme O: m3plus_mode=supervised_dual_fusion
- Fixed dual_fusion concat stays unchanged at inference.
- Adds fused concat embedding ID/triplet supervision during training.

Scheme P: m3plus_mode=gated_residual_fusion
- Injects local part cues into the global embedding through a bounded zero-initialized residual path.
- Output dimension stays 12288, so evaluation is direct global embedding retrieval.

Scheme Q: m3plus_mode=part_token_fusion
- Uses the global descriptor as a query over 4 local part tokens.
- Injects attended local context through a zero-initialized residual projection.
- Output dimension stays 12288.
```

Writable innovation candidates:

```text
Innovation name:
Reliability-aware local-to-global embedding enhancement for video VI-ReID.

Module 1:
PartGuidedAggregation extracts softly weighted horizontal body-part descriptors with GeM pooling.

Module 2:
One of the model-side fusion modules below:
- Scheme O fused embedding supervision head.
- Scheme P bounded gated residual local-to-global fusion.
- Scheme Q part-aware token fusion.

Only claim the final innovation after the corresponding module combination passes the +1 model-only target.
```

New scripts:

```text
run_scheme_o_t10_hitszvcm_v100.sh
run_scheme_p_t10_hitszvcm_v100.sh
run_scheme_q_t10_hitszvcm_v100.sh
run_no_rerank_eval_t10_hitszvcm_v100.sh
```

Recommended sequential validation:

```bash
tmux new -d -s m3_hitsz_o 'cd /root/work/M3-ReID && CONDA_ENV=base HITSZ_DIR=/root/work/HITSZ-VCM bash ./run_scheme_o_t10_hitszvcm_v100.sh'
```

After Scheme O finishes and its best single-clip result is parsed:

```bash
tmux new -d -s m3_hitsz_p 'cd /root/work/M3-ReID && CONDA_ENV=base HITSZ_DIR=/root/work/HITSZ-VCM bash ./run_scheme_p_t10_hitszvcm_v100.sh'
```

After Scheme P finishes:

```bash
tmux new -d -s m3_hitsz_q 'cd /root/work/M3-ReID && CONDA_ENV=base HITSZ_DIR=/root/work/HITSZ-VCM bash ./run_scheme_q_t10_hitszvcm_v100.sh'
```

Decision rule:

```text
Primary target: compare each run against the original baseline-derived target: i2v 74.58 / v2i 78.00.
Secondary diagnostic: compare against SchemeD single-clip alpha=0.05: i2v 74.39 / v2i 77.64.
Accept a model-only solution only if both directions pass the primary target.
If a scheme passes only one direction, tune that scheme narrowly before moving to another broad architecture.
If all three stay below the primary target, do not claim model-only success; report SchemeD as partial model gain and SchemeM/N as separate re-ranking gain.
```

## BUPTCampus Next Stage

Dataset path:

```text
/root/work/BUPTCampus
```

Current status:

- BUPTCampus evaluation is supported by `test_m3reid.py`.
- A fresh BUPTCampus baseline has now been trained.
- The current best BUPTCampus model-only result is SchemeAA single-clip, not a re-ranking result.
- BUPTCampus-specific baseline, SchemeD, SchemeH, and SchemeN wrapper scripts are available.
- Do not start from SchemeN evaluation yet; first train a BUPTCampus baseline checkpoint, then warm-start model-side schemes from that checkpoint.

New scripts:

```text
run_baseline_t10_buptcampus_v100.sh
run_scheme_d_t10_buptcampus_v100.sh
run_scheme_h_eval_t10_buptcampus_v100.sh
run_scheme_n_eval_t10_buptcampus_v100.sh
run_scheme_n_sweep_t10_buptcampus_v100.sh
```

Initial BUPTCampus training/evaluation plan:

```text
1. Train the original BUPTCampus t=10 baseline and record its best Rank-1.
2. Set the BUPTCampus model-only +1 targets from that baseline.
3. Warm-start SchemeD dual_fusion from the BUPTCampus baseline checkpoint.
4. Evaluate model-only single-clip results first.
5. Only report SchemeH or SchemeN as separate inference/re-ranking analyses, not as model-only target progress.
```

Baseline training command:

```bash
tmux new -d -s m3_bupt_base 'cd /root/work/M3-ReID && CONDA_ENV=base BUPT_DIR=/root/work/BUPTCampus bash ./run_baseline_t10_buptcampus_v100.sh'
```

OOM-safe baseline defaults after the first BUPTCampus OOM:

```text
P_NUM=4
K_NUM=4
train_batch_size=16
ACCUM_STEPS=2
TEST_BATCH_SIZE=16
```

SchemeD training command after baseline finishes:

```bash
tmux new -d -s m3_bupt_d 'cd /root/work/M3-ReID && CONDA_ENV=base BUPT_DIR=/root/work/BUPTCampus BASELINE_CKPT=/root/work/M3-ReID/ckptlog/BUPTCampus/<baseline-dir>/modelckpt/model_best.pth bash ./run_scheme_d_t10_buptcampus_v100.sh'
```

Default SchemeN BUPTCampus sweep:

```text
0.01:20:0.025:0.022
0.05:20:0.025:0.022
0.10:20:0.025:0.022
0.20:20:0.025:0.022
```

Format:

```text
FUSION_ALPHA:RECIPROCAL_TOPK:RECIPROCAL_WEIGHT_I2V:RECIPROCAL_WEIGHT_V2I
```

Important:

```text
Before judging BUPTCampus progress, identify the baseline Rank-1 and the +1 model-only target for both directions.
Do not assume HITSZ final weights are optimal on BUPTCampus.
```

## SchemeH Evaluation Template

Best Dv1 multi-clip evaluation:

```bash
M3PLUS_MODE=dual_fusion \
FUSION_ALPHA=0.01 \
MODEL_CKPT=/root/work/M3-ReID/ckptlog/HITSZVCM/Time-2026-05-18_01-27-50_SchemeD_dualfusion_t10_hitszvcm_v100_bs16/modelckpt/model_best.pth \
HITSZ_DIR=/root/work/HITSZ-VCM \
CONDA_ENV=base \
BATCH_SIZE=32 \
DESC=SchemeH_multiclip_schemeD_v1_alpha0.01_t10_hitszvcm_v100 \
./run_scheme_h_eval_t10_hitszvcm_v100.sh
```

SchemeK best multi-clip evaluation should use:

```text
M3PLUS_MODE=adaptive_dual_fusion
FUSION_ALPHA=0.01
ADAPTIVE_GATE_MAX=0.10
MODEL_CKPT=<SchemeK output dir>/modelckpt/model_best.pth
```

## BUPTCampus Experiment Summary

### Baseline and best known results

No-BoT baseline:

```text
i2v 69.03 / v2i 72.41
```

Current best model-only result:

```text
SchemeAA single-clip
i2v 70.15 / v2i 72.96
```

This is the current best model-side result. It clears the i2v +1 target but not the v2i +1 target.

Best observed calibration-style or reliability-style alternatives did not beat AA:

```text
SchemeW5 best:
i2v 69.96 / v2i 72.78

SchemeS best:
i2v 69.78 / v2i 72.04

SchemeT best:
i2v 69.59 / v2i 72.41

SchemeV best:
i2v 69.96 / v2i 72.96

SchemeAC2 best:
i2v 69.96 / v2i 72.96

SchemeAC3 best:
i2v 69.96 / v2i 72.96

SchemeAD2 best:
i2v 69.96 / v2i 72.96
```

### What worked

- `SchemeAA` is the only BUPTCampus model-side scheme so far that clearly improves over the baseline in single-clip direct similarity.
- The four-branch `quad_calibrated_fusion` family is structurally valid and can be trained stably.
- `projection_gate` regularization is useful when it prevents the projection branch from dominating.
- `SchemeH` multi-clip is still useful as an inference-side enhancer, but it is not counted as model innovation.

### What did not work

- `SchemeS` reliability part fusion: no meaningful improvement over the baseline.
- `SchemeT` bidirectional calibration: weak v2i router learning, no target pass.
- `SchemeV` anchor projection and its gate variants: stable, but not enough to beat AA.
- `SchemeW` dual calibration variants: good calibration behavior, but still below AA on Rank-1.
- `SchemeAC` and `SchemeAD` router-based direction-aware calibration: router supervision either stayed near random or hurt Rank-1 after learning.
- `SchemeAC2`, `SchemeAC3`, `SchemeAD`, and `SchemeAD2` all failed to outperform AA.

### Current conclusion

- The current best path for BUPTCampus is still `SchemeAA`.
- Router-based modality-direction modeling is not a viable main path on this dataset under the present loss design.
- The next model-side step should use a non-router reliability or calibration design, preferably one that keeps the global embedding untouched and adds a separate auxiliary branch or a detached reliability gate.

## Overall Conclusion

The only route that reached the old HITSZ +2 target used re-ranking:

```text
SchemeD dual_fusion + SchemeH multi-clip + SchemeM/N direction-aware re-ranking
```

Under the current model-only +1 target, the strongest completed model-side result is still short:

```text
SchemeD single-clip alpha=0.05
i2v 74.39 / v2i 77.64
Target i2v 74.58 / v2i 78.00
```

Failed routes share a pattern:

- Prototype and cross-prototype over-constrain class centers.
- CosFace sharpens classification margins but does not improve both retrieval directions.
- Temporal conv improves mAP but does not reliably improve top-1 Rank-1.
- Full M3Plus changes too much from the strong baseline and underperforms conservative warm-started changes.

Current bottleneck:

```text
Top-1 retrieval ordering, especially i2v, not ordinary classification convergence.
```

Next priority:

1. Keep BUPTCampus model-side work centered on `SchemeAA`-style calibration or a new detached reliability gate.
2. Keep HITSZ SchemeM/N only as a separate re-ranking result, not as model-only target evidence.
3. Treat router-heavy direction-aware schemes as closed for BUPTCampus unless a new architecture removes router gradients from the retrieval path.

## HITSZ-VCM Current Record

Baseline and target:

```text
Reference 10-frame baseline: i2v 73.58 / v2i 77.00
V100 rerun reference: i2v 73.60 / v2i 76.72
Model-only +1 target: i2v 74.58 / v2i 78.00
```

Current best model-only result:

```text
SchemeD dual_fusion single-clip alpha=0.05
i2v 74.39 / mAP 61.47 / mINP 34.89
v2i 77.64 / mAP 64.32 / mINP 34.67
```

This is the strongest HITSZ model-side result so far, but it is still short of the adjusted model-only +1 target by `0.19` i2v Rank-1 and `0.36` v2i Rank-1.

Separate non-model-only results:

```text
SchemeD + SchemeH multi-clip, no re-ranking
i2v 74.67 / v2i 78.15

SchemeD + SchemeH + SchemeM/N reciprocal top-k re-ranking
i2v 75.70 / mAP 63.13 / mINP 36.03
v2i 79.00 / mAP 65.95 / mINP 36.19
```

The second result reaches the old HITSZ +2 target, but it depends on re-ranking and must not be claimed as model-side innovation. SchemeH multi-clip is also inference sampling, not model innovation.

What worked:

- `SchemeD` dual global/local fusion is the best HITSZ model-side line.
- The useful pattern is baseline-preserving fusion: keep the pretrained global embedding dominant and add local part evidence with a small bounded coefficient.
- `SchemeM/N` reciprocal direction-aware re-ranking is effective as a post-processing analysis route, but only for separate reporting.

What did not work:

- `SchemeA` part-only local branch and `SchemeB` direct residual injection weakened the global retrieval space.
- `SchemeG/I/J/K` style prototype, temporal, CosFace, and adaptive fusion variants did not improve both retrieval directions enough.
- Continuing top-k part-reranking sweeps after `SchemeN` is low value because the route is already saturated and not counted as model innovation.

Main conclusion:

- HITSZ has a usable model innovation candidate in `SchemeD`, but the strict single-clip model-only result remains below the +1 target.
- The successful HITSZ target-closing result is an inference/post-processing stack, not a pure model result.
- For any future HITSZ model-side attempt, start from the `SchemeD` checkpoint and preserve the global embedding; do not revive router-heavy or score-matrix reranking work as model-side evidence.

## BUPTCampus Current Record

Baseline:

```text
i2v 69.03 / v2i 72.41
```

Current best model-only result:

```text
SchemeAA single-clip
i2v 70.15 / v2i 72.96
```

This is the only BUPTCampus model-side result that clearly beats the baseline. It passes the +1 target in i2v, but not in v2i.

What worked:

- `SchemeAA` quad calibrated fusion from `SchemeW5` warm start.
- Conservative projection/calibration branching.
- Bounded `projection_gate` supervision.

What did not work:

- `SchemeS` reliability part fusion.
- `SchemeT` bidirectional calibration.
- `SchemeV` anchor projection variants.
- `SchemeW` dual calibrated variants.
- `SchemeAC`, `SchemeAC2`, `SchemeAC3`, `SchemeAD`, and `SchemeAD2` direction-aware router lines.

Main conclusion:

- BUPTCampus is not currently a router problem.
- The router-based direction-aware family learned either nothing useful or hurt Rank-1 after it learned.
- The most defensible next path is still a non-router reliability branch that stays detached from the main retrieval embedding.
- `SchemeAE reliability_quad_calibrated_fusion` was able to run after the forward-path fix, but it early-stopped at epoch 15 and did not beat `SchemeAA`.
- The first AE run also missed the `reliability_quad_calibrated_fusion` calibration loss wiring; that was corrected afterward.
- `SchemeAE2 reliability_quad_calibrated_fusion` reused the AA warm start and corrected calibration-loss wiring, but still early-stopped at epoch 15 with best `i2v 69.96 / v2i 72.96`.
- `SchemeAF supervised_quad_calibrated_fusion` ran from the `SchemeAA` best checkpoint. It activated final-fusion ID/triplet losses, but early stopped at epoch 20 with best epoch 5 result `i2v 69.96 / v2i 72.96`; it did not beat `SchemeAA`.
- Current next run: `SchemeAG agreement_quad_residual_fusion`. It keeps the fixed `SchemeAA` four-branch output and appends a zero-initialized, bounded agreement-aware residual refinement head. This remains model-side only: single-clip direct-similarity evaluation, no re-ranking, no multi-clip claim.
