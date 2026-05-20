# M3Plus Project Handoff: Scheme A to K

## Project Context

- Project: M3-ReID / M3Plus
- Local path: `C:\code\M3-ReID`
- Server path: `/root/work/M3-ReID`
- Server GPU: Tesla V100 32GB
- Workflow: edit locally, manually commit and push to GitHub, pull on server, train in `tmux`
- Main protocol: 10-frame
- Goal: improve Rank-1 by 2 points over baseline on both HITSZ-VCM and BUPTCampus
- HITSZ-VCM target: `i2v 75.58 / v2i 79.00`

Current best HITSZ-VCM result:

```text
SchemeD Dv1 + SchemeH multi-clip alpha=0.01
i2v 74.67 / mAP 62.39
v2i 78.15 / mAP 65.23
```

Remaining gap on HITSZ-VCM:

```text
i2v: -0.91
v2i: -0.85
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

## Overall Conclusion

The only clearly effective route so far is:

```text
SchemeD dual_fusion + SchemeH multi-clip
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

1. Analyze Scheme K logs.
2. If Scheme K fails, move to Scheme L: part-level cross-modal matching score.
3. Avoid more loss-only tuning unless it directly targets top-1 ranking.
