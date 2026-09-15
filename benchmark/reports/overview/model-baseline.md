# Model baseline — Tagtwo AI hit pipeline

Status of the production model as of **1 September 2026**: what it is, what it
scores, what it costs, and where its weights come from.

Every number below is either **measured** (command given, reproducible) or
marked **unverified**. Nothing is carried over from a filename or a README
without saying so.

Intended home: `tagtwo-monorepo/src/server/ai-engine/`, beside `SPEC.md`.

**Measurement environment**

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti, 16 GB, driver 570.172.08 |
| Runtime | PyTorch 2.9.1+cu128, FP32, no TensorRT |
| Input | 640×640, square resize |

Production runs FP16 TensorRT. Every latency figure here is therefore a
**ceiling**, not a production measurement.

---

## 1. Architecture

One shared backbone feeding three task heads, built by
`SharedBackboneDualDecoder` in `model_surgery/shared_arch.py`.

```
HGNetv2-B5 backbone + HybridEncoder        (shared)
├── det_decoder     ← D-FINE-X obj2coco
├── pose_decoder    ← DETRPose, 17 keypoints
└── seg_head        ← FPN-ASPP, 7 body-part classes (built in-house)
```

Head dimensions read directly out of `best_modelsurgery.pth`, not from config:

| Tensor | Shape | Meaning |
|---|---|---|
| `det_decoder.enc_score_head.weight` | `(80, 256)` | 80-class COCO detector, **not** the 2-class soldier set |
| `seg_head.decoder.7.weight` | `(7, 192, 1, 1)` | the 7 body-part classes |

The segmentation head has **no upstream original**. D-FINE ships no
segmentation head; this one was written in-house. There is consequently no
"before model surgery" segmentation number to regress against — mIoU 48 is a
starting point, not a decline.

### Body-part classes

`utils/hit_detector.py:7-14`

```
0 background   1 head       2 torso        3 upper_arms
4 lower_arms   5 upper_legs 6 lower_legs
```

### Output contract

The graph emits raw tensors; all postprocessing happens in the runtime.

| Tensor | Form | Runtime postprocessing |
|---|---|---|
| `det_pred_logits` | logits | sigmoid (focal-loss style) → argmax class |
| `det_pred_boxes` | raw | canonical xyxy, coordinate space auto-detected |
| `seg_logits` | N,C,H,W logits | `np.argmax(axis=0)` → uint8 class mask |
| `pose_pred_logits` | logits | keypoint decoding |

### Hit decision

`detect_person_part_hit` takes the crosshair — **hardcoded to the image
centre**, `hit_detector.py:345-346` — reads a 3×3 region of the class mask
around it, and returns the argmax over non-background classes.

Pose is not independent of this: on occlusion, `_infer_occluded_body_part`
falls back to a torso quadrilateral built from keypoints 5, 6, 11 and 12 (COCO
shoulders and hips). A change to the pose head therefore moves hit results too.

---

## 2. Metrics

### 2.1 Detection — measured

**COCO val2017, all 5000 images.** The detection head is a standard 80-class
COCO detector, so these numbers are directly comparable to published figures.

| | AP | AP50 | AP75 | AP_S | AP_M | AP_L | AR100 |
|---|---|---|---|---|---|---|---|
| **Production merged (segpose)** | **43.6** | 59.0 | 48.0 | 26.7 | 47.5 | 56.7 | 66.5 |
| Stock D-FINE-X obj2coco | 59.3 | 76.8 | 64.6 | 42.2 | 64.2 | 76.4 | 77.3 |

**The control run is the reason to trust these.** Evaluating the stock
checkpoint through the identical pipeline returned **59.3 AP**, matching the
59.3 that D-FINE publishes for X-obj2coco exactly. Preprocessing, postprocessing
and category mapping are therefore correct, and the merged model's 43.6 is real
rather than a harness artefact.

Person class only — the class that matters for this product:

| | person AP | person AP50 | person AR100 |
|---|---|---|---|
| **Production merged** | **61.1** | 84.6 | 72.4 |
| Stock D-FINE-X obj2coco | 67.6 | 89.3 | 76.8 |

### 2.2 Detection on domain data — measured

COCO is not Tagtwo. This is `soldier_coco` val: 116 images, 291 annotated
persons, `score_thr 0.5`, `iou_thr 0.5`, both ground-truth classes treated as
person.

| | Recall | Found | False pos. | FP/image | Box IoU |
|---|---|---|---|---|---|
| **Production merged** | 0.863 | 251/291 | **48** | **0.41** | 0.893 |
| Stock D-FINE-X obj2coco | 0.904 | 263/291 | 59 | 0.51 | 0.905 |

**Read this before acting on the COCO gap.** On domain data the multi-task model
is not simply worse — it trades recall for precision. It misses 12 more persons
out of 291 (−4.1 points recall) but produces 11 fewer false persons, at
essentially identical localisation quality.

For hit detection that trade is not obviously bad: a false person becomes a
false hit. The 15.7-point COCO AP gap overstates the product consequence
substantially, because multi-task training degrades the 79 classes this product
never uses.

### 2.3 Pose — measured

COCO `person_keypoints_val2017`, all images containing annotated people,
`maxDets=20` per the COCO keypoint protocol.

| | OKS AP | AP50 | AP75 | AP_M | AP_L | AR |
|---|---|---|---|---|---|---|
| **Production merged pose head** | **48.2** | 74.7 | 50.9 | 39.2 | 61.1 | 54.5 |

Against the DETRPose family this head is derived from:

| Model | OKS AP |
|---|---|
| **Production merged** | **48.2** |
| DETRPose-N (smallest) | 57.2 |
| DETRPose-S | 67.0 |
| DETRPose-M | 69.4 |
| DETRPose-L | 72.5 |
| DETRPose-X (same size class) | 73.3 |

**This is the widest gap anywhere in the pipeline.** The pose head scores 25
points below DETRPose-X — the variant matching the backbone size actually
deployed — and 9 points below DETRPose-N, the smallest model the authors ship.

#### Control run — the gap is real

The official DETRPose-X checkpoint was downloaded and run through this same
harness, along two paths, to separate a genuine model gap from a measurement
fault:

| Path | OKS AP |
|---|---|
| Published DETRPose-X | 73.3 |
| Official model + DETRPose's own postprocessor, scored here | **74.4** |
| Official model + our `DETRPosePostProcessor`, scored here | **74.4** |
| Production merged pose head | **48.2** |

Both paths reproduce the published figure, and they agree with each other
exactly. The evaluation path, the keypoint scoring and our own postprocessor
are therefore all sound, and **the 48.2 is a property of the trained weights,
not of the measurement.**

(The 74.4 sits 1.1 above the published 73.3 most likely because this evaluation
restricts itself to images containing annotated people, which removes images
where a false positive could only hurt. It does not affect the conclusion.)

Two further notes:

- Running with `--orig-size-order hw` collapses to 0.000 AP, confirming `wh` is
  the correct convention and the keypoints are not transposed.
- The head was trained on in-house data, so COCO OKS understates its performance
  on Tagtwo footage. But the shared backbone *is* COCO-trained, and a 26-point
  gap against the same-size official model is far too large to be domain shift.

### 2.4 Segmentation — UNVERIFIED

`mIoU 48` appears in the production engine filename
(`dfine_segpose_x_fp16_miou48_…`) and in the S3 key (`dfine_0.48.pth`). It has
**not** been reproduced, and the evaluation protocol behind it is unknown.

No annotated 7-class body-part dataset was found anywhere in the repo or the
monorepo, so it cannot currently be measured. Per-class IoU is also unknown —
we cannot say whether arms and legs drag the mean down or whether the weakness
is uniform.

This matters more than the detection numbers: the class mask is what decides
which body part gets hit.

---

## 3. Latency

Measured on `David.mp4`, 563 frames, 360×640 source, PyTorch FP32.

| Stage | ms/frame |
|---|---|
| Model | 30.6 |
| Postprocess | 0.35 |
| Render (overlays) | 6.1 |
| **Total** | **39.4** → ~25 FPS |

COCO evaluation, 5000 images at native resolution: **33.0 ms/image** model time.

No percentiles yet — these are means. p50/p95 need the benchmark harness.
TensorRT FP16 is unmeasured; expect it to be materially faster.

---

## 4. Provenance

| Artefact | Location | Notes |
|---|---|---|
| Trained weights | S3 (Digital Ocean Spaces) | `dfine_0.48.pth`, `model_manager.py:36` |
| Production engine | `models/engines/` | `dfine_segpose_x_fp16_miou48_NVIDIA_GeForce_RTX_5070_Ti_trt1013035.engine`, 153.9 MB |
| Older engine | `models/engines/` | same name at `trt109034`, 154.5 MB |
| `models/checkpoints/` | — | **empty** |
| `models/original_weights/` | — | **empty** |

Engines are built per GPU and TensorRT version, named
`dfine_{task}_{size}_{precision}_miou{miou}_{gpu}_trt{version}.engine`
(`model_manager.py:285-286`).

### The local checkpoint

`D-FINE-NEWBRINGER/Traning/best_modelsurgery.pth` (738.8 MB) loads into the
shared architecture with `missing=0 unexpected=0`. That proves the
**architecture** matches. It does **not** prove it is the same weight file
production runs — that requires hashing it against the S3 `dfine_0.48.pth`,
which has not been done.

A second local engine, `dfine_segmentation_640_fp16_map73_rtx5070ti_trt10.15.1.engine`,
is named for a detection mAP rather than a segmentation mIoU and uses a
different TensorRT version. It is a different artefact from production.

---

## 5. Discrepancies found

These are real inconsistencies discovered while verifying the above. None has
been fixed.

**Preprocessing disagreement.** The repo inference tool applies
`Resize + ToTensor` only. The monorepo runtime (`utils/dfine_segmentation.py:88-100`)
applies `/255` followed by ImageNet mean/std normalisation. These are different
inputs to the same model. The control run reproducing 59.3 AP exactly used the
no-normalisation path, which is evidence that it is the correct one — and
therefore that the runtime path may be feeding the model wrongly scaled input.
**Worth investigating.**

**Segmentation mask resolution.** `seg_logits` comes out at the segmentation
head's resolution, not the frame's. `detect_person_part_hit` indexes it with
frame-derived centre coordinates without rescaling
(`hit_detector.py:345-359`). Unless the two happen to match, the crosshair
reads the wrong pixel. The benchmark harness rescales with `INTER_NEAREST`
first.

**Threshold mismatch.** Production uses `confidence_threshold = 0.5`
(`utils/config.py:27`); the repo tool defaults to `--score-thr 0.35`. Any
comparison must pin one value.

**Broken config paths.** The configs in `Traning/` carry `__include__` paths
that resolve outside the repository root, so they cannot be loaded from where
they sit. Working copies live at `segmentation_sivert/base_dfine/` and
`pose_estimation_berna/base_dfine/` and are byte-identical.

---

## 6. Reproducing these numbers

From the benchmark worktree, `/home/berna/D-FINE-NEWBRINGER-benchmark`.

```bash
DET=segmentation_sivert/base_dfine/dfine_hgnetv2_x_obj2coco.yml
POSE=pose_estimation_berna/base_dfine/dfine_hgnetv2_x_obj2coco_detrpose_paper.yml
CKPT=/home/berna/D-FINE-NEWBRINGER/Traning/best_modelsurgery.pth
STOCK=/home/berna/D-FINE-NEWBRINGER/segmentation_sivert/base_dfine/dfine_x_obj2coco.pth
COCO=/home/berna/D-FINE-NEWBRINGER/datasets/coco

# COCO val2017 detection AP — production model
python benchmark/harness/eval_coco_det.py --det-config $DET --pose-config $POSE \
  --merged-ckpt $CKPT --coco-root $COCO --out-json runs/coco_det_val2017.json

# Same, stock D-FINE-X — the control that validates the harness
python benchmark/harness/eval_coco_det.py --det-config $DET --pose-config $POSE \
  --plain-ckpt $STOCK --coco-root $COCO --out-json runs/coco_det_stock.json

# Person recall / false positives / box IoU on domain data
python benchmark/harness/eval_person_domain.py --det-config $DET --pose-config $POSE \
  --merged-ckpt $CKPT --plain-ckpt $STOCK \
  --ann /home/berna/D-FINE-NEWBRINGER/datasets/soldier_coco/annotations/instances_val.json \
  --img-dir /home/berna/D-FINE-NEWBRINGER/datasets/soldier_coco/images/val

# Annotated video with crosshair and hit verdict per frame
python benchmark/harness/infer_crosshair.py --det-config $DET --pose-config $POSE \
  --merged-ckpt $CKPT --input /home/berna/D-FINE-NEWBRINGER/datasets/David.mp4 \
  --out runs/david.mp4 --hit-json runs/david.json \
  --image-size 640 --seg-num-classes 7 --seg-feature-dim 384 \
  --score-thr 0.5 --profile
```

---

## 7. What is still missing

1. **Segmentation mIoU**, per class. Blocked — no annotated body-part data
   found. The cheap substitute is a visual check: 30–50 representative frames,
   mask overlays, mark only obvious errors at the crosshair. That answers "is it
   good enough" without pixel labelling or retraining.
3. **TensorRT FP16 latency**, with p50/p95 rather than means.
4. **Checkpoint identity**: hash `best_modelsurgery.pth` against S3 `dfine_0.48.pth`.
5. **Real Tagtwo weapon footage.** `David.mp4` is an 18-second portrait phone
   clip of one person walking indoors, with no occlusion, no distance variation
   and no crosshair. It works as a smoke test; it cannot separate candidate
   models from one another.

## 8. Assessment

**Detection needs no action.** DEIMv2-X reports 57.8 AP and ECDet-X 57.9 — both
*below* the 59.3 of the D-FINE-X obj2coco weights this pipeline already builds
on. D-FINE itself has published nothing since November 2024: no releases, no
tags. The multi-task trade-off in §2.2 is small and arguably favourable for hit
detection, since it buys fewer false persons at the cost of a few missed ones.

**Pose is the one real finding, and it is now confirmed.** At 48.2 OKS AP the
pose head sits 26 points below the official DETRPose-X measured through this
same harness (74.4), and 9 below the smallest DETRPose variant the authors ship.
The control run rules out a measurement fault. Nothing else in this pipeline
shows a gap of that size.

The fix is not a new model. The head is *already* DETRPose — the official
weights for the same architecture score 74.4. This is a training problem, not a
technology-selection problem, and it does not require migrating to EdgeCrafter
or anything else.

Pose also feeds the hit decision directly — `_infer_occluded_body_part` builds
its torso quadrilateral from keypoints 5, 6, 11 and 12, so keypoint quality
governs occlusion handling, one of the four criteria this project set out to
improve.

**Segmentation remains unquantified.** mIoU 48 has never been reproduced and no
annotated body-part data exists to reproduce it with. The mask decides which
body part is hit, so this is the most consequential unknown — but it cannot be
worked on until that data exists. Producing it is a labelling task, not a
modelling one.

**Recommended order:**

1. **Retrain the pose head.** Freeze the backbone, train the head, then finetune
   jointly and carefully — and regression-test all three heads afterwards. That
   last step is what was missed last time: nobody noticed detection had given up
   6.5 person AP during the surgery.
2. **Visual segmentation check** on 30–50 frames. Cheap, no labelling.
3. **Detection: no action.**

The whole question this project set out to answer — "can a newer model do
better?" — resolves to no. The gap that exists is against weights that already
ship for the architecture in use.
