# Current model landscape and fit for this project

## Executive answer

There are newer and stronger standalone models, especially for pose. There is no verified current
model that can replace this project's detection, pose, and seven-class semantic segmentation in one
drop-in shared graph. Most advertised multi-task families ship separate checkpoints, instance masks
instead of body-part masks, or research pipelines too heavy for the current runtime.

## Ranked next tests

1. **DETRPose-X adapter/distillation** — highest confidence. It reproduces 74.41 pose AP locally,
   uses a compatible output task and permissive license, and the 0.51-AP transplant control proves
   exactly why an adapter must be trained.
2. **RTMO-L standalone control** — one-stage, Apache-2.0, 74.8 published AP and real-time design.
   It cannot share the current encoder directly, but can prove whether DETRPose is still the right
   pose teacher.
3. **RTMPose-M top-down control** — 75.8 published AP and broad deployment support. Measure total
   latency with the existing detector because runtime cost scales with detected people.
4. **RF-DETR standalone detection control** — permissive and the only current candidate in this
   audit reporting over 60 COCO AP. Compare person/domain quality, not only 80-class AP.
5. **SCHP Pascal-7 segmentation control** — closest public semantic parsing checkpoint. It needs
   label reconciliation and must be measured on the reconstructed dataset before conclusions.

## Detection

Stock D-FINE-X remains a strong source at a locally reproduced 59.3 AP. DEIMv2-X (57.8), YOLO26-X
(57.5), and RTMDet-X (52.8) do not justify replacement on accuracy. ECDet-X with Objects365 reports
59.9, only 0.6 above stock, while requiring a new backbone, attached-head retraining, and a
commercial license. RF-DETR is the only permissive candidate with enough claimed headroom to merit
a separate local test.

## Pose

Pose has real headroom: DETRPose-X, RTMO-L, RTMPose-M and ECPose-X report roughly 74–76 AP versus
48.2 for the merged head. This does not make them interchangeable. RTMPose is top-down; RTMO and
ECPose are standalone one-stage networks; ECPose has a commercial-license gate. DETRPose-X is the
best controlled teacher because it has already reproduced 74.41 locally. Direct decoder transplant
fails, so feature isolation—not another blind surgery—is the next experiment.

## Body-part segmentation

The needed task is semantic body-part parsing with project labels, not COCO instance segmentation.
ECSeg, RF-DETR-Seg and normal YOLO segmentation therefore do not answer it. SCHP is the closest
public seven-class control, while Sapiens is a useful high-resolution teacher. SST and D2FP offer
training ideas but lack a verified low-risk deployment artifact in this audit. The current 73.64%
checkpoint metric must first be reproduced on the reconstructed Pascal corpus.

## Genuine multitask systems

BBoxMaskPose is the closest functional idea because boxes, masks and pose refine one another, but it
is an iterative RTMDet/SAM/PMPose pipeline and its masks are instances. UniHCP genuinely learns
shared human-centric representations across detection, pose and parsing, but uses a large research
stack and top-down pose. HumanBench is a pretraining framework, not simultaneous inference.
EdgeCrafter, Sapiens and YOLO26 support multiple tasks as product families; they do not provide the
required single checkpoint with simultaneous box, COCO-17 pose, and body-part outputs.

## Why “newer” is not automatically better here

- Published tasks and label spaces frequently differ.
- Standalone AP omits the compatibility cost of shared features.
- Instance masks cannot score anatomical hits.
- Top-down pose adds per-person work and detector dependence.
- TensorRT numbers from T4/V100 do not predict RTX 5070 Ti latency directly.
- DEIMv2 and EdgeCrafter default licenses prohibit commercial use.
- A one-head gain cannot hide detection, segmentation, hit, or latency regressions.

The complete 25-entry evidence registry is `registry.json`. Published claims and local measurements
are separate fields throughout.
