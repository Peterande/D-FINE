# AI model improvement program

- Status: Open
- Owner: Berna / AI and vision research
- Depends on: none
- Production consumer: `/home/berna/tagtwo-monorepo/src/server/ai-engine`

## Goal

Determine whether detection, pose, segmentation, or multitask changes materially improve Tagtwo's
hit pipeline, and promote only a locally measured winner.

## Known evidence

- ModelSurgery is one 640x640 graph with D-FINE detection, DETRPose, and an in-house semantic
  body-part head.
- Local `best_modelsurgery.pth` has 96,628,088 parameters.
- Measured merged results: 43.6 detection AP and 48.2 pose OKS AP.
- The unchanged segmentation head came from `dfine_0.73.pth`, whose saved validation mIoU is 73.64%.
- Pose phase 2 changed every shared-encoder tensor while leaving backbone, detection decoder, and
  segmentation head unchanged.
- Runtime currently misreads `dfine_0.48.pth` as mIoU metadata; 0.48 likely originated as pose AP.

## Program gates

- [x] Exact baseline artifact, preprocessing, labels, and outputs are frozen.
- [x] One immutable evaluation corpus and metric protocol exist.
- [x] Current-model repair and external candidates use identical inputs and gates.
- [x] All three heads, hit quality, latency, memory, and deployability are reported.
- [ ] A winner passes production TensorRT and authoritative game-server validation.

## Work items

- [x] `20260903-1342-ai-production-baseline.md`
- [x] `20260903-1343-ai-evaluation-dataset.md`
- [x] `20260903-1344-ai-end-to-end-metrics.md`
- [x] `20260903-1345-ai-baseline-contract-drift.md`
- [x] `20260903-1346-ai-multitask-encoder-repair.md`
- [x] `20260903-1347-ai-model-candidate-benchmark.md`
- [ ] `20260903-1348-ai-model-production-promotion.md`
- [ ] `20260903-1349-ai-anatomical-hit-validation.md`

## Constraints

- Never select a model from paper AP alone.
- Instance segmentation is not equivalent to the required seven-class semantic segmentation.
- Do not rebuild AIW1, TensorRT runtime, or game-server authority unless evidence requires it.
- Keep research code here and production code in `tagtwo-monorepo`.

## Validation

- Every child issue is `Done` with linked reports, hashes, and reproducible commands.

## Next step

Execute `20260903-1343-ai-evaluation-dataset.md`.
