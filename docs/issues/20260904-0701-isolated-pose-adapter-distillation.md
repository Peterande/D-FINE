# Train an isolated DETRPose-X adapter with teacher distillation

- Status: Open
- Owner: Berna / AI and vision research
- Depends on: `20260904-0700-current-model-landscape.md`
- Repository scope: research checkout only

## Goal

Improve pose toward the locally reproduced 74.41 OKS AP DETRPose-X teacher without modifying the
detection-compatible shared encoder or degrading detection and segmentation.

## Evidence

- Current merged pose: 48.2 OKS AP.
- Official DETRPose-X: 74.41 OKS AP on the same local evaluation protocol.
- Direct decoder transplant: 0.51 OKS AP, proving feature semantics are incompatible.
- Encoder interpolation improved detection but reduced pose and domain recall.

## Experiments

1. Frozen shared backbone and encoder control.
2. Per-feature-level 1x1 residual adapters before the pose decoder.
3. Adapter plus pose-decoder fine-tuning on COCO keypoints.
4. Feature, logit, and keypoint distillation from the frozen official DETRPose-X teacher.
5. Optional low-rank or gated adapters only if the 1x1 control is insufficient.

## Protected invariants

- Backbone, shared encoder, detection decoder, and segmentation head remain byte-identical unless a
  later experiment explicitly opens them behind all-task regression gates.
- Detection AP, soldier-domain recall/FP, segmentation logits, direct hit decisions, latency, and
  memory are reported with pose.
- Generated checkpoints and teacher outputs remain outside Git with hashes in reports.

## Acceptance criteria

- [ ] Training config, seeds, source hashes, optimizer, schedule, and resume state are recorded.
- [ ] Frozen-module identity is proven before and after training.
- [ ] At least one adapter-only control and one teacher-distilled run complete.
- [ ] Every candidate is evaluated with the frozen end-to-end protocol.
- [ ] A winner improves pose materially without violating protected-task or runtime gates, or a
  supported no-change conclusion is recorded.

## Outputs

- `benchmark/pose_adapter/` training and evaluation code
- `benchmark/pose_adapter/REPORT.md`
- Checkpoints and raw teacher/candidate outputs under ignored `.cache/`

## Validation

- Unit-test adapter shapes, residual identity initialization, and frozen-module hashes.
- Resume a short training run and prove deterministic continuation.
- Run full COCO pose plus detection, domain, segmentation, and runtime gates on the best candidate.

## Next step

Implement identity-initialized per-level residual adapters and a one-batch frozen-module training
smoke test before launching expensive training.
