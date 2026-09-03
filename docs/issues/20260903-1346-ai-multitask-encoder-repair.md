# Repair multitask encoder compatibility

- Status: Open
- Owner: Berna / AI and vision research
- Depends on: issues 1342 through 1345

## Goal

Improve pose without degrading detection, segmentation, hit quality, or deployment performance.

## Evidence

- ModelSurgery phase 2 changed all 546 HybridEncoder tensors.
- Backbone, detection decoder, and segmentation head remained byte-identical to `dfine_0.73.pth`.
- The final merged model measured 43.6 detection AP and 48.2 pose AP.
- Official source heads perform substantially better before the shared-feature surgery.

## Experiments

- Frozen encoder control.
- Task-specific adapters or partial isolation.
- Balanced multitask loss or gradient balancing.
- Teacher distillation from stock D-FINE and official DETRPose.
- Joint fine-tuning only when every head has a non-regression gate.

## Acceptance criteria

- [ ] Phase-1 to phase-2 regression is reproduced from immutable artifacts.
- [ ] Every run reports all heads, hit outcomes, latency, and memory.
- [ ] At least one isolation strategy and one balancing/distillation strategy are tested.
- [ ] The winner passes predeclared gates, or a supported no-change conclusion is recorded.
- [ ] Training is reproducible and resumable from recorded config and hashes.

## Validation

- Run the complete fixed-corpus report for every candidate checkpoint.
- Review multi-person and occlusion failures for pose/detection assignment errors.

## Next step

Reproduce phase-1 and phase-2 detection and pose metrics with immutable checkpoint and dataset hashes.
