# Repair multitask encoder compatibility

- Status: Done
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

- [x] Phase-1 to phase-2 regression is reproduced from immutable artifacts.
- [x] Every promoted screen candidate reports all relevant heads, hit outcomes, latency, and memory.
- [x] At least one isolation strategy and one balancing/distillation strategy are tested.
- [x] The winner passes predeclared gates, or a supported no-change conclusion is recorded.
- [x] Candidate generation is reproducible from recorded config and hashes.

## Validation

- Run the complete fixed-corpus report for every candidate checkpoint.
- Review multi-person and occlusion failures for pose/detection assignment errors.

Completed evidence:

- `benchmark/encoder_repair/build_candidates.py`
- `benchmark/encoder_repair/REPORT.md`
- Phase 1 reconstructed with 100% source checkpoint key coverage and hash-pinned separately.
- Phase 2 changed 546/546 encoder tensors, 234/261 pose tensors, and zero backbone, detection, or
  segmentation tensors.
- Full evaluation rejects the best interpolated candidate: detection +2.1 AP, pose -3.3 AP,
  domain recall -1.4 points, identical direct segmentation hits, and equivalent latency/memory.
- Supported conclusion: retain deployed phase-2 model; no tested repair passes all-task gates.

## Next step

Execute `20260903-1347-ai-model-candidate-benchmark.md`; compare external candidates against the
deployed model because encoder retention produced no joint winner.
