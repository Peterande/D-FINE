# External model candidate verdicts

## Result

No external candidate is promotable as a drop-in replacement for the current single-pass model.
The conclusion is based on task compatibility, local controls, license, and integration cost—not
paper AP alone.

## What was tested locally

- Official DETRPose-X reproduces **74.41 OKS AP** standalone on the pinned COCO protocol.
- Transplanting that exact official decoder onto the shared production encoder collapses to
  **0.51 OKS AP**. Shape compatibility is therefore not feature compatibility.
- Sapiens-0.3B was compared on the same 30 office frames. It took 227.0 ms/model inference versus
  33.9 ms for the baseline, agreed on only 56.7% of crosshair verdicts, and has no ground-truth
  evidence that those changes are improvements.
- EdgeCrafter and DEIMv2 were revision- and license-audited before weights/training. Both default
  licenses forbid commercial use; neither is a compatible multitask checkpoint. Their expensive
  paths were stopped by the predeclared cheap gate.

## Decisions

- **DEIMv2-X — reject:** 57.8 published COCO AP is below stock obj2coco D-FINE-X at 59.3; changing
  its DINOv3 backbone would invalidate both attached heads.
- **ECDet-X O365 — monitor:** 59.9 published AP is only +0.6 over stock and requires full retraining
  plus a separate commercial license.
- **ECPose-X — monitor:** strong standalone paper result, but not a drop-in decoder; the DETRPose
  transplant control demonstrates the risk directly.
- **ECSeg — reject:** instance segmentation is not seven-class body-part segmentation.
- **Official DETRPose-X — retrain:** best locally validated pose source, but only behind task-
  specific feature isolation or end-to-end retraining.
- **Sapiens — monitor as teacher:** too slow and not validated as a replacement.
- **SCHP Pascal — reject:** lower published mIoU, mismatched label semantics, separate full model.

## Recommendation

Keep the deployed model. The only evidence-backed next research direction is a task-specific pose
adapter/distillation experiment using official DETRPose-X as teacher while preserving detection
features. That is a new training project, not a ready model promotion.

Raw results remain outside Git; their hashes are recorded in `registry.json` and the baseline
reports. Official repository revisions and license text were read directly on 4 September 2026.
