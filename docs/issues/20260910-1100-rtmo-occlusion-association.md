# Diagnose RTMO association under heavy occlusion

- Status: Done
- Repository scope: research checkout only

## Goal

Determine whether David's 0.87 match rate is caused by association geometry or missing model outputs.

## Result

Per-frame instrumentation over all 563 frames found:

- 482 fully matched frames
- 14 geometry-rejection frames
- 27 pose-missing frames
- 17 detection-missing frames
- 23 frames where both were missing

Only 2.5% of frames are addressable by geometry matching. The heavy-occlusion gap is primarily
model availability and must be handled through temporal tracker prediction or deliberately lower
acquisition thresholds, not looser matching that risks cross-person swaps.

## Validation

- Every frame records detection count, pose count, pairs, method, boxes, scores and match rate.
- Existing tracker/smoothing behavior remained unchanged.
- No production files were modified.

## Follow-up result

Duplicate suppression and a longer, uncertainty-aware latch were implemented as opt-in research
controls and tested on both office videos. They reduced duplicate tracks slightly but did not bridge
David's fully hidden interval, while the crosshair hit outputs were byte-for-byte equivalent at the
decision level. The candidate is therefore retained for reproducible research and rejected as a new
default until identity-labelled occlusion data exists.
