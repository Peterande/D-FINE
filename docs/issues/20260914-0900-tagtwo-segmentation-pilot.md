# Prepare a TagTwo segmentation pilot set

- Status: Done (awaiting manual annotation review)
- Repository scope: research checkout only

## Goal

Create a small, representative body-part segmentation pilot from existing office videos without
changing any model or production code.

## Result

- 80 frames: 40 from David and 40 from SogO.
- Stratified selection covers high motion, low sharpness, low light, temporal diversity and the
  known David chair-occlusion interval.
- Every image has a seven-class current-model preannotation and a visual overlay.
- `manifest.csv` records source video, zero-based frame, timestamp, selection reason and review state.
- All preannotations are explicitly marked `needs_manual_review`; they are not ground truth.

Local artifact: `/home/berna/D-FINE-NEWBRINGER/ZKeepResults/tagtwo_segmentation_pilot_v1`.

## Next step

Manually correct the 80 masks into a separate `ground_truth/` directory. Keep the pilot out of
training if it will be used as the final product evaluation set.
