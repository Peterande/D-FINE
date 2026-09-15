# Fine-tune the ModelSurgery segmentation head for low light

- Status: Done (research candidate)
- Repository scope: research checkout only

## Goal

Improve extreme-low-light segmentation using dark/noisy/colour-cast Pascal Person Parts
augmentation and stronger hand/foot losses without changing backbone, detection or pose.

## Result

Two independent one-epoch runs were blended conservatively with the original segmentation weights.
The resulting single-head candidate improves 10x-dark validation mIoU from 49.59% to 50.63%, while
normal mIoU changes from 73.61% to 73.36%. Dark feet improve from 23.63% to 27.26%. Dark hands are
effectively unchanged across the stable blend; the improvement seen in one seed did not reproduce.

Protected modules remained hash-identical. No production or TagTwo files were modified.

## Decision

Retain as a research candidate, not a production replacement. Validate on real arena captures next.
