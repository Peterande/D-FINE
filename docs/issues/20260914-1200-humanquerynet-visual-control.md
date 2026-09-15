# Evaluate HumanQueryNet as a multitask visual control

- Status: Done (visual compatibility gate)
- Repository scope: research checkout only

## Goal

Run the closest published single-stage human multitask model on the same David and SogO controls
and determine whether it is a direct replacement candidate.

## Result

The official DSPC checkpoint successfully produced detection, pose, person instance masks and
attributes. Side-by-side videos were generated with one-in-ten temporal sampling because the
released CUDA/PyTorch 1.13 environment does not support the RTX 5070 Ti and had to run on CPU.

HumanQueryNet segmentation is a binary person instance mask, not anatomical body-part parsing.
It therefore cannot drive the current seven-class hit logic without a new parsing head and
retraining. The repository also restricts code/data use to non-commercial purposes absent a
separate agreement.

## Decision

Keep HumanQueryNet as the primary paper architecture baseline, not a product replacement. A metric
comparison requires a shared multitask dataset and a modern-runtime port. No production files were
modified.
