# Reproduce the seven-class segmentation baseline

- Status: Done
- Owner: Berna / AI and vision research
- Depends on: `20260903-1343-ai-evaluation-dataset.md`

## Goal

Re-evaluate the unchanged production backbone and segmentation head on the reconstructed Pascal
Person Parts validation split and determine whether the historical 73.64% mIoU is reproducible.

## Acceptance criteria

- [x] Use the frozen reconstructed split and original training preprocessing.
- [x] Report global-confusion mIoU, pixel accuracy, and all seven class IoUs.
- [x] Record checkpoint, dataset, code, and result hashes.
- [x] Compare against 73.64% without claiming the reconstruction is byte-identical to lost data.
- [x] Commit a supported conclusion and reproducible command.

## Next step

Test SCHP Pascal-7 as the closest public semantic segmentation control. Current baseline is
73.6079% mIoU; hands and feet are the weakest classes.

## Completion evidence

- `benchmark/harness/eval_pascal_seg.py`
- `benchmark/reports/segmentation/baseline.md`
- Full 1,829-image result: 73.6079% mIoU and 95.4991% pixel accuracy.
- Historical difference: only -0.0321 percentage points, while preserving the reconstruction caveat.
