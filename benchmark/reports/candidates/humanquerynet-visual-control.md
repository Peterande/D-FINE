# HumanQueryNet visual control

HumanQueryNet was run from the official COCO-UniHuman repository at revision
`027519359f034c8d72d6b09bd21f522b76513249`. The official DSPC checkpoint SHA-256 is
`59eba3313a91f99db935f43cc94f09ba11259ab829d8a5ace822b8825260b6ed`.

## What was verified

- One HumanQueryNet graph emits person detection, 17-point pose, person instance masks and
  gender/age attributes.
- David and SogO were rendered at one-in-ten temporal sampling while preserving video duration.
- The released mask is a binary person instance mask, not seven-class anatomical human parsing.
- The official environment is Python 3.10/PyTorch 1.13/MMDetection 2.25/MMCV 1.7.
- Its CUDA build does not support the local RTX 5070 Ti (`sm_120`), so visual inference used CPU.

The isolated 20-frame CPU smoke measured 4,729 ms/model frame. Parallel full-video rendering
measured 7,595 ms/frame on David and 7,249 ms/frame on SogO due to CPU contention. These numbers
describe legacy-environment compatibility, not deployable HQNet latency, and must not be compared
with the TensorRT baseline.

## Verdict

HumanQueryNet is a valid paper-level architectural baseline because it shares human queries across
detection, pose and segmentation. It is not a drop-in product replacement: its mask cannot identify
TagTwo anatomical hit classes, its official code requires a legacy runtime, and its published code
and data are non-commercial unless separately licensed. A fair metric comparison would require
training/evaluating both models on COCO-UniHuman or another shared multitask dataset.

Visual controls:

- `ZKeepResults/research_20260904/david_current_vs_humanquerynet.mp4`
- `ZKeepResults/research_20260904/sogo_current_vs_humanquerynet.mp4`
