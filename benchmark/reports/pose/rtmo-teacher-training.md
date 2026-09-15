# RTMO pseudo-label distillation result

RTMO-L was used only during offline training. Inference remains the existing combined
`SharedBackboneDualDecoder` plus an isolated pose feature adapter; no second RTMO engine is required.

## Reproducible run

- Teacher: pinned RTMO-L Body7 ONNX already qualified in [`rtmo-deployment.md`](rtmo-deployment.md)
- Training data: first 200 person-containing COCO train2017 images, 686 accepted poses
- Training: 100 steps, batch 2, seed 123, learning rate 1e-5, adapter only
- Validation: separate COCO val2017 split
- Candidate checkpoint SHA-256: `fe30d426c83d81fdb696ef5047ffcb375871c88e03b44bb7187be1c001e376b7`
- Pseudo-label SHA-256: `bdc55b36f734d64ba708a5ab952bc3ba930f80729e26826c8bf9ddd9bbd2d841`
- Full prediction SHA-256: `260ecb1de905690e4ba724737ea100f72340b7f7d5d56880e9db76e7e1f3acbe`

| Model | Screen AP | Full AP |
|---|---:|---:|
| Current combined pose | 47.1 | 48.2 |
| RTMO pseudo-trained adapter | **48.1** | **49.2** |

The +1.0 AP result is promising but based on a small teacher subset. It establishes that RTMO can
teach the pose branch without becoming a separate runtime model. It does not yet justify replacing
the current checkpoint. Detection and segmentation are structurally protected and recorded as
hash-identical by the training checkpoint.

Visual control: `ZKeepResults/research_20260904/david_single_model_rtmo_taught.mp4`.

## Full-scale seed 123

The positive gate was followed by a three-epoch adapter-only run over all 54,876 usable training
images (257,911 RTMO pseudo poses, batch 4, 41,157 steps). Full COCO validation improved to
**51.1 AP**, or **+2.9 AP** over the 48.2 current-model baseline.

- Full pseudo-label SHA-256: `645e6c5bbec4436cd2da3d956713734d0059a03a5e73b524301193acd3c54f56`
- Candidate checkpoint SHA-256: `87b9830b5824acf715aecd86c074b450d4953b49c3c940bca7c0661330f59934`
- Validation predictions SHA-256: `7cdc0d2cbdd91de500d78b7578557cca67f776358773dd850e816ef28765ac3e`

This confirms the improvement for seed 123. Independent seeds are still required to quantify run
variance before the candidate can be called production-ready.

## Three-seed result

| Seed | Full COCO pose AP | Gain over 48.2 |
|---:|---:|---:|
| 123 | 51.1 | +2.9 |
| 456 | 51.7 | +3.5 |
| 789 | 51.5 | +3.3 |
| **Mean** | **51.43** | **+3.23** |

The range is 0.6 AP and the sample standard deviation is 0.31 AP. All three independent runs beat
the baseline, so the quality improvement is reproducible. Seed 456 is the current best candidate.

- Seed 456 checkpoint: `5bf3c6742877cc80ca57ad9f6de1a426aed3e9977574209b6795a0fe1c4ace68`
- Seed 456 predictions: `606fa938df6d92f17ff35c28f37f4e7907b18a0c8bda67c2b7615bbb8dc944dc`
- Seed 789 checkpoint: `892bdd11106257d8d9c97f5dcc309e809ab8f15419270f242fd0473571451365`
- Seed 789 predictions: `6da78a28e96bc77c3b4caae0c4aebfe552edfe0179b66debded0ba4876895795`

The remaining gates are export/latency parity and office-video inspection. No production files have
been changed.
