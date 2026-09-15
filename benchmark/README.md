# ModelSurgery research benchmark

This directory contains the reproducible evaluation code and concise evidence for the combined
D-FINE detection, DETRPose pose and seven-class body-part segmentation model.

## Start here

1. Read [`reports/overview/recommendation.md`](reports/overview/recommendation.md) for the decision.
2. Read [`reports/overview/results.md`](reports/overview/results.md) for the measured result table.
3. Use [`reports/README.md`](reports/README.md) to find detailed evidence by topic.

## Layout

| Path | Purpose |
|---|---|
| `baseline/` | Pinned production provenance and checkpoint contract |
| `candidates/` | Candidate registries and candidate-specific runtime assets |
| `datasets/` | Dataset manifests and preparation tools; no large datasets in Git |
| `harness/` | Evaluation, rendering and runtime benchmark commands |
| `metrics/` | Shared metric aggregation |
| `pose_adapter/` | RTMO-teacher adapter model and training code |
| `segmentation/lighting/` | Low-light robustness and fine-tuning code |
| `reports/` | Human-readable results grouped by topic |
| `tests/` | Unit tests for metrics, matching and association |

Large checkpoints, ONNX files, videos and raw predictions stay under ignored `.cache/`, `runs/`,
or `ZKeepResults`. Reports record their hashes. Nothing in this benchmark directory changes the
TagTwo production runtime.
