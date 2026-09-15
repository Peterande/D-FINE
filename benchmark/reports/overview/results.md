# ModelSurgery research results

This is the authoritative result table. Detailed evidence is indexed in [`../README.md`](../README.md).

## Current model

One 640x640 graph shares an HGNetv2-B5 backbone across D-FINE detection, DETRPose pose and a
seven-class FPN/ASPP body-part head (`background`, `head`, `torso`, `arms`, `hands`, `legs`, `feet`).

| Task | Current result | Decision |
|---|---:|---|
| Detection | 43.60 COCO AP | Keep D-FINE path |
| Pose | 48.20 COCO OKS AP | Improved candidate available |
| Body-part segmentation | 73.61% Pascal Person Parts mIoU | Keep current head |

## Candidate results

| Candidate | Relevant result | Outcome |
|---|---:|---|
| Stock D-FINE-X | 59.31 detection AP | Teacher/control; not a combined replacement |
| RF-DETR-Large | 56.54 detection AP | Lower domain recall; rejected |
| DETRPose-X | 74.41 pose AP | Teacher/control; direct transplant failed |
| RTMO-L | 64.75 pose AP | Standalone control; separate runtime rejected as final design |
| SCHP Pascal-7 remap | 49.07 mIoU | Cannot represent hands/feet correctly; rejected |
| HumanQueryNet | Visual test only | Binary person mask, not anatomical parsing; rejected |

## Selected research improvement

RTMO-L was used only as an offline teacher for an isolated pose feature adapter. Runtime remains one
ModelSurgery graph; RTMO is not present at inference.

| Pose run | Full COCO AP |
|---|---:|
| Current ModelSurgery | 48.20 |
| Seed 123 | 51.10 |
| Seed 456 | 51.70 |
| Seed 789 | 51.50 |
| Three-seed mean | **51.43** |
| Qualified seed-456 ONNX | **51.83** |

Detection, segmentation, backbone and encoder remain protected.

## Deployment comparison

ONNX Runtime 1.23.2 CUDA, RTX 5070 Ti, batch 1, 640x640, GPU-resident I/O:

| Graph | p50 | p95 | FPS from p50 | GPU memory |
|---|---:|---:|---:|---:|
| Current ModelSurgery | 24.077 ms | 24.137 ms | 41.53 | 1,282 MiB |
| RTMO-taught adapter | 24.811 ms | 24.876 ms | 40.30 | 1,280 MiB |

The adapter costs 0.734 ms p50 (3.05%) and has no measurable memory increase. TensorRT 10.15 is
rejected because it does not preserve pose quality; ONNX Runtime CUDA is the qualified path.

## Segmentation robustness

The current head scores 73.61% normal-light mIoU and 49.59% at one-tenth brightness. Low-light
training raises the latter to 50.63% but lowers normal mIoU to 73.36%. Camera exposure/noise control
therefore offers more likely product value than this small model change.

## Final answer

No tested model replaces the entire ModelSurgery system. Keep current detection and segmentation.
The only accepted research improvement is the RTMO-taught pose adapter through ONNX Runtime CUDA.
Visual improvement on office videos is modest, so no production change is made from this branch.

No TagTwo files were modified.
