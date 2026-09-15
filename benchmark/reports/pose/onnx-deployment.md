# RTMO-taught single-model ONNX deployment result

## Decision

The seed-456 ModelSurgery candidate now has a **quality-valid ONNX file**. The one graph contains
the existing backbone, detection, seven-class segmentation and the RTMO-taught pose adapter/head;
RTMO is not present or required at inference.

The original failure was nondeterministic tie ordering in the DETRPose encoder TopK selection.
PyTorch, ONNX and TensorRT could select similar feature values attached to different spatial
anchors. Eval-only score quantization plus an integer index tie-break makes PyTorch and ONNX choose
the same anchors without changing the training path.

## Quality

| Runtime | COCO pose AP | Difference |
|---|---:|---:|
| PyTorch, full 2,693-image validation | 51.70 | reference |
| ONNX Runtime CPU, full validation | **51.83** | +0.13 |

On a real David frame, pose keypoints have 0.0000051 mean absolute normalized difference between
PyTorch and ONNX. Segmentation pixel agreement is 99.9985%. Detection boxes have 0.00000054 mean
absolute difference on the parity control.

## Same-method GPU comparison

ONNX Runtime 1.23.2 CUDA Execution Provider, RTX 5070 Ti, batch 1, 640x640, GPU-resident input and
outputs, 50 warmups and 300 measured runs:

| Graph | p50 | p95 | p99 | FPS from p50 | GPU memory |
|---|---:|---:|---:|---:|---:|
| Current ModelSurgery | 24.077 ms | 24.137 ms | 24.167 ms | 41.53 | 1,282 MiB |
| RTMO-taught pose adapter | 24.811 ms | 24.876 ms | 24.900 ms | 40.30 | 1,280 MiB |

The adapter costs 0.734 ms p50 (3.05%) and 1.78 MB on disk. The two-process memory difference is
within measurement noise, so no measurable GPU-memory increase is claimed.

Candidate ONNX SHA-256:
`d350c8a174fa6ae2f25450796e467ef7ce5f9295314075fe3f339a87dd662cc3`.

Full ONNX predictions SHA-256:
`578654cdf689a3edab6bb066ac8221d11cc43ea2429325fd24dee47c43bb9ec5`.

## TensorRT finding

TensorRT 10.15 does not preserve the six-layer iterative pose decoder numerics on this graph. Each
isolated attention operator and one decoder layer passes, but recursive composition amplifies
backend drift; FP32 loses pose AP and FP16 emits NaNs. TensorRT engines are rejected. ONNX Runtime
CUDA is the qualified research deployment path. No TagTwo files were modified.
