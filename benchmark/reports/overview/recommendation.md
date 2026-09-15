# Final research recommendation

Keep the current production model unchanged. Retain one research candidate: the RTMO-taught
single-graph pose adapter exported through ONNX Runtime CUDA.

| Area | Decision | Reason |
|---|---|---|
| Detection | Keep current D-FINE path | No replacement improved the complete system |
| Pose | Retain RTMO-taught adapter | 51.43 mean AP vs 48.20 across three seeds |
| Segmentation | Keep current seven-class head | 73.61% mIoU; external control was worse |
| Runtime | Use ONNX Runtime CUDA for candidate tests | 51.83 AP, +0.734 ms p50 |
| Product | Do not promote yet | Office-video improvement is visually modest |

RTMO is an offline teacher only and is not required at inference. TensorRT is rejected because it
does not preserve pose quality. No TagTwo files were modified.
