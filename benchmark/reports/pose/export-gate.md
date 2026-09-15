# Pose adapter export gate

| Runtime | Result | Decision |
|---|---:|---|
| PyTorch | 51.70 pose AP | Reference |
| ONNX Runtime | 51.83 pose AP | Pass |
| TensorRT FP32/TF32 | Pose quality not preserved | Reject |
| TensorRT FP16 | NaN pose outputs | Reject |

Undefined tie ordering in DETRPose encoder TopK initially broke ONNX anchor selection. Eval-only
score quantization plus an integer index tie-break restored parity without changing training.
Real-frame keypoint mean absolute difference is 0.0000051 normalized units; segmentation pixel
agreement is 99.9985%.

TensorRT remains invalid because its six-layer iterative pose decoder amplifies small backend
numeric differences even though isolated attention and individual decoder layers pass. Failed
TensorRT engines are not release artifacts. See [`onnx-deployment.md`](onnx-deployment.md).

No TagTwo files were modified.
