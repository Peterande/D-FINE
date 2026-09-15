# Export and qualify the RTMO-taught pose adapter

- Status: Done (ONNX passed; TensorRT rejected)
- Repository scope: research checkout only

## Goal

Export the best 51.7-AP single-model pose candidate and verify ONNX/TensorRT quality, latency and
memory without changing production.

## Result

ONNX and TensorRT graphs build, and same-method FP16 latency is unchanged at 5.619 ms median. The
pose export is invalid: FP32 TensorRT scores 1.11 AP and FP16 emits NaNs, while detection and
segmentation retain numerical parity. The candidate is rejected for deployment.

## Decision

Do not use the exported engine or its latency as a production claim. Repair DETRPose ONNX
deformable-attention parity first, with a real-frame parity test before engine construction. No
TagTwo files were modified.

## Follow-up resolution

The ONNX failure was repaired with deterministic eval-only TopK selection. Full ONNX Runtime
evaluation scores 51.83 AP versus 51.70 PyTorch. CUDA A/B latency is 24.077 ms p50 current versus
24.811 ms candidate (+0.734 ms / 3.05%). TensorRT remains invalid and is explicitly rejected;
ONNX Runtime CUDA is the accepted research deployment path. The ONNX portion of the gate is done.
