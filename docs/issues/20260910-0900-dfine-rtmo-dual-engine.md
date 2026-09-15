# Build and evaluate the D-FINE plus RTMO dual-engine prototype

- Status: Done
- Repository scope: research checkout only

## Goal

Combine current D-FINE detection/segmentation with RTMO-L pose and measure full TensorRT accuracy,
sequential latency, memory, and det-to-pose association without changing production.

## Acceptance criteria

- [x] RTMO FP16 TensorRT is evaluated on the full pinned COCO pose split.
- [x] TensorRT quality is compared with RTMO ONNX and current pose baselines.
- [x] A research-only dual-engine path runs on David and SogO.
- [x] Combined model latency, memory, and multi-person association design are reported.
- [x] Artifacts remain outside Git and results are hash-pinned.

## Next step

Research complete. Recommend D-FINE detection/segmentation plus RTMO-L FP16 pose; production
integration requires a separately authorized task.
