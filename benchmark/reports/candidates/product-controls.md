# RTMO and RF-DETR product controls

## RTMO-L

- Full local pose quality: 64.75 OKS AP versus 48.2 for merged baseline.
- David CPU ONNX: 176.9 ms mean, 175.4 ms p50, 190.5 ms p95 over 563 frames.
- SogO CPU ONNX: 124.1 ms mean, 96.1 ms p50, 185.7 ms p95 over 706 frames.
- Quality is promising; CPU latency is not production-suitable. GPU/TensorRT remains required.

## RF-DETR-Large

Soldier-domain validation at score 0.5:

| Metric | Merged baseline | RF-DETR |
|---|---:|---:|
| Recall | 86.3% | 84.2% |
| False positives/image | 0.41 | 0.17 |
| Matched box IoU | 0.893 | 0.908 |

RF-DETR is more conservative and precise but misses more people. Together with its lower full COCO
AP than stock D-FINE-X, it is not a general replacement winner.
