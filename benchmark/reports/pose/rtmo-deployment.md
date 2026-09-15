# RTMO-L deployment gate

The official Body7 RTMO-L ONNX was converted to FP16 TensorRT 10.15.1 on the RTX 5070 Ti.

| Metric | Result |
|---|---:|
| Local COCO OKS AP | 64.75 |
| TensorRT FP16 COCO OKS AP | 64.70 |
| Current merged pose AP | 48.20 |
| TensorRT model p50 | 2.224 ms |
| TensorRT model p95 | 2.252 ms |
| TensorRT model mean | 2.226 ms |
| Engine execution memory | 780.49 MiB |
| ONNX/TRT keypoint MAE | 0.0869 |
| ONNX/TRT keypoint max absolute | 1.2425 px |

Engine SHA-256:
`ba330a8af430a3746fa708dd9a8195deba58511baea0447e801525228253fb92`.
Latency/parity report SHA-256:
`5acc3bd615b1325690f0f3569978db00cefb74158294c70ae604d0bb5e5e10d1`.

RTMO is one-stage, so neural inference does not multiply per detected person; only its dynamic NMS
output size changes. The measurement is engine execution only and must not be presented as full
camera-to-result latency.

## Decision

RTMO-L is a **qualified pose candidate**, not yet a replacement for the combined graph. It provides
a locally measured 16.55-point pose improvement with a viable TensorRT execution time. The next
architecture decision is whether the extra ~2.2 ms and ~780 MiB are acceptable as a separate pose
engine, or whether its outputs should supervise a newly designed shared pose head.

Full FP16 evaluation uses the official second-stage NMS and differs from ONNX by only 0.05 AP.
