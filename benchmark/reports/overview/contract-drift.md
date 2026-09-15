# Baseline contract drift — measured research verdict

## Preprocessing

The controlled COCO val2017 experiment used the same stock D-FINE-X checkpoint, 5,000 images,
square resize, output decoder, and evaluator. The only changed variable was ImageNet normalization.

| Input after RGB resize | COCO AP | AP50 | AP75 | AR100 |
|---|---:|---:|---:|---:|
| `/255` only | 59.315% | 76.823% | 64.621% | 77.250% |
| `/255`, then ImageNet mean/std | 40.503% | 54.339% | 43.566% | 63.458% |

ImageNet normalization costs **18.812 AP**. The stock control without normalization reproduces the
published 59.3 AP. The research contract is therefore RGB, direct 640x640 resize, float `/255`,
NCHW, and **no mean/std normalization**.

Raw prediction artifacts remain outside Git:

- `runs/coco_det_stock.json`, SHA-256
  `820c6810da875436991dad0bce6cc68a7aac417d79775fdc7d471b44b602746f`
- `runs/coco_det_stock_imagenet_normalized.json`, SHA-256
  `3213254d1ea9fa38cfcbfa41e23148cc08bba6dd7736a25a0af5a7c4d13d0c3c`

## Anatomical classes

The owner and checked-in segmentation configs define:

`0 background, 1 head, 2 torso, 3 arms, 4 hands, 5 legs, 6 feet`.

The deterministic PASCAL-Part reconstruction maps fine-grained source annotations into exactly
those IDs. The production Python and Rust runtimes currently label IDs 3..6 as upper/lower
arms/legs. The numeric tensors are unaffected, but emitted body-part names and damage semantics may
be wrong.

## Coordinates

All inspected paths use a direct, non-letterboxed 640x640 resize. The crosshair is the image centre,
which remains the same normalized point under the stretch. The ONNX and inspected TensorRT 10.15
engine have identical input/output shapes. Dense segmentation is evaluated at model coordinates;
box/keypoint postprocessing maps normalized coordinates to the requested width/height.

Raw ONNX-to-TensorRT comparison on `David.mp4` frame 120 found near-exact segmentation agreement
(cosine 1.0, MAE 0.00584), but large detection/pose differences. The rewritten plugin ONNX cannot
run in ONNX Runtime, and the available unrewritten ONNX is not proven to be the exact parent of the
inspected engine. Full graph parity is therefore **not proven**.

## Metric metadata

The `miou48` filename token is not accepted as evidence. The typed research manifest records:

- detection COCO AP: 0.436, locally measured
- pose COCO OKS AP: 0.482, locally measured
- source segmentation validation mIoU: 0.7364, checkpoint-derived and not yet reproduced

Production still ranks/discovers artifacts using the ambiguous filename token. Correcting that and
the Python/Rust class/preprocessing contracts requires a focused monorepo change with its own SPEC,
tests, and ADR decision; it is not silently changed from this research checkout.
