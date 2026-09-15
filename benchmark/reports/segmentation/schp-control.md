# SCHP Pascal-7 comparison

Full 1,829-image evaluation using the publisher's INT8 ONNX artifact:

| Model | Project-semantic mIoU |
|---|---:|
| Current segmentation head | **73.61%** |
| SCHP Pascal-7 | **49.07%** |

SCHP uses upper/lower limbs instead of arms/hands/legs/feet. After the only valid remap it cannot
predict hands or feet, producing 0% IoU for both. Raw ID mIoU (51.60%) is reported but is not a
semantically valid comparison. SCHP is rejected as a replacement.

Result SHA-256: `0c15456131bd953fa470df78a2a44c3e6b1a553519884b69c1d1a70874bdc1a5`.
Model SHA-256: `66b12766d7f1ddbc3de972e67e8626be727507e7feeeca34e1b23b6f45e756d2`.
