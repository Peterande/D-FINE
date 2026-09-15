# Evaluation datasets

`inventory.json` records every locally available research source, its annotation type, intended
use, and known rights. `tagtwo_v1/manifest.json` freezes the first product-acceptance frame list
without copying private video into Git.

Run the deterministic identity/leakage/coverage audit with:

```bash
python benchmark/datasets/audit_manifest.py
```

The product sample list deliberately reports missing product-specific annotations. Detection and
pose ground truth come from hash-pinned COCO/soldier validation files. Seven-class segmentation
ground truth comes from the deterministic Pascal Person Parts reconstruction under `.cache/`.
Predictions from the current model are never used as ground truth.

Rebuild the external anatomical dataset from the two hash-pinned official archives with
`prepare_pascal_person_parts.py`. The converter records ordered split hashes and refuses unknown
person-part names. The reconstruction is not claimed to be byte-identical to the lost preparation.
