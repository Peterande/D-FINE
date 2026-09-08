# Establish the canonical AI evaluation dataset

- Status: Done
- Owner: Berna / AI and vision research
- Depends on: `20260903-1342-ai-production-baseline.md` for labels and input contract

## Goal

Give every baseline and candidate identical representative Tagtwo inputs and ground truth.

## Scope

- Inventory COCO, soldier, Pascal-part, camouflage, and real gun-camera data.
- Cover distance, lighting, movement, camouflage, multiple people, partial visibility, and occlusion.
- Version sample IDs, source hashes, splits, boxes, classes, COCO-17 keypoints, body masks,
  crosshair positions, and hit outcomes where available.
- Store private or large media externally; commit manifests only.

## Acceptance criteria

- [x] Train, validation, and product-acceptance splits cannot leak into each other.
- [x] Scenario coverage and missing annotation coverage are quantified.
- [x] Seven-class masks follow the proven authoritative label map.
- [x] An agent can reproduce exact sample ordering and hashes without tribal knowledge.

## Evidence and key files

- `benchmark/datasets/`
- `/home/berna/D-FINE-NEWBRINGER/datasets/`
- `/home/berna/tagtwo-monorepo/src/e2e/fixtures/`

## Validation

- Run a manifest audit for schema, hashes, duplicates, split leakage, and label ranges.
- Double-review representative annotations from every required scenario.

Current evidence:

- `benchmark/datasets/inventory.json` inventories all available local datasets, annotation types,
  split sizes, intended use, and known licensing constraints.
- `benchmark/datasets/tagtwo_v1/manifest.json` freezes 20 deterministic frames from `David.mp4`
  and `SogO.mp4` by source SHA-256 and zero-based frame index.
- `benchmark/datasets/audit_manifest.py` validates hashes, ordering, frame/crosshair bounds,
  source-group split leakage, scenario coverage, and any supplied body-part label ranges.
- COCO detection/pose validation and soldier-domain validation annotation files are hash-pinned.
- Available MCS1K, CAMO, and COD10K prepared masks are binary camouflage masks, not seven-class
  anatomical ground truth.
- The model owner confirms that the missing segmentation source was Pascal Person Parts with
  `background, head, torso, arms, hands, legs, feet`; `dfine_0.73.pth` records 73.64% validation
  mIoU on that prepared dataset.
- Camouflage coverage is explicitly deferred from the current v1 scope by the model owner.

Completion evidence:

- Official PASCAL-Part annotations (MD5 matches the publisher's `2fa0a19ee9b5e43b2bee520166111120`)
  and VOC2010 train/validation images were downloaded into ignored `.cache/datasets` storage.
- `prepare_pascal_person_parts.py` deterministically rebuilt 1,714 train and 1,829 validation
  examples with labels `background, head, torso, arms, hands, legs, feet` and zero split overlap.
- Every label 0..6 occurs in both splits. Two stratified visual review passes covered 16 validation
  masks, including multi-person, small-person, indoor, outdoor, and low-light examples.
- Product-specific low-light/outdoor recordings and all missing per-shot annotations remain
  quantified gaps, not hidden ground truth. They are not required to reproduce the current public
  model metrics and will be expanded under anatomical-hit validation.

## Next step

Execute `20260903-1344-ai-end-to-end-metrics.md` on the frozen manifests.
