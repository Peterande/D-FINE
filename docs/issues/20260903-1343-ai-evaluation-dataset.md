# Establish the canonical AI evaluation dataset

- Status: Open
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

- [ ] Train, validation, and product-acceptance splits cannot leak into each other.
- [ ] Scenario coverage and missing annotation coverage are quantified.
- [ ] Seven-class masks follow the proven authoritative label map.
- [ ] An agent can reproduce exact sample ordering and hashes without tribal knowledge.

## Evidence and key files

- `benchmark/datasets/`
- `/home/berna/D-FINE-NEWBRINGER/datasets/`
- `/home/berna/tagtwo-monorepo/src/e2e/fixtures/`

## Validation

- Run a manifest audit for schema, hashes, duplicates, split leakage, and label ranges.
- Double-review representative annotations from every required scenario.

## Next step

Write the dataset inventory with location, sample count, annotation types, provenance, and usage rights.
