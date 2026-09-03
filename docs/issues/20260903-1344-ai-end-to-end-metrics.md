# Implement end-to-end AI model metrics

- Status: Open
- Owner: Berna / AI and vision research
- Depends on: `20260903-1342-ai-production-baseline.md`, `20260903-1343-ai-evaluation-dataset.md`

## Goal

Evaluate models repeatably across individual heads, product hit quality, and runtime cost.

## Required metrics

- Detection: COCO AP plus person recall, false positives, and localization.
- Pose: OKS AP and product-relevant joint quality.
- Segmentation: global-confusion-matrix mIoU and every class IoU.
- Product: direct hit, body part, occlusion, false hit, missed hit, and decision source.
- Runtime: preprocess/inference/postprocess/end-to-end p50 and p95, memory, and hardware/software.

## Acceptance criteria

- [ ] One command evaluates any conforming model adapter on the fixed corpus.
- [ ] Raw outputs are retained independently from rendered reports.
- [ ] Reports contain model, dataset, code, settings, and environment hashes.
- [ ] Comparisons fail when inputs, thresholds, preprocessing, or protocols differ.
- [ ] Locally measured results and paper claims are never mixed.

## Evidence and key files

- `benchmark/harness/`
- `benchmark/MODEL_BASELINE.md`
- `/home/berna/tagtwo-monorepo/src/server/ai-engine/utils/hit_detector.py`

## Validation

- Run the same baseline twice and require stable results within declared tolerance.
- Use golden fixtures for metric and report-schema calculations.

## Next step

Define versioned normalized raw-output and evaluation-report schemas.
