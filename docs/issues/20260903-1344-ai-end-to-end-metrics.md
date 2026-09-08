# Implement end-to-end AI model metrics

- Status: Done
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

- [x] One command evaluates any conforming model adapter on the fixed corpus.
- [x] Raw outputs are retained independently from rendered reports.
- [x] Reports contain model, dataset, code, settings, and environment hashes.
- [x] Comparisons fail when inputs, thresholds, preprocessing, or protocols differ.
- [x] Locally measured results and paper claims are never mixed.

## Evidence and key files

- `benchmark/harness/`
- `benchmark/MODEL_BASELINE.md`
- `/home/berna/tagtwo-monorepo/src/server/ai-engine/utils/hit_detector.py`

## Validation

- Run the same baseline twice and require stable results within declared tolerance.
- Use golden fixtures for metric and report-schema calculations.

Completed evidence:

- `benchmark/run_evaluation.py` loads any `module:function` adapter and writes immutable JSONL raw
  output separately from its normalized report.
- `benchmark/metrics/core.py` implements global-confusion segmentation metrics, hit/body-part and
  decision-source metrics, plus p50/p95 stage latency and peak memory.
- Existing pinned COCO and domain evaluators provide detection AP/recall/FP/localization and pose
  OKS AP without mixing those local measurements with paper claims.
- Dataset, settings, and preprocessing form a protocol fingerprint. `compare_reports.py` refuses
  comparison when that fingerprint differs while allowing different model identities.
- Golden fixture tests cover confusion math, hits, body parts, timing percentiles, and protocol
  fingerprint stability. Two complete fixture runs produced identical raw and metric results.

## Next step

Execute `20260903-1345-ai-baseline-contract-drift.md` with the frozen protocol.
