# Correct AI baseline contract drift

- Status: Done
- Owner: Berna / AI and vision research
- Depends on: `20260903-1342-ai-production-baseline.md`; metrics may be developed in parallel

## Goal

Remove known correctness ambiguity before candidate model comparisons.

Production runtime correction is explicitly deferred by the owner. This issue freezes the
research-side contract only; it does not claim that `tagtwo-monorepo` has changed.

## Problems to reproduce

- Stock D-FINE reproduced 59.3 AP without ImageNet normalization, while monorepo Python runtime applies it.
- Crosshair and dense mask coordinate alignment differs between paths.
- Training describes IDs 3-6 as arms/hands/legs/feet; runtime says upper/lower arms/legs.
- Runtime derives `miou48` from a filename rather than typed checkpoint metadata.

## Acceptance criteria

- [x] Correct research preprocessing is proven by a controlled one-variable experiment.
- [x] Research PyTorch/ONNX/TensorRT coordinate assumptions are recorded with unknown parity explicit.
- [x] The class map is traced to deterministic annotation generation, not comments alone.
- [x] Research artifact metrics are typed and no longer inferred from filenames.
- [x] Each research correction has an exact local regression test.

## Constraints

- Reproduce before fixing.
- Do not preserve two conflicting class maps as compatibility behavior.
- Any shared contract change in `tagtwo-monorepo` requires its contract/ADR workflow.

## Validation

- Re-run the exact failing parity or control proof after every correction.
- Run ai-engine multihead and hit-detector tests for any promoted runtime fix.

Measured findings:

- Full COCO val2017 controlled run: `/255` only = 0.59315 AP; ImageNet normalization = 0.40503 AP.
  The production normalization is a measured 18.812-point regression for stock D-FINE-X.
- The research class map is corrected to the owner/config-defined
  `background, head, torso, arms, hands, legs, feet` and backed by the deterministic annotation
  converter. Production Python/Rust still emit conflicting names for IDs 3..6.
- ONNX and TensorRT binding shapes agree. Segmentation raw parity is close, but detection and pose
  are not; exact graph lineage is missing, so full numerical parity is not proven.
- Typed metric provenance is frozen in `benchmark/baseline/production_manifest.json`; production
  still interprets the ambiguous `miou48` filename token.
- Full evidence and raw-output hashes are in `benchmark/CONTRACT_DRIFT.md`.

Deferred outside this issue:

- Production Python/Rust preprocessing, names, metadata, and exact engine parity remain unchanged
  and must be revalidated only if a candidate is later promoted.

## Next step

Execute `20260903-1346-ai-multitask-encoder-repair.md` against this corrected research contract.
