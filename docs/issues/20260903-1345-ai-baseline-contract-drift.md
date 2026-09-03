# Correct AI baseline contract drift

- Status: Open
- Owner: Berna / AI and vision research
- Depends on: `20260903-1342-ai-production-baseline.md`; metrics may be developed in parallel

## Goal

Remove known correctness ambiguity before candidate model comparisons.

## Problems to reproduce

- Stock D-FINE reproduced 59.3 AP without ImageNet normalization, while monorepo Python runtime applies it.
- Crosshair and dense mask coordinate alignment differs between paths.
- Training describes IDs 3-6 as arms/hands/legs/feet; runtime says upper/lower arms/legs.
- Runtime derives `miou48` from a filename rather than typed checkpoint metadata.

## Acceptance criteria

- [ ] Correct preprocessing is proven by a controlled one-variable experiment.
- [ ] PyTorch, ONNX, TensorRT, Python, and Rust coordinate transforms agree.
- [ ] The class map is traced to annotation generation, not comments alone.
- [ ] Artifact metrics are typed and no longer inferred from filenames.
- [ ] Each correction has an exact regression test in the owning repository.

## Constraints

- Reproduce before fixing.
- Do not preserve two conflicting class maps as compatibility behavior.
- Any shared contract change in `tagtwo-monorepo` requires its contract/ADR workflow.

## Validation

- Re-run the exact failing parity or control proof after every correction.
- Run ai-engine multihead and hit-detector tests for any promoted runtime fix.

## Next step

Run stock D-FINE twice with identical inputs, changing only ImageNet normalization, and compare COCO AP and raw tensors.
