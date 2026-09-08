# Promote a verified model through the production runtime

- Status: Dropped
- Owner: AI-engine maintainers
- Depends on: `20260903-1346-ai-multitask-encoder-repair.md` or `20260903-1347-ai-model-candidate-benchmark.md` producing a winner
- Target repository: `/home/berna/tagtwo-monorepo`

## Goal

Move only the proven winning model into the existing ai-engine runtime with reproducible artifacts,
parity, compatibility metadata, and rollback.

## Resolution

No candidate from issues 1346 or 1347 passed the all-task promotion gates. The owner also scoped
current work to the research repository, excluding production integration. Making a production
change without a winner would violate this issue's goal and constraints.

No files in `/home/berna/tagtwo-monorepo` were changed. Reopen this issue only when a future report
identifies a candidate with passing local accuracy, hit-quality, latency, deployability, and
license gates.

## Scope

- Minimal architecture/export support under `src/server/ai-engine`.
- Versioned checkpoint, ONNX, TensorRT, plugin, manifest, and hashes outside Git as appropriate.
- PyTorch/ONNX/TensorRT parity and native runtime performance.
- Hailo/APK work only when the selected deployment target requires it.

## Acceptance criteria

- [ ] The linked candidate report passes agreed accuracy and product gates.
- [ ] Export is reproducible from pinned source and checkpoint.
- [ ] Runtime parity tolerances and artifact compatibility are explicit.
- [ ] Production latency, memory, exact data path, and rollback are verified.
- [ ] Specs and tests in every affected monorepo root are aligned.

## Constraints

- Do not import research notebooks, datasets, visualizers, or abandoned code paths.
- Do not rewrite the Rust TensorRT runtime without measured need.
- Use the monorepo worktree, SPEC, shared-contract, ADR, packaging, and validation workflows.

## Validation

- Run selected ai-engine tests, native TensorRT parity, artifact packaging proof, and the live shot flow.

## Next step

Dropped with no promotion. Future work must open a new promotion issue linked to a winning report.
