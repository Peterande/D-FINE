# Promote a verified model through the production runtime

- Status: Open
- Owner: AI-engine maintainers
- Depends on: `20260903-1346-ai-multitask-encoder-repair.md` or `20260903-1347-ai-model-candidate-benchmark.md` producing a winner
- Target repository: `/home/berna/tagtwo-monorepo`

## Goal

Move only the proven winning model into the existing ai-engine runtime with reproducible artifacts,
parity, compatibility metadata, and rollback.

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

Remain open until a benchmark winner exists, then link its exact report and decide the monorepo ADR boundary.
