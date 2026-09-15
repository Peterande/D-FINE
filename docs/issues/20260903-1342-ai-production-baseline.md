# Freeze the AI production baseline

- Status: Done
- Owner: Berna / AI and vision research
- Depends on: none

## Goal

Create one machine-readable, evidence-backed definition of the exact ModelSurgery model deployed today.

## Scope

- Hash checkpoint, ONNX, TensorRT engine, plugin, and configs.
- Match Spaces `dfine_0.48.pth` to local `Traning/best_modelsurgery.pth` or prove they differ.
- Record architecture, parameters, tensor names/shapes, preprocessing, thresholds, and coordinates.
- Use typed fields for detection AP, pose OKS AP, and segmentation mIoU.

## Acceptance criteria

- [x] Production checkpoint and engine identities have SHA-256 hashes.
- [x] A manifest names all inputs, outputs, tasks, labels, and runtime requirements.
- [x] Every fact is marked measured, artifact-derived, inferred, or unknown.
- [x] Large artifacts remain outside Git at immutable recorded locations.

## Evidence and key files

- `Traning/best_modelsurgery.pth`
- `Traning/singlepass_plugin_all_gather.onnx`
- `tools/model_surgery/shared_arch.py`
- `/home/berna/tagtwo-monorepo/src/server/ai-engine/utils/model_manager.py`

## Validation

- Reconstruct the model with zero missing/unexpected checkpoint keys.
- Inspect ONNX/TensorRT bindings and compare them with the manifest.

Completed evidence:

- `benchmark/baseline/production_manifest.json`
- `benchmark/baseline/audit_manifest.py`
- Spaces `dfine_0.48.pth` and local `Traning/best_modelsurgery.pth` are byte-identical with SHA-256
  `a0d9b62b804ece6cd240740036467113a633dd9d2159aa7a7b4fc53044f4c0a7`.
- `SharedBackboneDualDecoder` reconstructed all 1,768 state tensors with zero missing and zero
  unexpected keys.
- ONNX and the TensorRT 10.15 reference engine expose one input and the same five named outputs.
- The production TensorRT 10.13 engine is hashed, but its bindings remain explicitly unknown
  because the installed 10.15 runtime correctly rejects the older serialized plan.

## Next step

Execute `20260903-1343-ai-evaluation-dataset.md` using the frozen labels and input contract.
