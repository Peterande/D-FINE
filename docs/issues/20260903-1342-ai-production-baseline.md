# Freeze the AI production baseline

- Status: Open
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

- [ ] Production checkpoint and engine identities have SHA-256 hashes.
- [ ] A manifest names all inputs, outputs, tasks, labels, and runtime requirements.
- [ ] Every fact is marked measured, artifact-derived, inferred, or unknown.
- [ ] Large artifacts remain outside Git at immutable recorded locations.

## Evidence and key files

- `Traning/best_modelsurgery.pth`
- `Traning/singlepass_plugin_all_gather.onnx`
- `tools/model_surgery/shared_arch.py`
- `/home/berna/tagtwo-monorepo/src/server/ai-engine/utils/model_manager.py`

## Validation

- Reconstruct the model with zero missing/unexpected checkpoint keys.
- Inspect ONNX/TensorRT bindings and compare them with the manifest.

## Next step

Fetch the configured Spaces checkpoint through existing tooling and compare its SHA-256 with `Traning/best_modelsurgery.pth`.
