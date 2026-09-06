# Benchmark alternative detection, pose, and segmentation models

- Status: Done
- Owner: Berna / AI and vision research
- Depends on: issues 1342 through 1345; compare against issue 1346 when its repaired baseline is ready

## Goal

Determine whether current model families offer meaningful product value over both deployed and
repaired ModelSurgery baselines.

## Scope

- Detection/backbone candidates including DEIMv2 and other current official releases.
- Pose heads that can attach without corrupting detection representations.
- Semantic body-part segmentation compatible with the required seven classes.
- EdgeCrafter only where its outputs are genuinely comparable or trainable for this task.
- TensorRT, memory, license, integration cost, and relevant edge feasibility.

## Acceptance criteria

- [x] A registry records official source, revision, weights, license, task, metric, and deployment constraints.
- [x] Cheap compatibility/rejection gates precede expensive training.
- [x] Every surviving candidate uses the same data and metric protocol.
- [x] Every candidate ends with promote, retrain, monitor, or reject plus evidence.
- [x] A candidate cannot win by improving one head while hiding product regressions.

## Constraints

- Use primary papers and official implementations for external claims.
- Separate published from locally reproduced results.
- Do not compare instance segmentation directly to body-part semantic segmentation.

## Validation

- Adapter contract smoke test, then complete evaluation for survivors.
- Visual review is diagnostic evidence, never the sole acceptance proof.

Completed evidence:

- `benchmark/candidates/registry.json` pins seven candidate paths and official revisions.
- `benchmark/candidates/REPORT.md` records the gates and final disposition for every candidate.
- Official DETRPose-X reproduced 74.41 OKS AP locally, but a direct shared-encoder transplant
  collapsed to 0.51 AP and is rejected without retraining/isolation.
- Sapiens was locally compared as a semantic parsing teacher, not treated as ground truth.
- DEIMv2 and EdgeCrafter were stopped before expensive runs because their default licenses forbid
  commercial deployment and their task graphs are not drop-in compatible. Published claims remain
  clearly separate from local results.
- No candidate passes every task gate; supported conclusion is no promotion.

## Next step

Retain the current model. If research continues, train an isolated pose adapter using official
DETRPose-X as teacher; do not transplant the decoder directly.
