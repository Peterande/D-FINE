# Benchmark alternative detection, pose, and segmentation models

- Status: Open
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

- [ ] A registry records official source, revision, weights, license, task, metric, and deployment constraints.
- [ ] Cheap compatibility/rejection gates precede expensive training.
- [ ] Every surviving candidate uses the same data and metric protocol.
- [ ] Every candidate ends with promote, retrain, monitor, or reject plus evidence.
- [ ] A candidate cannot win by improving one head while hiding product regressions.

## Constraints

- Use primary papers and official implementations for external claims.
- Separate published from locally reproduced results.
- Do not compare instance segmentation directly to body-part semantic segmentation.

## Validation

- Adapter contract smoke test, then complete evaluation for survivors.
- Visual review is diagnostic evidence, never the sole acceptance proof.

## Next step

Create the candidate registry and classify each candidate by task compatibility and expected deployment risk.
