# Match pose queries before teacher distillation

- Status: Done
- Owner: Berna / AI and vision research
- Depends on: `20260904-0701-isolated-pose-adapter-distillation.md`
- Repository scope: research checkout only

## Goal

Test whether comparing unordered DETR queries by index caused the failed pose-adapter experiment.

## Acceptance criteria

- [x] Matching recovers a known synthetic query permutation.
- [x] Baseline, query-index and matched runs use identical data, seed and optimizer settings.
- [x] Protected modules remain hash-identical.
- [x] The candidate is screened against the same 200-image pose subset.
- [x] Full evaluation runs only if the candidate beats the 47.1 screen baseline.
- [x] Results include reproducible settings and artifact hashes.

## Next step

No-change conclusion. Matching fixed the distillation objective but the adapter still scored 43.6
AP versus the 47.1 baseline. Test RTMO-L as the independent pose-family control next.
