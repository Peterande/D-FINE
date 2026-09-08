# Map the current real-time vision model landscape

- Status: Done
- Owner: Berna / AI and vision research
- Depends on: `20260903-1342-ai-production-baseline.md`, `20260903-1344-ai-end-to-end-metrics.md`
- Repository scope: research checkout only

## Goal

Produce an evidence-backed answer to what newer detection, human-pose, semantic body-part
segmentation, and genuine multitask models exist; how each would fit the current
`SharedBackboneDualDecoder`; and why each should be tested, retrained, monitored, or rejected.

## Scope

- Search current primary papers and official repositories for relevant 2024–2026 candidates.
- Cover detection, pose, Pascal-style human parsing, and combined box/mask/pose approaches.
- Record exact source revision, checkpoint, training data, task definition, license, latency hardware,
  export support, commercial-use constraints, and integration cost.
- Distinguish standalone task results from models that actually share computation across tasks.
- Apply cheap gates before downloading weights or starting training.
- Locally reproduce surviving candidates on the frozen protocol where feasible.

## Required questions per candidate

- Is its published metric comparable to ours?
- Does it improve the deployed baseline, the stronger standalone source, or neither?
- Can it reuse the current backbone, encoder, and head interfaces?
- What must be retrained or distilled?
- Does it support ONNX and TensorRT with acceptable operators and precision?
- What are the parameter, memory, latency, data, licensing, and integration costs?
- Which existing task or hit behavior could regress?

## Acceptance criteria

- [x] Candidate registry covers at least five credible models per task where five exist.
- [x] Every claim links to a primary paper or official implementation and pinned revision.
- [x] Published and locally reproduced measurements are stored separately.
- [x] Every candidate has an explicit `test`, `retrain`, `monitor`, or `reject` verdict.
- [x] A ranked shortlist identifies the cheapest decisive local experiment for each task.
- [x] Final report explains both likely upgrades and evidence-based rejection reasons.

## Outputs

- `benchmark/landscape/registry.json`
- `benchmark/landscape/REPORT.md`
- Local raw outputs under ignored `.cache/` or `runs/`

## Validation

- Validate registry schema, source URLs, revisions, licenses, task comparability, and verdicts.
- Run adapter and metric smoke tests for every locally surviving candidate.
- Do not treat visual examples or paper AP alone as a promotion result.

## Next step

Execute `20260904-0701-isolated-pose-adapter-distillation.md`. DETRPose-X is the first experiment;
RTMO-L is the independent pose-family control after the adapter smoke test.

## Completion evidence

- `benchmark/landscape/registry.json`: 25 candidates across four categories with primary sources,
  pinned repository revisions where code exists, licenses, separate paper/local evidence, fit, and
  verdict.
- `benchmark/landscape/REPORT.md`: ranked tests plus explicit explanations of incompatible and
  rejected models.
- `benchmark/landscape/validate_registry.py`: schema, minimum coverage, source, evidence separation,
  uniqueness, and verdict validation.
