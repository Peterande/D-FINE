# Map the current real-time vision model landscape

- Status: Open
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

- [ ] Candidate registry covers at least five credible models per task where five exist.
- [ ] Every claim links to a primary paper or official implementation and pinned revision.
- [ ] Published and locally reproduced measurements are stored separately.
- [ ] Every candidate has an explicit `test`, `retrain`, `monitor`, or `reject` verdict.
- [ ] A ranked shortlist identifies the cheapest decisive local experiment for each task.
- [ ] Final report explains both likely upgrades and evidence-based rejection reasons.

## Outputs

- `benchmark/landscape/registry.json`
- `benchmark/landscape/REPORT.md`
- Local raw outputs under ignored `.cache/` or `runs/`

## Validation

- Validate registry schema, source URLs, revisions, licenses, task comparability, and verdicts.
- Run adapter and metric smoke tests for every locally surviving candidate.
- Do not treat visual examples or paper AP alone as a promotion result.

## Next step

Expand the existing seven-candidate registry into task-specific longlists, beginning with human
pose because it has the largest measured quality gap.
