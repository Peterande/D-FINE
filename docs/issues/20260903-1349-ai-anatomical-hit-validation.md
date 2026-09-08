# Validate anatomical hits and game-server authority

- Status: Dropped
- Owner: AI-engine and game-server maintainers
- Depends on: `20260903-1345-ai-baseline-contract-drift.md`, `20260903-1348-ai-model-production-promotion.md`
- Target repository: `/home/berna/tagtwo-monorepo`

## Goal

Prove that model outputs produce correct, attributable anatomical hit decisions while game-server
remains authoritative for player, shot, damage, and game state.

## Resolution

The owner explicitly scoped current work away from Tagtwo production and game-server behavior.
Issue 1348 also produced no promoted model, so this issue's hard dependency is intentionally
unmet. Running or changing the authenticated leased-worker shot flow would be unrelated production
work rather than model-candidate research.

No anatomical-hit or game-server claims are made from the model-only benchmarks. The fixed dataset
records missing per-shot ground truth, and the candidate report does not use visual agreement as a
substitute. Open a new production validation issue if a future model passes promotion gates.

## Scope

- Direct body-part, pose fallback, segmentation fallback, occluded-person, object-hit, and no-hit cases.
- Crosshair, coordinate, class, confidence, frame, stream, shot, and player correlation.
- Wrong/stale result rejection and decision-source observability.
- AIW1 changes only if the existing generic response is insufficient.

## Acceptance criteria

- [ ] Deterministic fixtures cover every decision path and authoritative body class.
- [ ] Decision source and confidence are retained in evidence.
- [ ] Wrong frame, stream, shot, player, class, and stale responses fail safely.
- [ ] AI worker owns visual classification; game-server owns gameplay consequences.
- [ ] The exact authenticated leased-worker shot flow passes with the promoted model.

## Constraints

- Never make local tracker IDs authoritative player IDs.
- Do not move game rules into ai-engine.
- Shared contract or ownership changes require the monorepo ADR and contract workflows.

## Validation

- Run deterministic ai-engine hit tests, game-server consumer tests, and one retained live verdict trace.

## Next step

Dropped as out of current scope; no production candidate exists to validate.
