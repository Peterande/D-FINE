# Isolated pose-adapter experiment

## Architecture and safety

Three zero-initialized 1x1 residual adapters were inserted between the frozen shared encoder and
the pose decoder. They add 443,520 parameters. Before training, every detection, box, pose and
segmentation output was bit-identical to the baseline.

Backbone, encoder, detection decoder and segmentation head were frozen and hashed before/after
every run. All hashes remained identical. No production checkpoint was modified.

## Runs

| Run | Trainable modules | Steps | Distillation | 200-image pose AP |
|---|---|---:|---:|---:|
| Baseline | none | 0 | no | 47.1 |
| Adapter control | adapters | 1 | no | smoke only |
| Adapter distilled | adapters | 500 | yes | 43.4 |
| Adapter + decoder distilled | adapters + pose decoder | 500 | yes | 41.6 |

Artifacts outside Git:

- adapter distilled: SHA-256 `00caf182c3fac18f0948dc3a2b0e93584e996c0507baa8dda443cdb24ca6bd80`
- adapter + decoder: SHA-256 `94baa6f4c9eea924aa884fee737ad4ed6add303c008dc157e9563c850ba39026`
- resumed two-step proof: SHA-256 `e12e6ab9e80dbb24c8eceaa4f0e86da2bde22d635bbcb971b341c7c51c138957`

The training checkpoints record seed, optimizer, learning rate, source arguments, step, losses,
optimizer state, adapter/decoder state, and protected-module hashes. A one-step distilled run was
successfully resumed for its second step.

## Verdict

**No change.** Both trained candidates failed the fixed 200-image promotion screen, so running the
more expensive full all-task suite would not change their rejection. Protected detection and
segmentation outputs remain unchanged by construction, but pose itself regressed.

The likely weakness is direct query-index distillation: teacher and student DETR queries are sets,
not guaranteed semantic pairs. A future attempt should use Hungarian/OKS matching or distill
encoder feature distributions before query decoding. That is a materially different experiment,
not grounds to hide the failed result here.
