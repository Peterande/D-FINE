# Multitask encoder repair report

## Inputs

- Reconstructed phase 1: SHA-256
  `bc6098892408d58a87c6275c5c250e5469832bdd62ad05eaa59ea5492a0862b5`
- Deployed phase 2: SHA-256
  `a0d9b62b804ece6cd240740036467113a633dd9d2159aa7a7b4fc53044f4c0a7`
- Detection/segmentation source: SHA-256
  `a02d039731f56b307d01fadc9ed82077a409ce3df003946ce748809ff0342701`
- Pose source: SHA-256
  `e4dbb64142811ac1775f4695ee53c947a53a9935942b8d07e437d5f0148f1bc6`

The original phase-1 artifact is unavailable. `phase1_reconstructed.pth` was recreated with the
checked-in surgery tool at 100% shape/key coverage and is explicitly not called the original.

## Reproduced change

| Module | State tensors | Changed phase 1 → phase 2 | Relative L2 change |
|---|---:|---:|---:|
| backbone | 650 | 0 | 0 |
| encoder | 546 | 546 | 1.36008 |
| detection decoder | 245 | 0 | 0 |
| pose decoder | 261 | 234 | 0.00228 |
| segmentation head | 66 | 0 | 0 |

Full COCO evaluation shows the intended trade:

| Model | Detection AP | Pose OKS AP |
|---|---:|---:|
| reconstructed phase 1 | 45.9 | 0.0 |
| deployed phase 2 | 43.6 | 48.2 |

Pose phase 2 recovered pose from effectively unusable to 48.2 AP, while detection lost 2.3 AP.

## Tested repair strategies

### Isolation control

`alpha=0` restores the phase-1 encoder while keeping phase-2 task heads. On the fixed 200-image
screen it scored 51.9 detection AP and 19.2 pose AP. Isolation protects detection but destroys too
much pose compatibility, so it is rejected.

### Encoder-retention balancing

`build_candidates.py` performs deterministic weight-space retention:

`encoder = (1-alpha) * phase1_encoder + alpha * phase2_encoder`

| Alpha | Detection AP, 200-image screen | Pose AP, 200-image screen |
|---:|---:|---:|
| 0.00 | 51.9 | 19.2 |
| 0.10 | 52.2 | 23.9 |
| 0.25 | 51.6 | 30.5 |
| 0.50 | 51.9 | 39.4 |
| 0.75 | 52.1 | 45.2 |
| 1.00 | 49.0 | 47.1 |

The only plausible compromise, alpha 0.75, was promoted to full evaluation:

| Metric | Deployed phase 2 | Alpha 0.75 | Verdict |
|---|---:|---:|---|
| COCO detection AP | 43.6 | 45.7 | +2.1 |
| COCO pose OKS AP | 48.2 | 44.9 | -3.3 |
| Soldier-domain recall | 86.3% | 84.9% | -1.4 points |
| Soldier-domain FP/image | 0.41 | 0.33 | improved |
| Direct segmentation hits, 200 David frames | 186 | 186 | identical |
| Model latency | 30.70 ms | 30.75 ms | equivalent |
| Total profiled latency | 39.57 ms | 39.64 ms | equivalent |
| Peak CUDA allocated | 532.69 MiB | 530.72 MiB | equivalent |

Segmentation logits and direct mask-hit decisions are exactly unchanged by construction because
that branch reads the unchanged backbone and unchanged segmentation head, not the encoder.
Pose-fallback hit ground truth remains unavailable and is not invented.

## Decision

**No change.** Alpha 0.75 is rejected because it hides a 3.3-point pose regression and a domain
recall regression behind a COCO detection improvement. Neither tested strategy passes a joint
non-regression gate. A future learned adapter/distillation run remains research, but there is no
evidence to replace the deployed phase-2 weights now.

All generated checkpoints and raw predictions remain under ignored `.cache/encoder_repair/`.
