# Low-light segmentation fine-tuning

Only the existing ModelSurgery segmentation head was trained. The backbone, encoder, detection
decoder and pose decoder were frozen and verified hash-identical after training. Pascal Person
Parts train2017 received a balanced mix of normal images, 3x/10x darkness, low-light noise and
colour casts. Cross-entropy and Dice losses weighted hands and feet 2.5x.

## Full Pascal validation

| Candidate | Normal mIoU | 10x-dark mIoU | Dark hands | Dark feet |
|---|---:|---:|---:|---:|
| Current baseline | 73.61% | 49.59% | 24.64% | 23.63% |
| Seed 20260914, epoch 1 | 73.28% | 50.98% | 25.58% | 28.24% |
| Seed 20260915, epoch 1 | 73.06% | 50.17% | 22.92% | 27.63% |
| Two-seed weight soup | **73.36%** | **50.63%** | 24.60% | **27.26%** |

The two-seed soup is the stable research candidate: +1.04 percentage points in extreme-dark mIoU
for a 0.25-point normal-light cost, inside the predefined 0.5-point regression gate. Extreme-dark
feet improve by 3.64 points. Hand improvement is not seed-stable and remains unresolved.

- Seed 20260914 SHA-256: `e7f87c80ab1353fb267a149bed327cdd9f0c56a7c8bf1fee7c400d8607ea0d66`
- Seed 20260915 SHA-256: `36d138df5847a350f8562240351e4afa0c3a46ba4ba4ec9285f8189cc84ef21e`
- Two-seed soup SHA-256: `754c29904f0f898977fdad161c601cf2b66f44d0fe02e8bd69b26fad23a16e5a`

This is not promoted to production. The modest gain must be checked on real arena footage and
exported through the combined graph before promotion.
