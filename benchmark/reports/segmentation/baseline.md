# Reproduced segmentation baseline

The unchanged ModelSurgery backbone and segmentation head were evaluated on all 1,829 validation
images in the deterministic Pascal Person Parts reconstruction, using the original training
preprocessing (RGB, direct 640x640 resize, `/255`, ImageNet mean/std).

| Metric | Result |
|---|---:|
| Global-confusion mIoU | **73.6079%** |
| Historical checkpoint mIoU | 73.64% |
| Difference | -0.0321 percentage points |
| Pixel accuracy | 95.4991% |

| Class | IoU |
|---|---:|
| background | 96.4858% |
| head | 87.6564% |
| torso | 73.9206% |
| arms | 70.6251% |
| hands | 64.1787% |
| legs | 66.3308% |
| feet | 56.0579% |

The 0.032-point difference is small enough to reproduce the historical result, but the rebuilt
dataset is still documented as a reconstruction rather than claimed byte-identical to the lost
preparation.

Evidence hashes:

- checkpoint: `a0d9b62b804ece6cd240740036467113a633dd9d2159aa7a7b4fc53044f4c0a7`
- dataset provenance: `3cefa42701c31153d863735f597528794af5d6a3be535383312fefdd0f2b4df2`
- raw result: `db4e6607143446013a19e87b4fee532bc1b2283d5ce2edb943f24bf79b6af7bd`

Conclusion: segmentation is not the primary quality gap. Future segmentation comparisons should
target hands and feet specifically and must beat 73.6079% under this exact protocol.
