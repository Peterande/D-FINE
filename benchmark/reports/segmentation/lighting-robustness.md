# ModelSurgery segmentation lighting robustness

The unchanged current ModelSurgery segmentation head was evaluated on 80 office-video frames under
eight deterministic lighting transformations (640 transformed inputs). Scores are mask consistency
against the same model on the normal image, not ground-truth accuracy.

| Lighting condition | Foreground consistency IoU | Hands | Feet |
|---|---:|---:|---:|
| Strong shadow | 94.3% | 92.6% | 91.5% |
| LED colour cast | 93.2% | 89.7% | 88.8% |
| Backlight | 92.2% | 90.0% | 84.1% |
| Dark | 92.1% | 87.9% | 87.5% |
| Cold | 90.1% | 84.1% | 87.7% |
| Warm | 89.4% | 84.6% | 86.5% |
| Three times darker | 90.3% | 84.6% | 86.8% |
| Ten times darker | 75.2% | 69.0% | 72.5% |
| Overexposed | 81.1% | 77.2% | 75.1% |
| Low light + noise | 78.1% | 69.7% | 74.8% |

## Finding

The model is comparatively stable under plain darkness, shadows, backlight and LED colour casts.
The largest sensitivity is low-light sensor noise, followed by clipping from overexposure. Hands
are the weakest small-part class under noisy low light (69.7% consistency); feet are weakest under
overexposure (75.1%). At exactly one third of normal brightness the model retains 90.3% foreground
consistency, showing that sensor noise is more damaging than brightness reduction alone. Arena
controls should therefore prioritize exposure control and low-noise capture before retraining.

At one tenth of normal brightness, consistency falls sharply to 75.2% overall. Hands drop to 69.0%,
feet to 72.5%, and torso to 70.9%. This establishes a clear synthetic low-light failure region even
without adding sensor noise.

The local visual controls and detailed CSV files are in
`ZKeepResults/segmentation_lighting_robustness_v1`. Real arena footage and manually reviewed masks
are still required to measure actual accuracy.
