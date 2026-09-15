# Test ModelSurgery segmentation under varied lighting

- Status: Done (synthetic robustness control)
- Repository scope: research checkout only

## Goal

Measure how stable the current seven-class ModelSurgery segmentation is under arena-relevant
lighting changes without retraining or modifying the model.

## Result

The 80-frame TagTwo pilot was evaluated under eight deterministic conditions. Low light with sensor
noise was worst at 78.1% foreground consistency IoU, with hands at 69.7%. Overexposure was second
worst at 81.1%, with feet at 75.1%. Strong shadows, LED colour cast, backlight and plain darkness
retained more than 92% mean foreground consistency.

These are controlled invariance measurements, not ground-truth accuracy. The actionable product
finding is to prioritize camera exposure and low-noise capture, then validate on real arena footage.
No checkpoint or production code was changed.

## Three-times-darker follow-up

An exact one-third-brightness condition retained 90.3% foreground consistency (hands 84.6%, feet
86.8%). Darkness alone is therefore not the main synthetic failure mode; low-light sensor noise
remains substantially worse.

At exactly one tenth brightness, foreground consistency falls to 75.2% (hands 69.0%, feet 72.5%,
torso 70.9%). This is the first pure-darkness condition to underperform the earlier noisy-low-light
control and establishes a clear robustness limit.
