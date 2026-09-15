# Distill RTMO poses into the combined model

- Status: Done (proof-of-concept gate)
- Repository scope: research checkout only

## Goal

Test whether RTMO can improve the existing pose path without adding a second production model or
changing the shared backbone, encoder, detection decoder, or segmentation head.

## Result

RTMO-L produced 686 pseudo-labelled poses on 200 COCO train2017 images. Training only zero-initialized
pose feature adapters for 100 steps improved held-out COCO val2017 pose AP:

| Evaluation | Current model | RTMO-trained adapter | Change |
|---|---:|---:|---:|
| Fixed 200-image screen | 47.1 | 48.1 | +1.0 |
| Full 2,693-image validation | 48.2 | 49.2 | +1.0 |

The checkpoint records identical before/after hashes for backbone, encoder, detection decoder and
segmentation head. It is therefore still one combined model graph; RTMO is only an offline teacher.

## Decision

The approach passes the research gate but is not production-ready. The run is deliberately small;
the next experiment should use a larger pinned train subset and compare multiple seeds before any
export or production work.

## Full-scale follow-up

The full-data seed-123 run completed for three epochs (54,876 usable images, 257,911 teacher poses,
41,157 steps) and scored **51.1 full-validation AP**, improving the current 48.2 baseline by
**2.9 AP**. Protected module hashes remained unchanged. The improvement is confirmed for one seed;
multi-seed variance remains the final research gate.

Seeds 456 and 789 subsequently scored 51.7 and 51.5 AP. Across seeds 123/456/789 the mean is
**51.43 AP** (sample standard deviation 0.31), and every run beats the 48.2 baseline. The training
improvement is therefore reproducible. Export parity, latency and office-video review remain
separate promotion gates.
