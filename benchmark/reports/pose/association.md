# RTMO association diagnosis

| David frame classification | Count | Share |
|---|---:|---:|
| Fully matched | 482 | 85.6% |
| Geometry rejected | 14 | 2.5% |
| RTMO pose missing | 27 | 4.8% |
| D-FINE detection missing | 17 | 3.0% |
| Both missing | 23 | 4.1% |

The observed average match rate of 0.87 is not primarily a matching-threshold problem. Perfectly
recovering all geometry rejections would still leave the long chair-occlusion interval dominated
by absent model outputs. Loosening center/IoU constraints is therefore rejected because it adds
multi-person swap risk for little possible gain.

Recommended next research, if required, is an occlusion-specific temporal test: compare existing
Kalman-predicted tracks with lower-confidence RTMO/D-FINE candidates and measure ID switches against
manual sequence annotations. This is separate from choosing RTMO as the pose model.

## Temporal threshold experiment

The manually reviewed chair interval is frames 340–470; the same person exists throughout, with
full occlusion around frames 400–439.

| Configuration | Fully matched | Geometry ambiguous | Mean match rate |
|---|---:|---:|---:|
| Current: det .50, pose .35/4 joints | 62 | 3 | 0.485 |
| Relax pose to .20/3 joints | 56 | 22 | 0.463 |
| Relax detection to .35 and pose | 63 | 50 | 0.561 |

Lower thresholds add too many ambiguous candidates and are rejected. Increasing tracker max age
from 30 to 60 frames reduced updated track IDs only from seven to six; it did not remove duplicate
tracks or the 19 empty-track frames. The safe conclusion is to retain current thresholds and avoid
claiming the 0.87 full-video match rate can be repaired by tuning alone.

A real next-generation fix needs occlusion-latched identity association with explicit duplicate
suppression and manually annotated track identity—not a looser IoU/center gate.

## Occlusion-latch follow-up

The research harness now exposes duplicate suppression for both D-FINE and RTMO candidates plus
configurable tracker age and uncertainty. A conservative candidate used NMS IoU 0.60, max age 60,
relative uncertainty 1.0 and absolute uncertainty 200 pixels.

| Control | Result |
|---|---|
| David frames 340–470 | Candidate track IDs fell from six to five; 19 frames still had no track |
| David full video | Association rose from 0.868 to 0.872; hit output remained exactly unchanged |
| SogO full video | Two track IDs, no empty-track frames and 1.000 association; hit output unchanged |

The candidate is **not promoted as the default configuration**. It suppresses some duplicates and
keeps predictions alive longer, but it cannot bridge the David interval where both models return no
observation. Wider uncertainty also increases identity-swap risk in multi-person scenes. Proving a
real identity-latch improvement requires manually annotated person IDs through occlusion; the office
videos currently provide no identity ground truth.

Visual controls:

- `ZKeepResults/research_20260904/david_rtmo_occlusion_latch.mp4`
- `ZKeepResults/research_20260904/sogo_rtmo_occlusion_latch.mp4`
