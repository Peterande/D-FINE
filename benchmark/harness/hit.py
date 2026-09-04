"""Crosshair hit decision — mirror of the production HitDetector.

Mirrors tagtwo-monorepo/src/server/ai-engine/utils/hit_detector.py so the
benchmark scores candidates against the same rule production uses. Kept as a
local mirror rather than a cross-repo import: the monorepo is a separate
checkout with its own layout, and importing across would couple the benchmark
to a path we do not control.

Two facts carried over verbatim from production:
  - The crosshair is the image centre. It is not logged or configurable there
    (hit_detector.py:345-346), so a candidate must be scored at the same point.
  - The body part is the argmax over *non-background* class counts in a small
    region around that point, not the single pixel under it.
"""

from typing import Dict, Optional

import numpy as np

# hit_detector.py:7-14
BODY_PARTS = {
    0: "background",
    1: "head",
    2: "torso",
    3: "upper_arms",
    4: "lower_arms",
    5: "upper_legs",
    6: "lower_legs",
}

DEFAULT_REGION_SIZE = 3


def crosshair_xy(frame_width: int, frame_height: int) -> tuple:
    """Crosshair position. Production hardcodes the image centre."""
    return frame_width // 2, frame_height // 2


def decide_hit(seg_map: Optional[np.ndarray],
               frame_width: int,
               frame_height: int,
               region_size: int = DEFAULT_REGION_SIZE) -> Dict:
    """Return the production hit verdict for one frame.

    seg_map is the argmax class mask (H, W) uint8, already at frame resolution.
    """
    cx, cy = crosshair_xy(frame_width, frame_height)
    miss = {
        "hit": False,
        "hit_type": "no_hit",
        "body_part": None,
        "body_part_id": None,
        "confidence": 0.0,
        "source": None,
        "crosshair": {"x": int(cx), "y": int(cy)},
    }

    if seg_map is None or seg_map.size == 0:
        return miss

    seg_h, seg_w = seg_map.shape
    half = region_size // 2
    y0, y1 = max(0, cy - half), min(seg_h, cy + half + 1)
    x0, x1 = max(0, cx - half), min(seg_w, cx + half + 1)

    region = seg_map[y0:y1, x0:x1]
    if region.size == 0:
        return miss

    # Non-background classes only; background wins ties in neither branch.
    unique, counts = np.unique(region, return_counts=True)
    fg = unique > 0
    if not np.any(fg):
        return miss

    unique, counts = unique[fg], counts[fg]
    body_part_id = int(unique[np.argmax(counts)])

    return {
        "hit": True,
        "hit_type": "person_part",
        "body_part": BODY_PARTS.get(body_part_id, "unknown"),
        "body_part_id": body_part_id,
        # Share of the region occupied by the winning class. Production does not
        # emit this; the benchmark needs a per-frame confidence to rank candidates.
        "confidence": float(counts.max() / region.size),
        "source": "mask",
        "crosshair": {"x": int(cx), "y": int(cy)},
    }
