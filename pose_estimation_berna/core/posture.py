from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


COCO17 = {
    "nose": 0,
    "l_eye": 1,
    "r_eye": 2,
    "l_ear": 3,
    "r_ear": 4,
    "l_sho": 5,
    "r_sho": 6,
    "l_elb": 7,
    "r_elb": 8,
    "l_wri": 9,
    "r_wri": 10,
    "l_hip": 11,
    "r_hip": 12,
    "l_kne": 13,
    "r_kne": 14,
    "l_ank": 15,
    "r_ank": 16,
}


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-9)


def _kpt_xy(kpts: np.ndarray, idx: int) -> Tuple[float, float, float]:
    """Return (x,y,score) for keypoint idx."""
    x, y, s = kpts[idx]
    return float(x), float(y), float(s)


@dataclass
class PostureResult:
    probs: Dict[str, float]  # stand/crouch/lie
    state: str
    conf: float
    valid: bool


class PostureEstimator:
    """
    Heuristic posture estimator for COCO-17 keypoints.

    Outputs probabilities over:
      - stand
      - crouch
      - lie

    Key design: occlusion-robust via hold/decay:
    - if too few keypoints are confident, keep previous probabilities and decay them slowly.
    """

    def __init__(
        self,
        kpt_thr: float = 0.2,
        hold_frames: int = 15,
        decay: float = 0.9,
    ):
        self.kpt_thr = float(kpt_thr)
        self.hold_frames = int(hold_frames)
        self.decay = float(decay)

        # per track state
        self._last_probs: Dict[int, np.ndarray] = {}
        self._missing: Dict[int, int] = {}

    def reset(self):
        self._last_probs = {}
        self._missing = {}

    def estimate(self, track_id: int, box_xyxy: np.ndarray, kpts: Optional[np.ndarray]) -> PostureResult:
        if kpts is None or kpts.size == 0:
            return self._fallback(track_id, valid=False)

        # collect core joints
        nose = _kpt_xy(kpts, COCO17["nose"])
        lsho = _kpt_xy(kpts, COCO17["l_sho"])
        rsho = _kpt_xy(kpts, COCO17["r_sho"])
        lhip = _kpt_xy(kpts, COCO17["l_hip"])
        rhip = _kpt_xy(kpts, COCO17["r_hip"])
        lkne = _kpt_xy(kpts, COCO17["l_kne"])
        rkne = _kpt_xy(kpts, COCO17["r_kne"])
        lank = _kpt_xy(kpts, COCO17["l_ank"])
        rank = _kpt_xy(kpts, COCO17["r_ank"])

        core = [nose, lsho, rsho, lhip, rhip, lkne, rkne, lank, rank]
        n_good = sum(1 for (_, _, s) in core if s >= self.kpt_thr)

        # if too occluded, hold previous
        if n_good < 4:
            return self._fallback(track_id, valid=False)

        # bbox geometry
        x1, y1, x2, y2 = [float(v) for v in box_xyxy.tolist()]
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)

        # compute midpoints when both sides available
        def mid(a, b):
            ax, ay, as_ = a
            bx, by, bs_ = b
            if as_ >= self.kpt_thr and bs_ >= self.kpt_thr:
                return (0.5 * (ax + bx), 0.5 * (ay + by), 0.5 * (as_ + bs_))
            # fall back to whichever is visible
            if as_ >= self.kpt_thr:
                return (ax, ay, as_)
            if bs_ >= self.kpt_thr:
                return (bx, by, bs_)
            return (float("nan"), float("nan"), 0.0)

        sho = mid(lsho, rsho)
        hip = mid(lhip, rhip)
        kne = mid(lkne, rkne)
        ank = mid(lank, rank)

        _, y_sho, s_sho = sho
        _, y_hip, s_hip = hip
        _, y_kne, s_kne = kne
        _, y_ank, s_ank = ank

        # features (scale-invariant-ish)
        # - vertical extent proxy: (ankle_y - shoulder_y) / bbox_h
        # - crouch proxy: (knee_y - hip_y) / bbox_h (smaller when crouched)
        # - lie proxy: bbox aspect ratio + low vertical extent
        vert_extent = (y_ank - y_sho) / bh if (s_ank >= self.kpt_thr and s_sho >= self.kpt_thr) else (bh / bh)
        hip_knee = (y_kne - y_hip) / bh if (s_kne >= self.kpt_thr and s_hip >= self.kpt_thr) else 0.2
        aspect = bw / bh

        # logits (hand-tuned, but stable enough for debugging)
        # stand: large vertical extent, moderate aspect, hips above knees.
        stand_logit = 3.0 * (vert_extent - 0.65) + 1.0 * (0.25 - aspect)
        # crouch: smaller hip_knee distance and still vertical-ish
        crouch_logit = 3.0 * (0.20 - hip_knee) + 1.5 * (vert_extent - 0.45)
        # lie: wide aspect and small vertical extent
        lie_logit = 3.0 * (aspect - 0.55) + 2.0 * (0.45 - vert_extent)

        probs_arr = _softmax(np.array([stand_logit, crouch_logit, lie_logit], dtype=np.float32))

        self._last_probs[track_id] = probs_arr
        self._missing[track_id] = 0

        probs = {"stand": float(probs_arr[0]), "crouch": float(probs_arr[1]), "lie": float(probs_arr[2])}
        state = max(probs, key=probs.get)
        conf = float(probs[state])
        return PostureResult(probs=probs, state=state, conf=conf, valid=True)

    def _fallback(self, track_id: int, valid: bool) -> PostureResult:
        prev = self._last_probs.get(track_id, None)
        miss = int(self._missing.get(track_id, 0)) + 1
        self._missing[track_id] = miss

        if prev is None:
            probs_arr = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
        else:
            # hold for a while, then decay towards uniform
            if miss <= self.hold_frames:
                probs_arr = prev
            else:
                probs_arr = self.decay * prev + (1.0 - self.decay) * np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
                probs_arr = probs_arr / (np.sum(probs_arr) + 1e-9)
            self._last_probs[track_id] = probs_arr

        probs = {"stand": float(probs_arr[0]), "crouch": float(probs_arr[1]), "lie": float(probs_arr[2])}
        state = max(probs, key=probs.get)
        conf = float(probs[state])
        return PostureResult(probs=probs, state=state, conf=conf, valid=valid)