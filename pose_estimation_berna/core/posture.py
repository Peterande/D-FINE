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
        switch_frames: int = 3,
        signal_ema: float = 0.0,
    ):
        self.kpt_thr = float(kpt_thr)
        self.hold_frames = int(hold_frames)
        self.decay = float(decay)
        self.switch_frames = max(1, int(switch_frames))
        # EMA smoothing for posture input signals (0 disables smoothing)
        self.signal_ema = float(signal_ema)
        if not np.isfinite(self.signal_ema):
            self.signal_ema = 0.0
        self.signal_ema = float(np.clip(self.signal_ema, 0.0, 0.999))

        # per track state
        self._last_probs: Dict[int, np.ndarray] = {}
        self._missing: Dict[int, int] = {}
        self._state: Dict[int, str] = {}
        self._pending_state: Dict[int, str] = {}
        self._pending_count: Dict[int, int] = {}
        self._feat_ema: Dict[int, np.ndarray] = {}  # [vert_extent, hip_knee, hip_to_ank, knee_norm, aspect]

    def reset(self):
        self._last_probs = {}
        self._missing = {}
        self._state = {}
        self._pending_state = {}
        self._pending_count = {}
        self._feat_ema = {}

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

        # Knee angle feature (more robust for stand vs crouch than raw y-deltas).
        def _angle_deg(a, b, c):
            # angle at point b for triangle a-b-c
            ax, ay, as_ = a
            bx, by, bs_ = b
            cx, cy, cs_ = c
            if as_ < self.kpt_thr or bs_ < self.kpt_thr or cs_ < self.kpt_thr:
                return None
            v1 = np.array([ax - bx, ay - by], dtype=np.float32)
            v2 = np.array([cx - bx, cy - by], dtype=np.float32)
            n1 = float(np.linalg.norm(v1))
            n2 = float(np.linalg.norm(v2))
            if n1 < 1e-3 or n2 < 1e-3:
                return None
            cosang = float(np.dot(v1, v2) / (n1 * n2))
            cosang = float(np.clip(cosang, -1.0, 1.0))
            return float(np.degrees(np.arccos(cosang)))

        l_knee_ang = _angle_deg(lhip, lkne, lank)
        r_knee_ang = _angle_deg(rhip, rkne, rank)
        knee_angles = [a for a in [l_knee_ang, r_knee_ang] if a is not None]
        if knee_angles:
            knee_ang = float(sum(knee_angles) / len(knee_angles))  # deg
        else:
            knee_ang = 170.0  # default to "straight-ish" (avoid false crouch when noisy)
        knee_norm = float(np.clip(knee_ang / 180.0, 0.0, 1.0))

        # features (scale-invariant-ish)
        # - vertical extent proxy: (ankle_y - shoulder_y) / bbox_h
        # - crouch proxy: (knee_y - hip_y) / bbox_h (smaller when crouched) + knee angle
        # - lie proxy: bbox aspect ratio + low vertical extent
        vert_extent = (y_ank - y_sho) / bh if (s_ank >= self.kpt_thr and s_sho >= self.kpt_thr) else (bh / bh)
        hip_knee = (y_kne - y_hip) / bh if (s_kne >= self.kpt_thr and s_hip >= self.kpt_thr) else 0.2
        hip_to_ank = (y_ank - y_hip) / bh if (s_ank >= self.kpt_thr and s_hip >= self.kpt_thr) else 0.4
        aspect = bw / bh

        # Optional EMA smoothing of input signals to reduce frame-to-frame flips.
        if self.signal_ema > 0.0:
            cur = np.array([vert_extent, hip_knee, hip_to_ank, knee_norm, aspect], dtype=np.float32)
            prev = self._feat_ema.get(track_id, None)
            if prev is None or prev.shape != cur.shape:
                sm = cur
            else:
                a = float(self.signal_ema)
                sm = a * prev + (1.0 - a) * cur
            self._feat_ema[track_id] = sm.astype(np.float32)
            vert_extent, hip_knee, hip_to_ank, knee_norm, aspect = [float(x) for x in sm.tolist()]

        # logits (hand-tuned, now with knee angle to reduce stand/crouch flips)
        # stand: straight knees + large vertical extent + hips not too low.
        stand_logit = (
            3.0 * (vert_extent - 0.62)
            + 2.0 * (knee_norm - 0.86)
            + 1.0 * (hip_to_ank - 0.32)
            + 0.8 * (0.35 - aspect)
        )
        # crouch: bent knees + hips closer to knees/ankles, but still vertical-ish.
        crouch_logit = (
            3.0 * (0.24 - hip_knee)
            + 2.0 * (0.83 - knee_norm)
            + 1.0 * (0.34 - hip_to_ank)
            + 1.2 * (vert_extent - 0.45)
        )
        # lie: wide aspect and small vertical extent
        lie_logit = 3.0 * (aspect - 0.55) + 2.0 * (0.45 - vert_extent)

        probs_arr = _softmax(np.array([stand_logit, crouch_logit, lie_logit], dtype=np.float32))

        self._last_probs[track_id] = probs_arr
        self._missing[track_id] = 0

        probs = {"stand": float(probs_arr[0]), "crouch": float(probs_arr[1]), "lie": float(probs_arr[2])}
        raw_state = max(probs, key=probs.get)
        raw_conf = float(probs[raw_state])

        # Hysteresis: require N consecutive frames before switching state (reduces flip-flop).
        cur_state = self._state.get(track_id, raw_state)
        if raw_state == cur_state:
            self._pending_state.pop(track_id, None)
            self._pending_count.pop(track_id, None)
            final_state = cur_state
        else:
            pend = self._pending_state.get(track_id, None)
            if pend != raw_state:
                self._pending_state[track_id] = raw_state
                self._pending_count[track_id] = 1
            else:
                self._pending_count[track_id] = int(self._pending_count.get(track_id, 0)) + 1
            if int(self._pending_count.get(track_id, 0)) >= int(self.switch_frames):
                final_state = raw_state
                self._state[track_id] = final_state
                self._pending_state.pop(track_id, None)
                self._pending_count.pop(track_id, None)
            else:
                final_state = cur_state

        # ensure state exists
        self._state.setdefault(track_id, final_state)
        conf = float(probs.get(final_state, raw_conf))
        return PostureResult(probs=probs, state=final_state, conf=conf, valid=True)

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