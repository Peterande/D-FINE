from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from pose_estimation_berna.core.depth import DepthRegion, occluder_front_fraction


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
    x = x.astype(np.float32)
    x = x - float(np.max(x))
    e = np.exp(x)
    return (e / (float(np.sum(e)) + 1e-9)).astype(np.float32)


def _posture_probs_from_kpts(
    box_xyxy: np.ndarray,
    kpts: Optional[np.ndarray],
    kpt_thr: float = 0.2,
) -> Optional[np.ndarray]:
    """
    Heuristic stand/crouch/lie probabilities from COCO-17 keypoints.
    Returns probs [3] or None if too occluded/invalid.
    """
    if kpts is None or kpts.size == 0:
        return None
    if kpts.shape[0] < 17 or kpts.shape[1] < 3:
        return None

    def _kpt(idx: int):
        x, y, s = kpts[int(idx)]
        return float(x), float(y), float(s)

    nose = _kpt(COCO17["nose"])
    lsho = _kpt(COCO17["l_sho"])
    rsho = _kpt(COCO17["r_sho"])
    lhip = _kpt(COCO17["l_hip"])
    rhip = _kpt(COCO17["r_hip"])
    lkne = _kpt(COCO17["l_kne"])
    rkne = _kpt(COCO17["r_kne"])
    lank = _kpt(COCO17["l_ank"])
    rank = _kpt(COCO17["r_ank"])
    core = [nose, lsho, rsho, lhip, rhip, lkne, rkne, lank, rank]
    n_good = sum(1 for (_, _, s) in core if s >= float(kpt_thr))
    if n_good < 4:
        return None

    x1, y1, x2, y2 = [float(v) for v in box_xyxy.tolist()]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    def mid(a, b):
        ax, ay, as_ = a
        bx, by, bs_ = b
        if as_ >= float(kpt_thr) and bs_ >= float(kpt_thr):
            return (0.5 * (ax + bx), 0.5 * (ay + by), 0.5 * (as_ + bs_))
        if as_ >= float(kpt_thr):
            return (ax, ay, as_)
        if bs_ >= float(kpt_thr):
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

    vert_extent = (y_ank - y_sho) / bh if (s_ank >= kpt_thr and s_sho >= kpt_thr) else 1.0
    hip_knee = (y_kne - y_hip) / bh if (s_kne >= kpt_thr and s_hip >= kpt_thr) else 0.2
    aspect = bw / bh

    stand_logit = 3.0 * (vert_extent - 0.65) + 1.0 * (0.25 - aspect)
    crouch_logit = 3.0 * (0.20 - hip_knee) + 1.5 * (vert_extent - 0.45)
    lie_logit = 3.0 * (aspect - 0.55) + 2.0 * (0.45 - vert_extent)
    return _softmax(np.array([stand_logit, crouch_logit, lie_logit], dtype=np.float32))


def _posture_decay_update(
    probs: Optional[np.ndarray],
    missing: int,
    hold_frames: int,
    decay: float,
) -> Tuple[np.ndarray, int]:
    """
    Hold posture probs for a while when missing, then decay towards uniform.
    Returns (new_probs, new_missing).
    """
    miss = int(missing) + 1
    if probs is None or getattr(probs, "size", 0) != 3:
        probs_arr = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
        return probs_arr, miss

    probs_arr = probs.astype(np.float32)
    if miss <= int(hold_frames):
        return probs_arr, miss
    d = float(decay)
    uni = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
    probs_arr = d * probs_arr + (1.0 - d) * uni
    probs_arr = probs_arr / (float(np.sum(probs_arr)) + 1e-9)
    return probs_arr.astype(np.float32), miss


COCO_SIGMAS = np.array(
    [
        0.026, 0.025, 0.025, 0.035, 0.035,
        0.079, 0.079, 0.072, 0.072, 0.062,
        0.062, 0.107, 0.107, 0.087, 0.087,
        0.089, 0.089,
    ],
    dtype=np.float32,
)


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """IoU for single boxes in xyxy."""
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 1e-9 else 0.0


def _area_xyxy(b: np.ndarray) -> float:
    return float(max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1])))


def _oks_like(
    kpts_a: Optional[np.ndarray],
    kpts_b: Optional[np.ndarray],
    area: float,
    kpt_thr: float = 0.2,
    sigmas: np.ndarray = COCO_SIGMAS,
) -> float:
    """
    Lightweight OKS-like similarity in [0,1] between two keypoint sets in pixel coords.

    kpts_*: [K,3] where last dim is (x_px, y_px, score)
    area: object area in pixel^2 (used to normalize distances)
    """
    if kpts_a is None or kpts_b is None:
        return 0.0
    if kpts_a.size == 0 or kpts_b.size == 0:
        return 0.0
    if kpts_a.shape[-1] < 2 or kpts_b.shape[-1] < 2:
        return 0.0

    K = int(min(kpts_a.shape[0], kpts_b.shape[0], sigmas.shape[0]))
    if K <= 0:
        return 0.0

    xa = kpts_a[:K, 0].astype(np.float32)
    ya = kpts_a[:K, 1].astype(np.float32)
    xb = kpts_b[:K, 0].astype(np.float32)
    yb = kpts_b[:K, 1].astype(np.float32)

    if kpts_a.shape[1] >= 3 and kpts_b.shape[1] >= 3:
        sa = kpts_a[:K, 2].astype(np.float32)
        sb = kpts_b[:K, 2].astype(np.float32)
        vis = (sa >= float(kpt_thr)) & (sb >= float(kpt_thr))
    else:
        vis = np.ones((K,), dtype=bool)

    vis_count = int(np.sum(vis))
    if vis_count <= 0:
        return 0.0

    dx = (xa - xb).astype(np.float32)
    dy = (ya - yb).astype(np.float32)
    d2 = dx * dx + dy * dy  # [K]

    area = float(max(1.0, area))
    vars_ = (sigmas[:K] * 2.0) ** 2  # [K]
    denom = (2.0 * vars_ * area)
    denom = np.maximum(1e-6, denom).astype(np.float32)

    e = d2 / denom  # [K]
    oks_per_kpt = np.exp(-e)  # [K]
    oks = float(np.sum(oks_per_kpt[vis]) / float(vis_count))
    if not np.isfinite(oks):
        return 0.0
    return float(np.clip(oks, 0.0, 1.0))


@dataclass
class Track:
    track_id: int
    box_xyxy: np.ndarray  # [4]
    score: float
    keypoints: Optional[np.ndarray]  # [K,3] in pixels
    pose_quality: float = 0.0
    posture_probs: Optional[np.ndarray] = None  # [3] stand/crouch/lie
    posture_missing: int = 0
    age: int = 0
    time_since_update: int = 0

    # Smoothed state (optional)
    box_smooth: Optional[np.ndarray] = None
    kpt_smooth: Optional[np.ndarray] = None
    # Occlusion bookkeeping (Kalman tracker primarily, but safe to keep here)
    last_meas_cxcywh: Optional[np.ndarray] = None  # [4] last measurement (cx,cy,w,h) in pixels
    y_foot_ref: Optional[float] = None  # last measured footline (y2) in pixels; update only when visible
    oof_count: int = 0  # consecutive "out of frame" predictions
    last_visible_box_xyxy: Optional[np.ndarray] = None  # [4] xyxy from last visible frame
    freeze_active: bool = False
    freeze_box_xyxy: Optional[np.ndarray] = None  # [4] xyxy frozen at first occlusion frame
    # Anti-FP / stability: require a few matches before "confirming" a track.
    hits: int = 0
    confirmed: bool = False

    # Depth / occlusion reasoning (optional; inference-only)
    inv_depth: Optional[float] = None  # inverse depth proxy (larger => closer), per-frame estimate
    inv_depth_vel: float = 0.0
    inv_depth_conf: float = 0.0  # 0..1 confidence/consistency proxy
    occluded_conf: float = 0.0   # 0..1 "likely behind occluder" confidence
    occluder_front_frac: float = 0.0  # 0..1 fraction of nearer pixels in predicted region


class _KalmanFilter2D:
    """
    Minimal constant-velocity Kalman filter for 1D scalar (e.g. inverse depth):
      x = [z, vz]
      z_meas = [z]
    """

    def __init__(self, init_z: float, process_noise: float = 1e-3, measurement_noise: float = 2e-2):
        self.x = np.zeros((2, 1), dtype=np.float32)
        self.x[0, 0] = np.float32(init_z)
        self.x[1, 0] = np.float32(0.0)

        self.P = np.eye(2, dtype=np.float32) * 1.0
        self.F = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        self.H = np.array([[1.0, 0.0]], dtype=np.float32)
        self.Q0 = np.eye(2, dtype=np.float32) * float(process_noise)
        self.Q = self.Q0.copy()
        self.R0 = np.eye(1, dtype=np.float32) * float(measurement_noise)
        self.R = self.R0.copy()

    def predict(self, q_scale: float = 1.0) -> float:
        self.x = self.F @ self.x
        self.Q = self.Q0 * float(max(0.0, q_scale))
        self.P = (self.F @ self.P @ self.F.T) + self.Q
        return float(self.x[0, 0])

    def update(self, z_meas: float, r_scale: float = 1.0):
        z = np.array([[float(z_meas)]], dtype=np.float32)
        self.R = self.R0 * float(max(1e-6, r_scale))
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        I = np.eye(2, dtype=np.float32)
        self.P = (I - (K @ self.H)) @ self.P

    def get(self) -> float:
        return float(self.x[0, 0])

    def get_vel(self) -> float:
        return float(self.x[1, 0])


class IoUTracker:
    """
    Simple online tracker:
    - matches detections to existing tracks by IoU (greedy)
    - keeps tracks alive for max_age frames without update
    - optional EMA smoothing for boxes/keypoints (reduces jitter)

    This is intentionally lightweight (no Kalman) to keep dependencies minimal.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        smooth_alpha: float = 0.8,
        smooth_boxes: bool = True,
        smooth_keypoints: bool = True,
        min_area_ratio: float = 0.6,
        posture_kpt_thr: float = 0.2,
        posture_poseq_thr: float = 0.0,
        posture_hold_frames: int = 15,
        posture_decay: float = 0.9,
        score_ema: float = 0.8,
        new_track_score_thr: float = 0.25,
        confirm_hits: int = 3,
        unconfirmed_max_age: int = 2,
    ):
        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self.smooth_alpha = float(smooth_alpha)
        self.smooth_boxes = bool(smooth_boxes)
        self.smooth_keypoints = bool(smooth_keypoints)
        self.min_area_ratio = float(min_area_ratio)
        self.posture_kpt_thr = float(posture_kpt_thr)
        self.posture_poseq_thr = float(posture_poseq_thr)
        self.posture_hold_frames = int(posture_hold_frames)
        self.posture_decay = float(posture_decay)
        self.score_ema = float(np.clip(float(score_ema), 0.0, 0.999))
        self.new_track_score_thr = float(new_track_score_thr)
        self.confirm_hits = max(1, int(confirm_hits))
        self.unconfirmed_max_age = max(0, int(unconfirmed_max_age))
        self._next_id = 1
        self._tracks: List[Track] = []

    @property
    def tracks(self) -> List[Track]:
        return list(self._tracks)

    def reset(self):
        self._tracks = []
        self._next_id = 1

    def _ema(self, prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
        a = self.smooth_alpha
        return (a * prev + (1.0 - a) * cur).astype(np.float32)

    def update(
        self,
        det_boxes_xyxy: np.ndarray,  # [N,4]
        det_scores: np.ndarray,      # [N]
        det_keypoints: Optional[np.ndarray] = None,  # [N,K,3]
        det_pose_quality: Optional[np.ndarray] = None,  # [N]
    ) -> List[Track]:
        # age all tracks
        for t in self._tracks:
            t.age += 1
            t.time_since_update += 1

        n = int(det_boxes_xyxy.shape[0]) if det_boxes_xyxy is not None else 0
        if n == 0:
            # prune dead tracks
            self._tracks = [t for t in self._tracks if t.time_since_update <= self.max_age]
            return self.tracks

        # greedy match by IoU
        unmatched_dets = set(range(n))
        unmatched_tracks = set(range(len(self._tracks)))
        matches: List[Tuple[int, int]] = []  # (track_idx, det_idx)

        if self._tracks:
            # Build all candidate pairs and greedily pick highest IoU first (global 1-1 matching).
            pairs: List[Tuple[float, int, int]] = []
            for ti, t in enumerate(self._tracks):
                for di in range(n):
                    iou = _iou_xyxy(t.box_xyxy, det_boxes_xyxy[di])
                    if iou >= self.iou_threshold:
                        pairs.append((iou, ti, di))
            pairs.sort(key=lambda x: x[0], reverse=True)

            used_tracks, used_dets = set(), set()
            for iou, ti, di in pairs:
                if ti in used_tracks or di in used_dets:
                    continue
                used_tracks.add(ti)
                used_dets.add(di)
                matches.append((ti, di))

            for ti, di in matches:
                unmatched_tracks.discard(ti)
                unmatched_dets.discard(di)

        # update matched tracks
        for ti, di in matches:
            t = self._tracks[ti]
            cur_box = det_boxes_xyxy[di].astype(np.float32)

            # Occlusion robustness: don't let the box collapse too much vs previous.
            prev_box = t.box_smooth if t.box_smooth is not None else t.box_xyxy
            prev_area = _area_xyxy(prev_box)
            cur_area = _area_xyxy(cur_box)
            if prev_area > 1.0 and cur_area / prev_area < self.min_area_ratio:
                # Keep previous box as the measurement to avoid shrink under occlusion.
                cur_box = prev_box.astype(np.float32)

            t.box_xyxy = cur_box
            det_s = float(det_scores[di])
            a = float(self.score_ema)
            t.score = float(a * float(t.score) + (1.0 - a) * det_s)
            t.hits = int(t.hits) + 1
            if int(t.hits) >= int(self.confirm_hits):
                t.confirmed = True
            if det_pose_quality is not None and det_pose_quality.size > di:
                t.pose_quality = float(det_pose_quality[di])
            t.time_since_update = 0

            cur_kpt = None
            if det_keypoints is not None:
                cur_kpt = det_keypoints[di].astype(np.float32)
            t.keypoints = cur_kpt

            # posture update gated by pose_quality (optional)
            pq_ok = (self.posture_poseq_thr <= 0.0) or (float(t.pose_quality) >= float(self.posture_poseq_thr))
            probs = _posture_probs_from_kpts(cur_box, cur_kpt, kpt_thr=self.posture_kpt_thr) if pq_ok else None
            if probs is not None and probs.size == 3:
                t.posture_probs = probs
                t.posture_missing = 0
            else:
                t.posture_probs, t.posture_missing = _posture_decay_update(
                    t.posture_probs, t.posture_missing, self.posture_hold_frames, self.posture_decay
                )

            if self.smooth_boxes:
                if t.box_smooth is None:
                    t.box_smooth = cur_box.copy()
                else:
                    t.box_smooth = self._ema(t.box_smooth, cur_box)
            else:
                t.box_smooth = None

            if self.smooth_keypoints and cur_kpt is not None:
                if t.kpt_smooth is None:
                    t.kpt_smooth = cur_kpt.copy()
                else:
                    # smooth xy only; keep score channel as-is
                    prev = t.kpt_smooth
                    sm_xy = self._ema(prev[..., :2], cur_kpt[..., :2])
                    t.kpt_smooth = np.concatenate([sm_xy, cur_kpt[..., 2:3]], axis=-1)
            else:
                t.kpt_smooth = None

        # create new tracks for unmatched detections
        for di in sorted(unmatched_dets):
            box = det_boxes_xyxy[di].astype(np.float32)
            score = float(det_scores[di])
            if float(score) < float(self.new_track_score_thr):
                continue
            kpt = det_keypoints[di].astype(np.float32) if det_keypoints is not None else None
            pq = float(det_pose_quality[di]) if det_pose_quality is not None and det_pose_quality.size > di else 0.0
            t = Track(
                track_id=self._next_id,
                box_xyxy=box,
                score=score,
                keypoints=kpt,
                pose_quality=pq,
            )
            t.hits = 1
            t.confirmed = int(t.hits) >= int(self.confirm_hits)
            pq_ok = (self.posture_poseq_thr <= 0.0) or (float(t.pose_quality) >= float(self.posture_poseq_thr))
            t.posture_probs = _posture_probs_from_kpts(box, kpt, kpt_thr=self.posture_kpt_thr) if pq_ok else None
            t.posture_missing = 0 if (t.posture_probs is not None) else 1
            if self.smooth_boxes:
                t.box_smooth = box.copy()
            if self.smooth_keypoints and kpt is not None:
                t.kpt_smooth = kpt.copy()
            self._tracks.append(t)
            self._next_id += 1

        # occlusion handling for unmatched tracks: hold/decay posture probs
        for ti in sorted(unmatched_tracks):
            t = self._tracks[ti]
            t.posture_probs, t.posture_missing = _posture_decay_update(
                t.posture_probs, t.posture_missing, self.posture_hold_frames, self.posture_decay
            )

        # prune dead tracks
        kept = []
        for t in self._tracks:
            if int(t.time_since_update) > int(self.max_age):
                continue
            # Unconfirmed tracks are allowed only briefly if they miss updates (suppresses false positives).
            if (not bool(t.confirmed)) and int(t.time_since_update) > int(self.unconfirmed_max_age):
                continue
            kept.append(t)
        self._tracks = kept
        return self.tracks


def _xyxy_to_cxcywh(b: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = [float(x) for x in b.tolist()]
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    return np.array([cx, cy, w, h], dtype=np.float32)


def _cxcywh_to_xyxy(s: np.ndarray) -> np.ndarray:
    cx, cy, w, h = [float(x) for x in s.tolist()]
    w = max(1.0, w)
    h = max(1.0, h)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return np.array([x1, y1, x2, y2], dtype=np.float32)


class _KalmanFilter8D:
    """
    Minimal constant-velocity Kalman filter for bbox in (cx,cy,w,h) with velocities:
      x = [cx, cy, w, h, vx, vy, vw, vh]
      z = [cx, cy, w, h]
    """

    def __init__(
        self,
        init_cxcywh: np.ndarray,
        process_noise: float = 1e-2,
        measurement_noise: float = 1e-1,
    ):
        x0 = init_cxcywh.astype(np.float32).reshape(4)
        self.x = np.zeros((8, 1), dtype=np.float32)
        self.x[0:4, 0] = x0

        self.P = np.eye(8, dtype=np.float32) * 10.0
        self.P[4:, 4:] *= 100.0  # velocity uncertainty higher

        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = 1.0  # dt=1

        self.H = np.zeros((4, 8), dtype=np.float32)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        q = float(process_noise)
        r = float(measurement_noise)
        self.Q0 = np.eye(8, dtype=np.float32) * q
        self.Q = self.Q0.copy()
        self.R = np.eye(4, dtype=np.float32) * r

    def predict(self, q_scale: float = 1.0, p_inflate: float = 1.0) -> np.ndarray:
        self.x = self.F @ self.x
        qs = float(max(0.0, q_scale))
        self.Q = self.Q0 * qs
        pi = float(max(1.0, p_inflate))
        self.P = (pi * (self.F @ self.P @ self.F.T)) + self.Q
        return self.x[0:4, 0].copy()

    def update(self, z_cxcywh: np.ndarray):
        z = z_cxcywh.astype(np.float32).reshape(4, 1)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        I = np.eye(8, dtype=np.float32)
        self.P = (I - (K @ self.H)) @ self.P

    def get_state(self) -> np.ndarray:
        return self.x[0:4, 0].copy()

    def get_center_std(self) -> float:
        # sqrt(var_cx + var_cy)
        var = float(max(0.0, self.P[0, 0])) + float(max(0.0, self.P[1, 1]))
        return float(np.sqrt(max(0.0, var)))

    def clamp_size(self, w_min: float, w_max: float, h_min: float, h_max: float):
        self.x[2, 0] = np.float32(np.clip(float(self.x[2, 0]), float(w_min), float(w_max)))
        self.x[3, 0] = np.float32(np.clip(float(self.x[3, 0]), float(h_min), float(h_max)))


@dataclass
class KalmanTrack(Track):
    kf: Optional[_KalmanFilter8D] = None
    zf: Optional[_KalmanFilter2D] = None


class KalmanTracker:
    """
    Occlusion-robust tracker:
    - Kalman predict keeps a bbox even when detector returns nothing (short occlusions)
    - matching is still IoU-based, but on the *predicted* box
    - optional EMA smoothing is preserved for drawing stability
    """

    def __init__(
        self,
        iou_threshold: float = 0.2,
        oks_threshold: float = 1.0,
        max_age: int = 60,
        smooth_alpha: float = 0.8,
        smooth_boxes: bool = True,
        smooth_keypoints: bool = True,
        min_area_ratio: float = 0.6,
        process_noise: float = 1e-2,
        measurement_noise: float = 1e-1,
        score_decay: float = 0.98,
        score_decay_grace: int = 0,
        score_ema: float = 0.8,
        iou_weight: float = 1.0,
        oks_weight: float = 0.0,
        kpt_thr: float = 0.2,
        kpt_age_decay: float = 0.97,
        posture_kpt_thr: float = 0.2,
        posture_poseq_thr: float = 0.0,
        posture_hold_frames: int = 15,
        posture_decay: float = 0.9,
        # Step 2: occlusion-aware update (deterministic, inference-only)
        fps: float = 30.0,
        occlusion_ttl_sec: float = 1.0,
        q_inflate_alpha: float = 0.02,
        p_inflate_per_frame: float = 1.02,
        size_clamp_min_scale: float = 0.7,
        size_clamp_max_scale: float = 1.3,
        uncertainty_rel: float = 0.35,
        uncertainty_abs_px: float = 80.0,
        out_of_frame_max: int = 5,
        new_track_score_thr: float = 0.25,
        freeze_bbox: bool = True,
        confirm_hits: int = 3,
        unconfirmed_max_age: int = 2,
        # Depth-based occlusion reasoning (optional; inference-only)
        depth_enabled: bool = False,
        depth_region: DepthRegion = "torso",
        depth_occ_margin_abs: float = 0.02,
        depth_occ_margin_rel: float = 0.08,
        depth_occ_frac_thr: float = 0.35,
        depth_occ_conf_ema: float = 0.85,
        depth_ttl_mult: float = 3.0,
        depth_score_decay_occluded: float = 0.995,
        depth_process_noise: float = 1e-3,
        depth_measurement_noise: float = 2e-2,
    ):
        self.iou_threshold = float(iou_threshold)
        self.oks_threshold = float(oks_threshold)
        self.max_age = int(max_age)
        self.smooth_alpha = float(smooth_alpha)
        self.smooth_boxes = bool(smooth_boxes)
        self.smooth_keypoints = bool(smooth_keypoints)
        self.min_area_ratio = float(min_area_ratio)
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self.score_decay = float(score_decay)
        self.score_decay_grace = int(score_decay_grace)
        self.score_ema = float(np.clip(float(score_ema), 0.0, 0.999))
        self.iou_weight = float(iou_weight)
        self.oks_weight = float(oks_weight)
        self.kpt_thr = float(kpt_thr)
        self.kpt_age_decay = float(kpt_age_decay)
        self.posture_kpt_thr = float(posture_kpt_thr)
        self.posture_poseq_thr = float(posture_poseq_thr)
        self.posture_hold_frames = int(posture_hold_frames)
        self.posture_decay = float(posture_decay)

        self.fps = float(fps) if float(fps) > 1e-3 else 30.0
        self.occlusion_ttl_sec = float(max(0.0, occlusion_ttl_sec))
        self.q_inflate_alpha = float(max(0.0, q_inflate_alpha))
        self.p_inflate_per_frame = float(max(1.0, p_inflate_per_frame))
        self.size_clamp_min_scale = float(max(0.1, size_clamp_min_scale))
        self.size_clamp_max_scale = float(max(self.size_clamp_min_scale, size_clamp_max_scale))
        self.uncertainty_rel = float(max(0.0, uncertainty_rel))
        self.uncertainty_abs_px = float(max(0.0, uncertainty_abs_px))
        self.out_of_frame_max = int(max(0, out_of_frame_max))
        self.new_track_score_thr = float(new_track_score_thr)
        self.freeze_bbox = bool(freeze_bbox)
        self.confirm_hits = max(1, int(confirm_hits))
        self.unconfirmed_max_age = max(0, int(unconfirmed_max_age))

        self.depth_enabled = bool(depth_enabled)
        self.depth_region = depth_region
        self.depth_occ_margin_abs = float(max(0.0, depth_occ_margin_abs))
        self.depth_occ_margin_rel = float(max(0.0, depth_occ_margin_rel))
        self.depth_occ_frac_thr = float(np.clip(float(depth_occ_frac_thr), 0.0, 1.0))
        self.depth_occ_conf_ema = float(np.clip(float(depth_occ_conf_ema), 0.0, 0.999))
        self.depth_ttl_mult = float(max(1.0, depth_ttl_mult))
        self.depth_score_decay_occluded = float(np.clip(float(depth_score_decay_occluded), 0.90, 0.9999))
        self.depth_process_noise = float(max(1e-8, depth_process_noise))
        self.depth_measurement_noise = float(max(1e-8, depth_measurement_noise))

        self._next_id = 1
        self._tracks: List[KalmanTrack] = []

    @property
    def tracks(self) -> List[KalmanTrack]:
        return list(self._tracks)

    def reset(self):
        self._tracks = []
        self._next_id = 1

    def set_fps(self, fps: float):
        if fps is None:
            return
        f = float(fps)
        if f > 1e-3 and np.isfinite(f):
            self.fps = f

    def _posture_label(self, t: KalmanTrack) -> str:
        # posture_probs order: [stand, crouch, lie(prone)]
        if t.posture_probs is None or getattr(t.posture_probs, "size", 0) != 3:
            return "stand"
        idx = int(np.argmax(t.posture_probs))
        return "stand" if idx == 0 else ("crouch" if idx == 1 else "prone")

    def _clamp_center_to_image(self, cx: float, cy: float, w: float, h: float, img_wh: Tuple[int, int]) -> Tuple[float, float]:
        iw, ih = int(img_wh[0]), int(img_wh[1])
        # clamp so bbox stays inside frame (reduces jitter at edges)
        cx = float(np.clip(cx, 0.5 * w, max(0.5 * w, float(iw) - 0.5 * w)))
        cy = float(np.clip(cy, 0.5 * h, max(0.5 * h, float(ih) - 0.5 * h)))
        return cx, cy

    def _ema(self, prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
        a = self.smooth_alpha
        return (a * prev + (1.0 - a) * cur).astype(np.float32)

    def update(
        self,
        det_boxes_xyxy: np.ndarray,  # [N,4]
        det_scores: np.ndarray,      # [N]
        det_keypoints: Optional[np.ndarray] = None,  # [N,K,3]
        det_pose_quality: Optional[np.ndarray] = None,  # [N]
        img_wh: Optional[Tuple[int, int]] = None,  # (w,h) for deterministic out-of-frame termination
        det_inv_depth: Optional[np.ndarray] = None,  # [N] inverse depth proxy per detection (aligned with det_boxes)
        inv_depth_map: Optional[np.ndarray] = None,  # [H,W] inverse depth proxy for current frame
    ) -> List[KalmanTrack]:
        # 1) predict all tracks forward + age (occlusion-aware)
        ttl_frames = int(min(float(self.max_age), round(self.occlusion_ttl_sec * float(self.fps))))
        ttl_frames = max(0, ttl_frames)
        for t in self._tracks:
            t.age += 1
            t.time_since_update += 1
            if t.kf is not None:
                # Inflate uncertainty during occlusion (time_since_update > 0)
                miss = int(max(0, t.time_since_update))
                q_scale = 1.0 + (self.q_inflate_alpha * float(miss * miss))
                p_inflate = float(self.p_inflate_per_frame) if miss > 0 else 1.0
                pred = t.kf.predict(q_scale=q_scale, p_inflate=p_inflate)

                # Clamp predicted size to last measured size to avoid explosion/collapse during short hallucination.
                if t.last_meas_cxcywh is not None:
                    mw = float(max(1.0, float(t.last_meas_cxcywh[2])))
                    mh = float(max(1.0, float(t.last_meas_cxcywh[3])))
                    t.kf.clamp_size(
                        w_min=self.size_clamp_min_scale * mw,
                        w_max=self.size_clamp_max_scale * mw,
                        h_min=self.size_clamp_min_scale * mh,
                        h_max=self.size_clamp_max_scale * mh,
                    )
                    pred = t.kf.get_state()

                # Step 3/4: Freeze-BBox policy (critical for demo):
                # - On first missing-detection frame: freeze width and y1/y2 from last visible frame.
                # - During occlusion: allow ONLY horizontal motion (cx) from Kalman prediction.
                if bool(self.freeze_bbox) and miss > 0 and t.last_visible_box_xyxy is not None:
                    # Activate freeze on first occluded frame only.
                    if miss == 1 and not bool(t.freeze_active):
                        t.freeze_active = True
                        t.freeze_box_xyxy = t.last_visible_box_xyxy.astype(np.float32).copy()

                    if bool(t.freeze_active) and t.freeze_box_xyxy is not None:
                        fx1, fy1, fx2, fy2 = [float(v) for v in t.freeze_box_xyxy.tolist()]
                        w = float(max(1.0, fx2 - fx1))
                        h = float(max(1.0, fy2 - fy1))
                        cx = float(pred[0])  # allow only horizontal motion
                        cy = 0.5 * (fy1 + fy2)  # freeze vertical extent

                        if img_wh is not None:
                            cx, _ = self._clamp_center_to_image(cx, cy, w, h, img_wh)
                            cy = float(np.clip(cy, 0.5 * h, max(0.5 * h, float(img_wh[1]) - 0.5 * h)))

                        # Write back into KF state for consistency (keep velocities as predicted)
                        t.kf.x[0, 0] = np.float32(cx)
                        t.kf.x[1, 0] = np.float32(cy)
                        t.kf.x[2, 0] = np.float32(w)
                        t.kf.x[3, 0] = np.float32(h)
                        pred = t.kf.get_state()

                t.box_xyxy = _cxcywh_to_xyxy(pred)

                # Out-of-frame termination bookkeeping (deterministic).
                if img_wh is not None:
                    w_img, h_img = int(img_wh[0]), int(img_wh[1])
                    x1, y1, x2, y2 = [float(v) for v in t.box_xyxy.tolist()]
                    oof = (x2 < 0.0) or (y2 < 0.0) or (x1 > float(w_img)) or (y1 > float(h_img))
                    t.oof_count = int(t.oof_count + 1) if oof else 0

            # predict depth if enabled
            if self.depth_enabled and t.zf is not None:
                miss = int(max(0, t.time_since_update))
                q_scale_z = 1.0 + 0.02 * float(miss * miss)
                try:
                    t.inv_depth = float(t.zf.predict(q_scale=q_scale_z))
                    t.inv_depth_vel = float(t.zf.get_vel())
                except Exception:
                    pass

            # update occluder evidence while occluded (per-track)
            if self.depth_enabled and inv_depth_map is not None and int(t.time_since_update) > 0 and t.inv_depth is not None:
                frac = occluder_front_fraction(
                    inv_depth_map,
                    t.box_xyxy,
                    inv_depth_expected=float(t.inv_depth),
                    region=self.depth_region,
                    margin_abs=self.depth_occ_margin_abs,
                    margin_rel=self.depth_occ_margin_rel,
                )
                t.occluder_front_frac = float(frac)
                occ_like = 1.0 if float(frac) >= float(self.depth_occ_frac_thr) else 0.0
                a = float(self.depth_occ_conf_ema)
                t.occluded_conf = float(a * float(t.occluded_conf) + (1.0 - a) * float(occ_like))

            # decay score when not updated (makes "ghost" tracks fade)
            if t.time_since_update > int(self.score_decay_grace):
                decay = float(self.score_decay)
                if self.depth_enabled and float(getattr(t, "occluded_conf", 0.0)) > 0.5:
                    decay = float(self.depth_score_decay_occluded)
                t.score = float(t.score * decay)

        n = int(det_boxes_xyxy.shape[0]) if det_boxes_xyxy is not None else 0
        if n == 0:
            # no detections => occlusion: decay posture towards unknown
            for t in self._tracks:
                t.posture_probs, t.posture_missing = _posture_decay_update(
                    t.posture_probs, t.posture_missing, self.posture_hold_frames, self.posture_decay
                )
            # Step 2 termination rules while occluded:
            # - hard TTL (<= 1s by default)
            # - uncertainty too large
            # - predicted bbox out-of-frame for too long
            kept = []
            for t in self._tracks:
                # Depth-aware TTL extension: keep alive longer when we have strong occluder evidence.
                ttl_eff = int(ttl_frames)
                if self.depth_enabled and ttl_frames > 0:
                    ttl_eff = int(round(float(ttl_frames) * (1.0 + float(t.occluded_conf) * (float(self.depth_ttl_mult) - 1.0))))
                if ttl_eff > 0 and int(t.time_since_update) > int(ttl_eff):
                    continue
                if t.kf is not None and t.last_meas_cxcywh is not None:
                    min_wh = float(max(1.0, min(float(t.last_meas_cxcywh[2]), float(t.last_meas_cxcywh[3]))))
                    thr = max(self.uncertainty_abs_px, self.uncertainty_rel * min_wh)
                    if float(t.kf.get_center_std()) > float(thr):
                        continue
                if self.out_of_frame_max > 0 and int(t.oof_count) > int(self.out_of_frame_max):
                    continue
                if int(t.time_since_update) <= int(self.max_age):
                    kept.append(t)
            self._tracks = kept
            return self.tracks

        unmatched_dets = set(range(n))
        unmatched_tracks = set(range(len(self._tracks)))
        matches: List[Tuple[int, int]] = []

        # 2) greedy global 1-1 matching by combined score:
        #    IoU(pred_box, det_box) + OKS-like(track_kpts, det_kpts)
        if self._tracks:
            pairs: List[Tuple[float, int, int]] = []
            for ti, t in enumerate(self._tracks):
                for di in range(n):
                    iou = _iou_xyxy(t.box_xyxy, det_boxes_xyxy[di])
                    oks = 0.0
                    use_oks = (self.oks_weight > 0.0) and (det_keypoints is not None) and (t.keypoints is not None)
                    if use_oks:
                        # Use a conservative area proxy to normalize distances.
                        area = min(_area_xyxy(t.box_xyxy), _area_xyxy(det_boxes_xyxy[di]))
                        oks = _oks_like(
                            t.keypoints,
                            det_keypoints[di],
                            area=area,
                            kpt_thr=self.kpt_thr,
                        )
                        # If the track is "lost" for a while, its stored keypoints get stale:
                        # decay OKS contribution with time_since_update to avoid wrong long-range re-association.
                        oks = float(oks * (self.kpt_age_decay ** int(max(0, t.time_since_update))))

                    # Default behavior (stable baseline): IoU-only matching.
                    # If OKS association is enabled (oks_weight>0), allow OKS to also create a candidate match.
                    if (iou >= self.iou_threshold) or (use_oks and (oks >= self.oks_threshold)):
                        comb = (self.iou_weight * float(iou)) + (self.oks_weight * float(oks))
                        pairs.append((comb, ti, di))

            pairs.sort(key=lambda x: x[0], reverse=True)

            used_tracks, used_dets = set(), set()
            for comb, ti, di in pairs:
                if ti in used_tracks or di in used_dets:
                    continue
                used_tracks.add(ti)
                used_dets.add(di)
                matches.append((ti, di))

            for ti, di in matches:
                unmatched_tracks.discard(ti)
                unmatched_dets.discard(di)

        # 3) update matched tracks with measurements
        for ti, di in matches:
            t = self._tracks[ti]
            cur_box = det_boxes_xyxy[di].astype(np.float32)

            prev_box = t.box_smooth if t.box_smooth is not None else t.box_xyxy
            prev_area = _area_xyxy(prev_box)
            cur_area = _area_xyxy(cur_box)
            if prev_area > 1.0 and cur_area / prev_area < self.min_area_ratio:
                cur_box = prev_box.astype(np.float32)

            # Kalman measurement update (cxcywh)
            meas = _xyxy_to_cxcywh(cur_box)
            if t.kf is None:
                t.kf = _KalmanFilter8D(
                    meas,
                    process_noise=self.process_noise,
                    measurement_noise=self.measurement_noise,
                )
            else:
                t.kf.update(meas)

            t.box_xyxy = cur_box
            det_s = float(det_scores[di])
            a = float(self.score_ema)
            t.score = float(a * float(t.score) + (1.0 - a) * det_s)
            t.time_since_update = 0
            t.hits = int(t.hits) + 1
            if int(t.hits) >= int(self.confirm_hits):
                t.confirmed = True
            if det_pose_quality is not None and det_pose_quality.size > di:
                t.pose_quality = float(det_pose_quality[di])
            t.last_meas_cxcywh = meas.astype(np.float32)
            t.y_foot_ref = float(cur_box[3])  # y2 in pixels (update only when visible)
            t.last_visible_box_xyxy = cur_box.astype(np.float32).copy()
            t.freeze_active = False
            t.freeze_box_xyxy = None
            t.oof_count = 0
            t.occluded_conf = 0.0
            t.occluder_front_frac = 0.0

            cur_kpt = None
            if det_keypoints is not None:
                cur_kpt = det_keypoints[di].astype(np.float32)
            t.keypoints = cur_kpt if cur_kpt is not None else t.keypoints

            # depth measurement update (inverse depth proxy)
            if self.depth_enabled and det_inv_depth is not None and int(det_inv_depth.size) > di:
                z_meas = float(det_inv_depth[di])
                if np.isfinite(z_meas):
                    if t.zf is None:
                        t.zf = _KalmanFilter2D(
                            init_z=z_meas,
                            process_noise=self.depth_process_noise,
                            measurement_noise=self.depth_measurement_noise,
                        )
                    else:
                        # regular measurement update
                        t.zf.update(z_meas, r_scale=1.0)
                    t.inv_depth = float(t.zf.get())
                    t.inv_depth_vel = float(t.zf.get_vel())
                    t.inv_depth_conf = float(np.clip(1.0 / (abs(float(t.inv_depth_vel)) + 1.0), 0.0, 1.0))

            pq_ok = (self.posture_poseq_thr <= 0.0) or (float(t.pose_quality) >= float(self.posture_poseq_thr))
            probs = _posture_probs_from_kpts(cur_box, cur_kpt, kpt_thr=self.posture_kpt_thr) if pq_ok else None
            if probs is not None and probs.size == 3:
                t.posture_probs = probs
                t.posture_missing = 0
            else:
                t.posture_probs, t.posture_missing = _posture_decay_update(
                    t.posture_probs, t.posture_missing, self.posture_hold_frames, self.posture_decay
                )

            if self.smooth_boxes:
                if t.box_smooth is None:
                    t.box_smooth = cur_box.copy()
                else:
                    t.box_smooth = self._ema(t.box_smooth, cur_box)
            else:
                t.box_smooth = None

            if self.smooth_keypoints and t.keypoints is not None:
                if t.kpt_smooth is None:
                    t.kpt_smooth = t.keypoints.copy()
                else:
                    prev = t.kpt_smooth
                    cur = t.keypoints
                    sm_xy = self._ema(prev[..., :2], cur[..., :2])
                    t.kpt_smooth = np.concatenate([sm_xy, cur[..., 2:3]], axis=-1)
            else:
                t.kpt_smooth = None

        # 4) spawn new tracks for unmatched detections
        for di in sorted(unmatched_dets):
            box = det_boxes_xyxy[di].astype(np.float32)
            score = float(det_scores[di])
            if float(score) < float(self.new_track_score_thr):
                continue
            kpt = det_keypoints[di].astype(np.float32) if det_keypoints is not None else None
            pq = float(det_pose_quality[di]) if det_pose_quality is not None and det_pose_quality.size > di else 0.0
            meas = _xyxy_to_cxcywh(box)
            t = KalmanTrack(
                track_id=self._next_id,
                box_xyxy=box,
                score=score,
                keypoints=kpt,
                pose_quality=pq,
                kf=_KalmanFilter8D(
                    meas,
                    process_noise=self.process_noise,
                    measurement_noise=self.measurement_noise,
                ),
            )
            t.hits = 1
            t.confirmed = int(t.hits) >= int(self.confirm_hits)
            t.last_meas_cxcywh = meas.astype(np.float32)
            t.y_foot_ref = float(box[3])  # y2 in pixels (update only when visible)
            t.last_visible_box_xyxy = box.astype(np.float32).copy()
            t.freeze_active = False
            t.freeze_box_xyxy = None
            t.oof_count = 0
            t.occluded_conf = 0.0
            t.occluder_front_frac = 0.0
            pq_ok = (self.posture_poseq_thr <= 0.0) or (float(t.pose_quality) >= float(self.posture_poseq_thr))
            t.posture_probs = _posture_probs_from_kpts(box, kpt, kpt_thr=self.posture_kpt_thr) if pq_ok else None
            t.posture_missing = 0 if (t.posture_probs is not None) else 1
            if self.smooth_boxes:
                t.box_smooth = box.copy()
            if self.smooth_keypoints and kpt is not None:
                t.kpt_smooth = kpt.copy()

            if self.depth_enabled and det_inv_depth is not None and int(det_inv_depth.size) > di:
                z_meas = float(det_inv_depth[di])
                if np.isfinite(z_meas):
                    t.zf = _KalmanFilter2D(
                        init_z=z_meas,
                        process_noise=self.depth_process_noise,
                        measurement_noise=self.depth_measurement_noise,
                    )
                    t.inv_depth = float(t.zf.get())
                    t.inv_depth_vel = float(t.zf.get_vel())
                    t.inv_depth_conf = 1.0
            self._tracks.append(t)
            self._next_id += 1

        # occlusion handling for unmatched tracks (not updated this frame)
        for ti in sorted(unmatched_tracks):
            t = self._tracks[ti]
            t.posture_probs, t.posture_missing = _posture_decay_update(
                t.posture_probs, t.posture_missing, self.posture_hold_frames, self.posture_decay
            )

            # Update occluder evidence while unmatched (detector missed / partial occlusion)
            if self.depth_enabled and inv_depth_map is not None and t.inv_depth is not None:
                frac = occluder_front_fraction(
                    inv_depth_map,
                    t.box_xyxy,
                    inv_depth_expected=float(t.inv_depth),
                    region=self.depth_region,
                    margin_abs=self.depth_occ_margin_abs,
                    margin_rel=self.depth_occ_margin_rel,
                )
                t.occluder_front_frac = float(frac)
                occ_like = 1.0 if float(frac) >= float(self.depth_occ_frac_thr) else 0.0
                a = float(self.depth_occ_conf_ema)
                t.occluded_conf = float(a * float(t.occluded_conf) + (1.0 - a) * float(occ_like))

        # 5) prune dead
        kept = []
        for t in self._tracks:
            if int(t.time_since_update) > int(self.max_age):
                continue
            if (not bool(t.confirmed)) and int(t.time_since_update) > int(self.unconfirmed_max_age):
                continue
            kept.append(t)
        self._tracks = kept
        return self.tracks

