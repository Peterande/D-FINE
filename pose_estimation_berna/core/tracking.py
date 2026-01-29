from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


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
    age: int = 0
    time_since_update: int = 0

    # Smoothed state (optional)
    box_smooth: Optional[np.ndarray] = None
    kpt_smooth: Optional[np.ndarray] = None


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
    ):
        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self.smooth_alpha = float(smooth_alpha)
        self.smooth_boxes = bool(smooth_boxes)
        self.smooth_keypoints = bool(smooth_keypoints)
        self.min_area_ratio = float(min_area_ratio)
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
            t.score = float(det_scores[di])
            t.time_since_update = 0

            cur_kpt = None
            if det_keypoints is not None:
                cur_kpt = det_keypoints[di].astype(np.float32)
            t.keypoints = cur_kpt

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
            kpt = det_keypoints[di].astype(np.float32) if det_keypoints is not None else None
            t = Track(
                track_id=self._next_id,
                box_xyxy=box,
                score=score,
                keypoints=kpt,
            )
            if self.smooth_boxes:
                t.box_smooth = box.copy()
            if self.smooth_keypoints and kpt is not None:
                t.kpt_smooth = kpt.copy()
            self._tracks.append(t)
            self._next_id += 1

        # prune dead tracks
        self._tracks = [t for t in self._tracks if t.time_since_update <= self.max_age]
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
        self.Q = np.eye(8, dtype=np.float32) * q
        self.R = np.eye(4, dtype=np.float32) * r

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
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


@dataclass
class KalmanTrack(Track):
    kf: Optional[_KalmanFilter8D] = None


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
        iou_weight: float = 1.0,
        oks_weight: float = 0.0,
        kpt_thr: float = 0.2,
        kpt_age_decay: float = 0.97,
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
        self.iou_weight = float(iou_weight)
        self.oks_weight = float(oks_weight)
        self.kpt_thr = float(kpt_thr)
        self.kpt_age_decay = float(kpt_age_decay)

        self._next_id = 1
        self._tracks: List[KalmanTrack] = []

    @property
    def tracks(self) -> List[KalmanTrack]:
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
    ) -> List[KalmanTrack]:
        # 1) predict all tracks forward + age
        for t in self._tracks:
            t.age += 1
            t.time_since_update += 1
            if t.kf is not None:
                pred = t.kf.predict()
                t.box_xyxy = _cxcywh_to_xyxy(pred)
            # decay score when not updated (makes "ghost" tracks fade)
            if t.time_since_update > int(self.score_decay_grace):
                t.score = float(t.score * self.score_decay)

        n = int(det_boxes_xyxy.shape[0]) if det_boxes_xyxy is not None else 0
        if n == 0:
            self._tracks = [t for t in self._tracks if t.time_since_update <= self.max_age]
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
            if t.kf is None:
                t.kf = _KalmanFilter8D(
                    _xyxy_to_cxcywh(cur_box),
                    process_noise=self.process_noise,
                    measurement_noise=self.measurement_noise,
                )
            else:
                t.kf.update(_xyxy_to_cxcywh(cur_box))

            t.box_xyxy = cur_box
            t.score = float(det_scores[di])
            t.time_since_update = 0

            cur_kpt = None
            if det_keypoints is not None:
                cur_kpt = det_keypoints[di].astype(np.float32)
            t.keypoints = cur_kpt if cur_kpt is not None else t.keypoints

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
            kpt = det_keypoints[di].astype(np.float32) if det_keypoints is not None else None
            t = KalmanTrack(
                track_id=self._next_id,
                box_xyxy=box,
                score=score,
                keypoints=kpt,
                kf=_KalmanFilter8D(
                    _xyxy_to_cxcywh(box),
                    process_noise=self.process_noise,
                    measurement_noise=self.measurement_noise,
                ),
            )
            if self.smooth_boxes:
                t.box_smooth = box.copy()
            if self.smooth_keypoints and kpt is not None:
                t.kpt_smooth = kpt.copy()
            self._tracks.append(t)
            self._next_id += 1

        # 5) prune dead
        self._tracks = [t for t in self._tracks if t.time_since_update <= self.max_age]
        return self.tracks

