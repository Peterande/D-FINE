"""
Hit registration logic for occluded players (geometry + tracker uncertainty only).

This module intentionally does NOT depend on any CV models or training-time changes.
It consumes tracker state (predicted bbox + time_since_update + uncertainty) and
applies deterministic, explainable gameplay rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import math


@dataclass(frozen=True)
class HitDecision:
    """
    Result of hit registration.
    - kind:
        - "person_direct": player was visible; normal hitbox logic should be used (not handled here)
        - "person_transferred": hit on occluder counts as hit on person (occluded)
        - "occluder_only": hit is absorbed by occluder
    """

    kind: str
    p: float  # transfer probability/plausibility in [0,1] (0 for non-transfers)
    damage_multiplier: float  # multiply base damage by this (0 for occluder_only)


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(min(max(float(x), float(lo)), float(hi)))


def _point_in_bbox_xyxy(pt_xy: Tuple[float, float], box_xyxy) -> bool:
    x, y = float(pt_xy[0]), float(pt_xy[1])
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    return (x >= x1) and (x <= x2) and (y >= y1) and (y <= y2)


def _expand_bbox_xyxy(box_xyxy, dx: float, dy: float):
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    return (x1 - float(dx), y1 - float(dy), x2 + float(dx), y2 + float(dy))


def _bbox_center_xy(box_xyxy) -> Tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def register_occluder_hit_2d(
    *,
    hit_xy: Tuple[float, float],
    occluder_id: Any,
    track: Any,
    ttl_frames: int,
    p_min: float = 0.4,
    k_sigma: float = 2.0,
    transfer_damage_min: float = 0.6,
    transfer_damage_max: float = 1.0,
    require_occluder_match: bool = True,
    track_occluder_attr: str = "current_occluder_id",
) -> HitDecision:
    """
    2D occluder-assisted hit registration.

    Inputs:
    - hit_xy: screen-space hit point on the occluder (pixels).
    - occluder_id: identifier of the occluder object hit by the bullet (game collision system).
    - track: tracker track object. Expected fields:
        - time_since_update: int
        - box_xyxy: iterable (x1,y1,x2,y2) for predicted/last bbox in pixels
        - kf: optional; if present and has get_center_std() -> float, used as sigma
    - ttl_frames: max frames to allow transfer while occluded (<= ~fps for 1s).

    Deterministic logic:
    - Only transfers when track is occluded (time_since_update > 0) and within TTL.
    - Only transfers when hit is inside an expanded region around predicted bbox (k*sigma).
    - Transfer plausibility p decays with distance from predicted center and occlusion age.
    - Damage is scaled by p (bounded) for fairness.
    """
    # Visible players should use your normal direct hitbox logic.
    tsu = int(getattr(track, "time_since_update", 0))
    if tsu <= 0:
        return HitDecision(kind="person_direct", p=0.0, damage_multiplier=1.0)

    ttl = max(1, int(ttl_frames))
    if tsu > ttl:
        return HitDecision(kind="occluder_only", p=0.0, damage_multiplier=0.0)

    if require_occluder_match:
        track_occ = getattr(track, track_occluder_attr, None)
        if track_occ is not None and occluder_id != track_occ:
            return HitDecision(kind="occluder_only", p=0.0, damage_multiplier=0.0)

    box = getattr(track, "box_xyxy", None)
    if box is None:
        return HitDecision(kind="occluder_only", p=0.0, damage_multiplier=0.0)

    # sigma from Kalman if available, else fall back to a fraction of bbox size.
    sigma = None
    kf = getattr(track, "kf", None)
    if kf is not None and hasattr(kf, "get_center_std"):
        try:
            sigma = float(kf.get_center_std())
        except Exception:
            sigma = None
    if sigma is None or not math.isfinite(sigma) or sigma <= 1e-3:
        x1, y1, x2, y2 = [float(v) for v in box]
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        sigma = 0.15 * min(bw, bh)

    sigma = max(1.0, float(sigma))
    k = float(max(0.0, k_sigma))
    region = _expand_bbox_xyxy(box, dx=k * sigma, dy=k * sigma)
    if not _point_in_bbox_xyxy(hit_xy, region):
        return HitDecision(kind="occluder_only", p=0.0, damage_multiplier=0.0)

    cx, cy = _bbox_center_xy(box)
    dx = float(hit_xy[0]) - float(cx)
    dy = float(hit_xy[1]) - float(cy)
    d2 = dx * dx + dy * dy

    # Gaussian plausibility around predicted center + linear time decay to 0 at TTL.
    w_sigma = math.exp(-d2 / (2.0 * sigma * sigma))
    w_t = max(0.0, 1.0 - (float(tsu) / float(ttl)))
    p = _clamp(w_sigma * w_t, 0.0, 1.0)

    if p < float(p_min):
        return HitDecision(kind="occluder_only", p=p, damage_multiplier=0.0)

    dmg_lo = float(transfer_damage_min)
    dmg_hi = float(transfer_damage_max)
    if dmg_hi < dmg_lo:
        dmg_hi, dmg_lo = dmg_lo, dmg_hi
    dmg = _clamp(dmg_lo + (dmg_hi - dmg_lo) * p, 0.0, 1.0)
    return HitDecision(kind="person_transferred", p=p, damage_multiplier=dmg)


