from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np


DepthModelName = Literal["midas_small", "dpt_hybrid", "dpt_large"]
DepthRegion = Literal["inner", "torso"]


def _clip_xyxy(box_xyxy: np.ndarray, w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy.tolist()]
    # allow partially out-of-frame, then clip
    x1i = int(np.floor(np.clip(x1, 0.0, float(max(0, w - 1)))))
    y1i = int(np.floor(np.clip(y1, 0.0, float(max(0, h - 1)))))
    x2i = int(np.ceil(np.clip(x2, 0.0, float(w))))
    y2i = int(np.ceil(np.clip(y2, 0.0, float(h))))
    if x2i <= x1i:
        x2i = min(w, x1i + 1)
    if y2i <= y1i:
        y2i = min(h, y1i + 1)
    return x1i, y1i, x2i, y2i


def _shrink_region(x1: int, y1: int, x2: int, y2: int, region: DepthRegion) -> Tuple[int, int, int, int]:
    bw = max(1, int(x2 - x1))
    bh = max(1, int(y2 - y1))
    # Conservative crop to reduce edge bleed from occluders/background.
    if region == "inner":
        fx1, fx2 = 0.20, 0.80
        fy1, fy2 = 0.20, 0.80
    else:  # "torso"
        fx1, fx2 = 0.20, 0.80
        fy1, fy2 = 0.25, 0.80
    rx1 = x1 + int(round(fx1 * bw))
    rx2 = x1 + int(round(fx2 * bw))
    ry1 = y1 + int(round(fy1 * bh))
    ry2 = y1 + int(round(fy2 * bh))
    # keep at least 2x2
    if rx2 <= rx1 + 1:
        rx1 = x1
        rx2 = x2
    if ry2 <= ry1 + 1:
        ry1 = y1
        ry2 = y2
    return rx1, ry1, rx2, ry2


@dataclass
class DepthStats:
    inv_depth_med: float
    inv_depth_mad: float
    inv_depth_center: float
    valid: bool


def bbox_inv_depth_stats(
    inv_depth: np.ndarray,
    box_xyxy: np.ndarray,
    *,
    region: DepthRegion = "torso",
) -> DepthStats:
    """
    Robust inverse-depth stats inside bbox.

    inv_depth: [H,W] float32, larger => closer (typical MiDaS convention).
    Returns median + MAD (robust spread) + center sample.
    """
    if inv_depth is None or getattr(inv_depth, "size", 0) == 0:
        return DepthStats(inv_depth_med=float("nan"), inv_depth_mad=float("nan"), inv_depth_center=float("nan"), valid=False)
    if inv_depth.ndim != 2:
        return DepthStats(inv_depth_med=float("nan"), inv_depth_mad=float("nan"), inv_depth_center=float("nan"), valid=False)

    h, w = int(inv_depth.shape[0]), int(inv_depth.shape[1])
    x1, y1, x2, y2 = _clip_xyxy(box_xyxy, w=w, h=h)
    rx1, ry1, rx2, ry2 = _shrink_region(x1, y1, x2, y2, region=region)
    patch = inv_depth[ry1:ry2, rx1:rx2]
    if patch.size < 4:
        return DepthStats(inv_depth_med=float("nan"), inv_depth_mad=float("nan"), inv_depth_center=float("nan"), valid=False)

    # robust center sample (clipped)
    cx = int(np.clip(round(0.5 * (x1 + x2)), 0, w - 1))
    cy = int(np.clip(round(0.5 * (y1 + y2)), 0, h - 1))
    zc = float(inv_depth[cy, cx])

    # median + MAD
    vals = patch.reshape(-1).astype(np.float32)
    vals = vals[np.isfinite(vals)]
    if vals.size < 8:
        return DepthStats(inv_depth_med=float("nan"), inv_depth_mad=float("nan"), inv_depth_center=zc, valid=False)
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) + 1e-6
    return DepthStats(inv_depth_med=med, inv_depth_mad=mad, inv_depth_center=zc, valid=True)


def occluder_front_fraction(
    inv_depth: np.ndarray,
    box_xyxy: np.ndarray,
    *,
    inv_depth_expected: float,
    region: DepthRegion = "torso",
    margin_abs: float = 0.02,
    margin_rel: float = 0.08,
) -> float:
    """
    Fraction of pixels inside bbox that are "in front of" expected depth.

    Using inverse depth, "in front" means inv_depth is larger than expected by a margin.
    """
    if inv_depth is None or getattr(inv_depth, "size", 0) == 0 or not np.isfinite(float(inv_depth_expected)):
        return 0.0
    if inv_depth.ndim != 2:
        return 0.0

    h, w = int(inv_depth.shape[0]), int(inv_depth.shape[1])
    x1, y1, x2, y2 = _clip_xyxy(box_xyxy, w=w, h=h)
    rx1, ry1, rx2, ry2 = _shrink_region(x1, y1, x2, y2, region=region)
    patch = inv_depth[ry1:ry2, rx1:rx2]
    if patch.size < 16:
        return 0.0
    vals = patch.reshape(-1).astype(np.float32)
    vals = vals[np.isfinite(vals)]
    if vals.size < 16:
        return 0.0

    zexp = float(inv_depth_expected)
    thr = zexp + float(margin_abs) + float(margin_rel) * abs(zexp)
    frac = float(np.mean(vals > thr))
    if not np.isfinite(frac):
        return 0.0
    return float(np.clip(frac, 0.0, 1.0))


class DepthEstimator:
    """
    Minimal MiDaS/DPT wrapper (torch.hub) for inverse depth.

    - Inference-only; no retraining
    - Runs at configurable inference resolution for speed
    - Returns inverse depth in original frame size (float32)
    """

    def __init__(
        self,
        *,
        model_name: DepthModelName = "midas_small",
        device: str = "cuda",
        input_short_side: int = 320,
        hub_repo: str = "intel-isl/MiDaS",
    ):
        self.model_name = str(model_name)
        self.device_str = str(device)
        self.input_short_side = int(max(128, input_short_side))
        self.hub_repo = str(hub_repo)

        self._model = None
        self._transform = None
        self._torch = None

    def _lazy_init(self):
        if self._model is not None:
            return

        # MiDaS hub currently depends on timm (even for MiDaS_small via some backbones).
        # Make the failure mode obvious.
        try:
            import timm  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "Monocular depth requires the 'timm' package. Install it with:\n"
                "  pip install timm\n"
                "or in this repo's venv:\n"
                "  /home/berna/D-FINE-NEWBRINGER/venv/bin/python3 -m pip install timm\n"
                f"Original import error: {e}"
            ) from e

        import torch  # local import to keep tracker lightweight when depth is disabled

        self._torch = torch
        device = torch.device(self.device_str if (self.device_str == "cpu" or torch.cuda.is_available()) else "cpu")

        # NOTE: torch.hub will download weights on first run.
        if self.model_name == "midas_small":
            hub_model = "MiDaS_small"
        elif self.model_name == "dpt_hybrid":
            hub_model = "DPT_Hybrid"
        elif self.model_name == "dpt_large":
            hub_model = "DPT_Large"
        else:
            raise ValueError(f"Unknown depth model_name: {self.model_name}")

        model = torch.hub.load(self.hub_repo, hub_model)
        model = model.to(device).eval()

        transforms = torch.hub.load(self.hub_repo, "transforms")
        if hub_model == "MiDaS_small":
            tfm = transforms.small_transform
        else:
            tfm = transforms.dpt_transform

        self._model = model
        self._transform = tfm
        self._device = device

    def predict_inv_depth(self, frame_rgb_u8: np.ndarray) -> np.ndarray:
        """
        frame_rgb_u8: [H,W,3] uint8 RGB
        returns inv_depth: [H,W] float32 (larger => closer)
        """
        self._lazy_init()
        assert self._model is not None
        assert self._transform is not None
        assert self._torch is not None

        import torch.nn.functional as F  # local import for same reason as torch

        if frame_rgb_u8 is None or frame_rgb_u8.ndim != 3 or frame_rgb_u8.shape[2] != 3:
            raise ValueError("frame_rgb_u8 must be HxWx3 RGB uint8")
        H, W = int(frame_rgb_u8.shape[0]), int(frame_rgb_u8.shape[1])

        torch = self._torch
        with torch.no_grad():
            # The hub transform expects RGB uint8 numpy.
            inp = self._transform(frame_rgb_u8)
            # MiDaS hub transforms vary by version:
            # - sometimes return Tensor [3,h,w]
            # - sometimes return Tensor [1,3,h,w]
            # - sometimes return dict with key 'image'
            if isinstance(inp, dict):
                inp = inp.get("image", None)
            if inp is None:
                raise RuntimeError("MiDaS transform returned None (unexpected).")
            if not hasattr(inp, "ndim"):
                raise RuntimeError(f"MiDaS transform returned unexpected type: {type(inp)}")
            if int(inp.ndim) == 3:
                inp = inp.unsqueeze(0)
            elif int(inp.ndim) == 4:
                pass
            else:
                raise RuntimeError(f"MiDaS transform returned tensor with unsupported ndim={int(inp.ndim)}")

            inp = inp.to(self._device)  # [1,3,h,w]
            pred = self._model(inp)  # [1,h',w'] or [1,1,h',w']
            if pred.ndim == 4:
                pred = pred[:, 0, :, :]

            # Upsample to original resolution
            pred_up = F.interpolate(pred.unsqueeze(1), size=(H, W), mode="bicubic", align_corners=False)[:, 0]
            inv_depth = pred_up.squeeze(0).float().detach().cpu().numpy().astype(np.float32)

        # Normalize per-frame to stabilize thresholds (keeps ordering, not metric).
        # This makes margin_abs usable across different scenes.
        lo = float(np.percentile(inv_depth, 2.0))
        hi = float(np.percentile(inv_depth, 98.0))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo + 1e-6:
            inv_depth = (inv_depth - lo) / (hi - lo)
        return inv_depth.astype(np.float32)

