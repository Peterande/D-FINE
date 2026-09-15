"""Body-part segmentation: production head vs Sapiens, on identical frames.

The production mask has never been measured — mIoU 48 comes from a filename and
no annotated body-part data exists to reproduce it. This is the cheap substitute:
run a far stronger model over the same frames and see how much the two disagree.

Sapiens-0.3B (Meta) is the reference. It is not a deployment candidate — a 0.3B
ViT at 1024x768 is nowhere near real-time — but it is the only model found that
does *semantic* body-part segmentation rather than instance segmentation, which
makes it the right yardstick. Its 28 Goliath classes collapse onto the
production 7.

Two questions get answered:
  agreement   how much of the person mask the two label the same way
  decision    would the hit verdict at the crosshair actually change

The second is what matters. A mask can differ a lot in the pixels nobody aims at.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
for p in [REPO, REPO / "src", REPO / "tools", REPO / "segmentation_sivert", REPO / "pose_estimation_berna"]:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from benchmark.harness.infer_crosshair import build_model_from_merged, PART_COLORS  # noqa: E402
from benchmark.harness.hit import BODY_PARTS, decide_hit  # noqa: E402

# Goliath 28 -> production 7. Clothing follows the body part it covers, which is
# what the production head was trained to do: it labels a clothed torso "torso".
GOLIATH_TO_PROD = {
    0: 0,                                    # Background
    1: 2,                                    # Apparel -> torso
    2: 1, 3: 1, 23: 1, 24: 1, 25: 1, 26: 1, 27: 1,   # Face/Hair/lips/teeth/tongue -> head
    21: 2, 22: 2,                            # Torso, Upper_Clothing -> torso
    10: 3, 19: 3,                            # Upper arms
    6: 4, 15: 4, 5: 4, 14: 4,                # Lower arms + hands
    11: 5, 20: 5, 12: 5,                     # Upper legs + Lower_Clothing
    7: 6, 16: 6, 4: 6, 13: 6, 8: 6, 17: 6, 9: 6, 18: 6,  # Lower legs, feet, shoes, socks
}

SAPIENS_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
SAPIENS_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def map_goliath(mask28: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask28, dtype=np.uint8)
    for src, dst in GOLIATH_TO_PROD.items():
        out[mask28 == src] = dst
    return out


def run_sapiens(model, frame_bgr, device, size_hw=(1024, 768)):
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size_hw[1], size_hw[0]), interpolation=cv2.INTER_LINEAR)
    x = (resized.astype(np.float32) - SAPIENS_MEAN) / SAPIENS_STD
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)

    t0 = time.perf_counter()
    with torch.inference_mode():
        logits = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    logits = torch.nn.functional.interpolate(logits, size=(h, w), mode="bilinear",
                                             align_corners=False)
    mask28 = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    return map_goliath(mask28), dt


def overlay(frame_bgr, mask7, alpha=0.45):
    out = frame_bgr.copy()
    tint = np.zeros_like(frame_bgr)
    for cid, colour in PART_COLORS.items():
        tint[mask7 == cid] = colour
    m = mask7 > 0
    out[m] = (out[m] * (1 - alpha) + tint[m] * alpha).astype(np.uint8)
    return out


def label_panel(img, text, colour=(255, 255, 255)):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det-config", required=True)
    ap.add_argument("--pose-config", required=True)
    ap.add_argument("--merged-ckpt", required=True)
    ap.add_argument("--sapiens-ckpt", required=True)
    ap.add_argument("--input", required=True, help="Video to sample frames from.")
    ap.add_argument("--num-frames", type=int, default=30)
    ap.add_argument("--image-size", type=int, default=640)
    ap.add_argument("--seg-num-classes", type=int, default=7)
    ap.add_argument("--seg-feature-dim", type=int, default=384)
    ap.add_argument("--seg-dropout", type=float, default=0.1)
    ap.add_argument("--crosshair-region", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="runs/sapiens_compare")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)

    prod, _ = build_model_from_merged(
        det_config=args.det_config, pose_config=args.pose_config,
        merged_ckpt=args.merged_ckpt, seg_num_classes=args.seg_num_classes,
        seg_feature_dim=args.seg_feature_dim, seg_dropout=args.seg_dropout,
        image_size=args.image_size)
    prod = prod.to(device).eval()

    sapiens = torch.jit.load(args.sapiens_ckpt, map_location=device).eval()

    tfm = T.Compose([T.Resize((args.image_size, args.image_size)), T.ToTensor()])

    cap = cv2.VideoCapture(args.input)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    picks = np.linspace(0, max(0, total - 1), args.num_frames).astype(int)

    inter = np.zeros(7, dtype=np.int64)
    union = np.zeros(7, dtype=np.int64)
    agree_px = 0
    total_px = 0
    decisions = []
    t_prod = t_sap = 0.0

    for n, fidx in enumerate(picks):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x = tfm(Image.fromarray(rgb)).unsqueeze(0).to(device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = prod(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_prod += time.perf_counter() - t0
        m_prod = torch.argmax(out["seg.logits"], dim=1)[0].cpu().numpy().astype(np.uint8)
        m_prod = cv2.resize(m_prod, (w, h), interpolation=cv2.INTER_NEAREST)

        m_sap, dt = run_sapiens(sapiens, frame, device)
        t_sap += dt

        for c in range(7):
            a, b = m_prod == c, m_sap == c
            inter[c] += int(np.logical_and(a, b).sum())
            union[c] += int(np.logical_or(a, b).sum())

        fg = (m_prod > 0) | (m_sap > 0)
        agree_px += int((m_prod[fg] == m_sap[fg]).sum())
        total_px += int(fg.sum())

        d_prod = decide_hit(m_prod, w, h, region_size=args.crosshair_region)
        d_sap = decide_hit(m_sap, w, h, region_size=args.crosshair_region)
        decisions.append({
            "frame": int(fidx),
            "production": {"hit": d_prod["hit"], "part": d_prod["body_part"]},
            "sapiens": {"hit": d_sap["hit"], "part": d_sap["body_part"]},
            "same": d_prod["hit"] == d_sap["hit"] and d_prod["body_part"] == d_sap["body_part"],
        })

        panel = np.hstack([
            label_panel(frame, "original"),
            label_panel(overlay(frame, m_prod), f"production  {d_prod['body_part'] or 'MISS'}"),
            label_panel(overlay(frame, m_sap), f"sapiens  {d_sap['body_part'] or 'MISS'}"),
        ])
        cv2.imwrite(str(out_dir / "frames" / f"cmp_{n:03d}_f{int(fidx):05d}.jpg"), panel)

    cap.release()
    n = max(1, len(decisions))

    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    same = sum(1 for d in decisions if d["same"])

    print("\nper-class IoU, production vs sapiens")
    for c in range(7):
        v = iou[c]
        print(f"  {c} {BODY_PARTS[c]:<12s} {'n/a' if np.isnan(v) else f'{v:.3f}'}")
    valid = iou[1:][~np.isnan(iou[1:])]
    print(f"\n  mean IoU (body parts only)   {valid.mean():.3f}")
    print(f"  pixel agreement on person    {agree_px / max(1, total_px):.3f}")
    print(f"  identical hit verdict        {same}/{n}  ({same / n:.2f})")
    print(f"\n  latency  production {1000 * t_prod / n:6.1f} ms   "
          f"sapiens {1000 * t_sap / n:7.1f} ms   "
          f"({t_sap / max(1e-9, t_prod):.1f}x slower)")

    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "frames": n,
            "per_class_iou": {BODY_PARTS[c]: (None if np.isnan(iou[c]) else float(iou[c]))
                              for c in range(7)},
            "mean_iou_body_parts": float(valid.mean()),
            "pixel_agreement": agree_px / max(1, total_px),
            "identical_verdict_rate": same / n,
            "latency_ms": {"production": 1000 * t_prod / n, "sapiens": 1000 * t_sap / n},
            "decisions": decisions,
        }, f, indent=2)
    print(f"\n[saved] {out_dir}/summary.json and {n} comparison frames")


if __name__ == "__main__":
    main()
