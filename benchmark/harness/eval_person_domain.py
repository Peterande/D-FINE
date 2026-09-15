"""Person detection on Tagtwo-domain data: merged multihead vs stock D-FINE.

COCO val2017 answers "is the detection head weaker", but not "does it matter
here". This runs both checkpoints over soldier_coco val — real domain images
with ground-truth boxes — and reports the three numbers that decide it:

  recall      persons found (a person never detected can never be hit)
  false pos.  detections with no matching person (a false hit)
  box IoU     localisation quality on the ones it did find

Both ground-truth classes (Civilian, Soldier) are persons, so they collapse to
one class. Model predictions are filtered to COCO's person category.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
for p in [REPO, REPO / "src", REPO / "tools", REPO / "segmentation_sivert", REPO / "pose_estimation_berna"]:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from benchmark.harness.infer_crosshair import build_model_from_merged  # noqa: E402

COCO_PERSON_ID = 1


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N,4) xyxy, b: (M,4) xyxy -> (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return (inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)).astype(np.float32)


def load_model(args, which):
    from src.core import YAMLConfig

    if which == "stock":
        det_cfg = YAMLConfig(str(args.det_config))
        base = det_cfg.model
        ckpt = torch.load(args.plain_ckpt, map_location="cpu")
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        if isinstance(ckpt, dict) and "ema" in ckpt and isinstance(ckpt["ema"], dict):
            state = ckpt["ema"].get("module", state)
        base.load_state_dict(state, strict=False)

        class _DetOnly(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                o = self.m(x)
                return {"det.pred_logits": o["pred_logits"], "det.pred_boxes": o["pred_boxes"]}

        return _DetOnly(base), det_cfg

    model, det_cfg = build_model_from_merged(
        det_config=args.det_config,
        pose_config=args.pose_config,
        merged_ckpt=args.merged_ckpt,
        seg_num_classes=args.seg_num_classes,
        seg_feature_dim=args.seg_feature_dim,
        seg_dropout=args.seg_dropout,
        image_size=args.image_size,
    )
    return model, det_cfg


def run(args, which, coco, img_ids, img_dir, device):
    model, det_cfg = load_model(args, which)
    model = model.to(device).eval()
    det_post = det_cfg.postprocessor.to(device).eval()
    if hasattr(det_post, "remap_mscoco_category"):
        det_post.remap_mscoco_category = True

    tfm = T.Compose([T.Resize((args.image_size, args.image_size)), T.ToTensor()])

    n_gt = 0
    n_matched = 0
    n_fp = 0
    ious = []

    for img_id in img_ids:
        info = coco.loadImgs(img_id)[0]
        path = img_dir / info["file_name"]
        if not path.exists():
            continue
        img = Image.open(path).convert("RGB")
        w0, h0 = img.size

        gt = []
        for a in coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=False)):
            x, y, w, h = a["bbox"]
            gt.append([x, y, x + w, y + h])
        gt = np.array(gt, dtype=np.float32).reshape(-1, 4)

        x = tfm(img).unsqueeze(0).to(device)
        with torch.inference_mode():
            out = model(x)
            res = det_post({"pred_logits": out["det.pred_logits"],
                            "pred_boxes": out["det.pred_boxes"]},
                           torch.tensor([[w0, h0]], device=device))[0]

        keep = (res["labels"].cpu().numpy() == COCO_PERSON_ID) & \
               (res["scores"].cpu().numpy() >= args.score_thr)
        pred = res["boxes"].cpu().numpy()[keep].reshape(-1, 4)

        n_gt += len(gt)
        M = iou_matrix(pred, gt)
        used_gt = set()
        matched_pred = set()
        if M.size:
            order = np.argsort(-M.max(axis=1)) if len(pred) else []
            for pi in order:
                gi = int(np.argmax(M[pi]))
                if M[pi, gi] >= args.iou_thr and gi not in used_gt:
                    used_gt.add(gi)
                    matched_pred.add(int(pi))
                    ious.append(float(M[pi, gi]))
        n_matched += len(used_gt)
        n_fp += len(pred) - len(matched_pred)

    recall = n_matched / max(1, n_gt)
    return {
        "model": which,
        "gt_persons": n_gt,
        "found": n_matched,
        "recall": recall,
        "false_positives": n_fp,
        "fp_per_image": n_fp / max(1, len(img_ids)),
        "mean_iou_matched": float(np.mean(ious)) if ious else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det-config", required=True)
    ap.add_argument("--pose-config", required=True)
    ap.add_argument("--merged-ckpt", required=True)
    ap.add_argument("--plain-ckpt", required=True)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--image-size", type=int, default=640)
    ap.add_argument("--seg-num-classes", type=int, default=7)
    ap.add_argument("--seg-feature-dim", type=int, default=384)
    ap.add_argument("--seg-dropout", type=float, default=0.1)
    ap.add_argument("--score-thr", type=float, default=0.5)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-json", default="runs/person_domain.json")
    args = ap.parse_args()

    from pycocotools.coco import COCO

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    coco = COCO(args.ann)
    img_ids = sorted(coco.getImgIds())
    img_dir = Path(args.img_dir)

    rows = [run(args, w, coco, img_ids, img_dir, device) for w in ("merged", "stock")]

    print(f"\n{'model':10s} {'recall':>8s} {'found/gt':>12s} {'FP':>6s} {'FP/img':>8s} {'IoU':>7s}")
    for r in rows:
        print(f"{r['model']:10s} {r['recall']:8.3f} "
              f"{r['found']:>5d}/{r['gt_persons']:<6d} {r['false_positives']:6d} "
              f"{r['fp_per_image']:8.2f} {r['mean_iou_matched']:7.3f}")

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({"score_thr": args.score_thr, "iou_thr": args.iou_thr,
                   "images": len(img_ids), "results": rows}, f, indent=2)
    print(f"\n[saved] {args.out_json}")


if __name__ == "__main__":
    main()
