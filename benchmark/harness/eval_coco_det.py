"""Measure the production detection head on COCO val2017.

The merged checkpoint carries an 80-class COCO detection head
(det_decoder.enc_score_head is (80, 256)), so its AP is directly comparable to
the numbers DEIMv2, ECDet and D-FINE report — unlike anything measured on the
2-class soldier set.

Preprocessing mirrors infer_video_singlepass_x.py exactly: resize to a square
image_size then ToTensor, with NO ImageNet mean/std. The monorepo runtime
normalises differently; that discrepancy is recorded in the audit, and matching
the tool that demonstrably produces correct detections is the safer baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
for p in [REPO, REPO / "src", REPO / "tools", REPO / "segmentation_sivert", REPO / "pose_estimation_berna"]:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from benchmark.harness.infer_crosshair import build_model_from_merged  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det-config", required=True)
    ap.add_argument("--pose-config", required=True)
    ap.add_argument("--merged-ckpt", help="Shared-backbone segpose checkpoint.")
    ap.add_argument("--plain-ckpt",
                    help="Stock D-FINE checkpoint, loaded straight into the detection model. "
                         "Control run: same preprocessing and postprocessing, no model surgery.")
    ap.add_argument("--coco-root", required=True, help="Directory holding val2017/ and annotations/")
    ap.add_argument("--split", default="val2017")
    ap.add_argument("--image-size", type=int, default=640)
    ap.add_argument("--seg-num-classes", type=int, default=7)
    ap.add_argument("--seg-feature-dim", type=int, default=384)
    ap.add_argument("--seg-dropout", type=float, default=0.1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument(
        "--imagenet-normalize",
        action="store_true",
        help="Controlled drift experiment: apply production ImageNet mean/std after ToTensor.",
    )
    ap.add_argument("--limit", type=int, default=None, help="Evaluate only the first N images.")
    ap.add_argument("--out-json", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    root = Path(args.coco_root)
    ann_file = root / "annotations" / f"instances_{args.split}.json"
    img_dir = root / args.split

    if bool(args.merged_ckpt) == bool(args.plain_ckpt):
        raise SystemExit("pass exactly one of --merged-ckpt or --plain-ckpt")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    if args.plain_ckpt:
        from src.core import YAMLConfig

        det_cfg = YAMLConfig(str(args.det_config))
        base = det_cfg.model
        ckpt = torch.load(args.plain_ckpt, map_location="cpu")
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        if isinstance(ckpt, dict) and "ema" in ckpt and isinstance(ckpt["ema"], dict):
            state = ckpt["ema"].get("module", state)
        missing, unexpected = base.load_state_dict(state, strict=False)
        print(f"[plain-load] missing={len(missing)} unexpected={len(unexpected)}")

        class _DetOnly(torch.nn.Module):
            """Wrap the stock model so it emits the same keys as the merged one."""

            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                o = self.m(x)
                return {"det.pred_logits": o["pred_logits"], "det.pred_boxes": o["pred_boxes"]}

        model = _DetOnly(base)
    else:
        model, det_cfg = build_model_from_merged(
            det_config=args.det_config,
            pose_config=args.pose_config,
            merged_ckpt=args.merged_ckpt,
            seg_num_classes=args.seg_num_classes,
            seg_feature_dim=args.seg_feature_dim,
            seg_dropout=args.seg_dropout,
            image_size=args.image_size,
        )

    model = model.to(device).eval()

    det_post = det_cfg.postprocessor.to(device).eval()
    if hasattr(det_post, "remap_mscoco_category"):
        # Emit real COCO category ids so COCOeval can match them.
        det_post.remap_mscoco_category = True

    transforms = [T.Resize((args.image_size, args.image_size)), T.ToTensor()]
    if args.imagenet_normalize:
        transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    tfm = T.Compose(transforms)

    coco = COCO(str(ann_file))
    img_ids = sorted(coco.getImgIds())
    if args.limit:
        img_ids = img_ids[: args.limit]

    results = []
    t_model = 0.0
    t0_all = time.perf_counter()

    for i, img_id in enumerate(img_ids):
        info = coco.loadImgs(img_id)[0]
        img = Image.open(img_dir / info["file_name"]).convert("RGB")
        w0, h0 = img.size

        x = tfm(img).unsqueeze(0).to(device)
        orig_size = torch.tensor([[w0, h0]], device=device)

        t0 = time.perf_counter()
        with torch.inference_mode():
            with torch.amp.autocast("cuda", enabled=bool(args.amp) and device.type == "cuda"):
                outputs = model(x)
            det_dict = {
                "pred_logits": outputs["det.pred_logits"],
                "pred_boxes": outputs["det.pred_boxes"],
            }
            res = det_post(det_dict, orig_size)[0]
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_model += time.perf_counter() - t0

        boxes = res["boxes"].cpu().numpy()
        scores = res["scores"].cpu().numpy()
        labels = res["labels"].cpu().numpy()
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = [float(v) for v in box]
            results.append({
                "image_id": int(img_id),
                "category_id": int(label),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
            })

        if (i + 1) % 500 == 0:
            print(f"[eval] {i + 1}/{len(img_ids)} images, {len(results)} dets", flush=True)

    wall = time.perf_counter() - t0_all
    print(f"[eval] done: {len(img_ids)} images in {wall:.1f}s "
          f"({1000.0 * t_model / len(img_ids):.1f} ms/img model time)")

    if not results:
        print("[eval] no detections produced — nothing to score")
        return

    out_json = args.out_json or "runs/coco_det_results.json"
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f)

    coco_dt = coco.loadRes(out_json)
    ev = COCOeval(coco, coco_dt, "bbox")
    ev.params.imgIds = img_ids
    ev.evaluate()
    ev.accumulate()
    ev.summarize()


if __name__ == "__main__":
    main()
