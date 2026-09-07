"""Measure the production pose head on COCO person_keypoints_val2017 (OKS).

Fills the largest gap in MODEL_BASELINE.md. Unlike the detection evaluation
there is no stock control available — no DETRPose checkpoint exists locally —
so these numbers stand on their own and cannot be validated against a published
figure the way the 59.3 AP control validated the detection harness. Treat them
as a baseline to compare future candidates against, not as proof the harness is
calibrated.

COCO keypoint evaluation scores the person category only, so every prediction
is emitted as category_id 1 regardless of the pose head's own class slot.
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

COCO_PERSON_ID = 1


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det-config", required=True)
    ap.add_argument("--pose-config", required=True)
    ap.add_argument("--merged-ckpt", required=True)
    ap.add_argument("--pose-adapter", type=Path, default=None,
                    help="Optional isolated pose-adapter training checkpoint.")
    ap.add_argument("--coco-root", required=True)
    ap.add_argument("--split", default="val2017")
    ap.add_argument("--image-size", type=int, default=640)
    ap.add_argument("--seg-num-classes", type=int, default=7)
    ap.add_argument("--seg-feature-dim", type=int, default=384)
    ap.add_argument("--seg-dropout", type=float, default=0.1)
    ap.add_argument("--pose-num-classes", type=int, default=2)
    ap.add_argument("--num-top-queries", type=int, default=300)
    ap.add_argument("--score-thr", type=float, default=0.0,
                    help="Keep all queries by default; COCOeval handles ranking itself.")
    ap.add_argument("--max-dets", type=int, default=20,
                    help="Per-image detection cap. COCO keypoint protocol uses 20.")
    ap.add_argument("--orig-size-order", choices=["wh", "hw"], default="wh")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-json", default="runs/coco_pose_results.json")
    return ap.parse_args()


def main():
    args = parse_args()
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pose_estimation_berna.core.postprocess_detrpose import DETRPosePostProcessor

    root = Path(args.coco_root)
    ann_file = root / "annotations" / f"person_keypoints_{args.split}.json"
    img_dir = root / args.split

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, _ = build_model_from_merged(
        det_config=args.det_config,
        pose_config=args.pose_config,
        merged_ckpt=args.merged_ckpt,
        seg_num_classes=args.seg_num_classes,
        seg_feature_dim=args.seg_feature_dim,
        seg_dropout=args.seg_dropout,
        image_size=args.image_size,
    )
    if args.pose_adapter is not None:
        from benchmark.pose_adapter.model import PoseAdapterModel

        adapted = PoseAdapterModel(model, [384, 384, 384])
        adapter_ckpt = torch.load(args.pose_adapter, map_location="cpu", weights_only=False)
        adapted.pose_adapters.load_state_dict(adapter_ckpt["pose_adapters"], strict=True)
        adapted.pose_decoder.load_state_dict(adapter_ckpt["pose_decoder"], strict=True)
        model = adapted
    model = model.to(device).eval()

    pose_post = DETRPosePostProcessor(
        num_classes=args.pose_num_classes,
        num_keypoints=17,
        num_top_queries=args.num_top_queries,
        remap_mscoco_category=False,
    ).to(device).eval()

    tfm = T.Compose([T.Resize((args.image_size, args.image_size)), T.ToTensor()])

    coco = COCO(str(ann_file))
    # Only images that actually contain annotated people can score.
    img_ids = sorted(coco.getImgIds(catIds=[COCO_PERSON_ID]))
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
        size = ([[w0, h0]] if args.orig_size_order == "wh" else [[h0, w0]])
        orig_size = torch.tensor(size, device=device)

        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model(x)
            res = pose_post({"pred_logits": out["pose.pred_logits"],
                             "pred_keypoints": out["pose.pred_keypoints"]},
                            orig_size)[0]
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_model += time.perf_counter() - t0

        scores = res["scores"].cpu().numpy()
        kpts = res["keypoints"].cpu().numpy()  # [top, K, 3]

        order = scores.argsort()[::-1][: args.max_dets]
        for idx in order:
            if scores[idx] < args.score_thr:
                continue
            flat = []
            for k in range(kpts.shape[1]):
                flat += [float(kpts[idx, k, 0]), float(kpts[idx, k, 1]), 1.0]
            results.append({
                "image_id": int(img_id),
                "category_id": COCO_PERSON_ID,
                "keypoints": flat,
                "score": float(scores[idx]),
            })

        if (i + 1) % 500 == 0:
            print(f"[pose-eval] {i + 1}/{len(img_ids)} images, {len(results)} dets", flush=True)

    wall = time.perf_counter() - t0_all
    print(f"[pose-eval] done: {len(img_ids)} images in {wall:.1f}s "
          f"({1000.0 * t_model / max(1, len(img_ids)):.1f} ms/img model time)")

    if not results:
        print("[pose-eval] no keypoint detections produced — nothing to score")
        return

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f)

    coco_dt = coco.loadRes(args.out_json)
    ev = COCOeval(coco, coco_dt, "keypoints")
    ev.params.imgIds = img_ids
    ev.evaluate()
    ev.accumulate()
    ev.summarize()


if __name__ == "__main__":
    main()
