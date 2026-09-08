#!/usr/bin/env python3
"""
Run D-FINE pose (per-query keypoints) on a single image and visualize results.

Outputs:
- annotated image with person boxes + COCO-17 keypoints skeleton
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.core import YAMLConfig  # noqa: E402


# COCO-17 skeleton in **0-indexed** keypoint order used by COCO annotations:
# [nose, leye, reye, lear, rear, lsho, rsho, lelb, relb, lwri, rwri, lhip, rhip, lkne, rkne, lank, rank]
COCO_SKELETON = [
    (15, 13),  # left ankle - left knee
    (13, 11),  # left knee - left hip
    (16, 14),  # right ankle - right knee
    (14, 12),  # right knee - right hip
    (11, 12),  # left hip - right hip
    (5, 11),   # left shoulder - left hip
    (6, 12),   # right shoulder - right hip
    (5, 6),    # left shoulder - right shoulder
    (5, 7),    # left shoulder - left elbow
    (7, 9),    # left elbow - left wrist
    (6, 8),    # right shoulder - right elbow
    (8, 10),   # right elbow - right wrist
    (0, 1),    # nose - left eye
    (0, 2),    # nose - right eye
    (1, 3),    # left eye - left ear
    (2, 4),    # right eye - right ear
    (3, 5),    # left ear - left shoulder
    (4, 6),    # right ear - right shoulder
    (1, 2),    # left eye - right eye
]

# Swap left/right keypoint indices for COCO-17 (useful for debugging if the model outputs are permuted).
COCO_KEYPOINT_FLIP_INDEX = [
    0,  # nose
    2, 1,  # left_eye <-> right_eye
    4, 3,  # left_ear <-> right_ear
    6, 5,  # left_shoulder <-> right_shoulder
    8, 7,  # left_elbow <-> right_elbow
    10, 9,  # left_wrist <-> right_wrist
    12, 11,  # left_hip <-> right_hip
    14, 13,  # left_knee <-> right_knee
    16, 15,  # left_ankle <-> right_ankle
]


def load_state_dict(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "ema" in ckpt and isinstance(ckpt["ema"], dict) and "module" in ckpt["ema"]:
            return ckpt["ema"]["module"]
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
    # fallback: assume it's already a state_dict
    return ckpt


def draw_pose(
    img_bgr: np.ndarray,
    box_xyxy: np.ndarray,
    keypoints: np.ndarray,
    color=(0, 255, 0),
    kpt_thr: float = 0.2,
    show_kpt_idx: bool = False,
):
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = box_xyxy.tolist()
    x1i = int(np.clip(round(x1), 0, max(0, w - 1)))
    y1i = int(np.clip(round(y1), 0, max(0, h - 1)))
    x2i = int(np.clip(round(x2), 0, max(0, w - 1)))
    y2i = int(np.clip(round(y2), 0, max(0, h - 1)))
    cv2.rectangle(img_bgr, (x1i, y1i), (x2i, y2i), color, 2)

    # keypoints: [17,3] (x,y,score)
    for i in range(keypoints.shape[0]):
        x, y, s = keypoints[i]
        if s < kpt_thr:
            continue
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        cv2.circle(img_bgr, (int(x), int(y)), 3, (0, 0, 255), -1)
        if show_kpt_idx:
            cv2.putText(
                img_bgr,
                str(int(i)),
                (int(x) + 4, int(y) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

    for a, b in COCO_SKELETON:
        xa, ya, sa = keypoints[a]
        xb, yb, sb = keypoints[b]
        if sa < kpt_thr or sb < kpt_thr:
            continue
        if xa < 0 or ya < 0 or xa >= w or ya >= h:
            continue
        if xb < 0 or yb < 0 or xb >= w or yb >= h:
            continue
        cv2.line(img_bgr, (int(xa), int(ya)), (int(xb), int(yb)), (255, 0, 0), 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        "-c",
        default="pose_estimation_berna/base_dfine/dfine_hgnetv2_x_obj2coco_pose.yml",
        help="D-FINE pose config (must enable num_keypoints=17)",
    )
    p.add_argument(
        "--checkpoint",
        "-r",
        default="outputs/lightweight/lightweight/best.pth",
        help="Checkpoint path (state_dict or dict with 'model'/'ema')",
    )
    p.add_argument("--image", "-i", required=True, help="Input image path")
    p.add_argument("--out", "-o", default="pose_result.jpg", help="Output image path")
    p.add_argument("--device", "-d", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--score-thr", type=float, default=0.4)
    p.add_argument("--kpt-thr", type=float, default=0.5, help="Keypoint score threshold for drawing")
    p.add_argument("--show-kpt-idx", action="store_true", help="Draw keypoint indices (0-16) next to points")
    p.add_argument(
        "--swap-lr-kpts",
        action="store_true",
        help="Swap left/right keypoint indices before drawing (debug for permuted outputs)",
    )
    p.add_argument(
        "--force-top1",
        action="store_true",
        help="Always draw the best-scoring person, even if below --score-thr (debug)",
    )
    p.add_argument("--verbose", action="store_true", help="Print debug info about detections")
    p.add_argument("--max-persons", type=int, default=5)
    args = p.parse_args()

    cfg = YAMLConfig(args.config)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    state = load_state_dict(args.checkpoint)
    cfg.model.load_state_dict(state, strict=False)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = cfg.model.deploy().to(device).eval()
    post = cfg.postprocessor.to(device).eval()  # DON'T deploy(); we want dict results incl keypoints
    # Match video inference: map contiguous labels -> MSCOCO category ids (person == 1)
    post.remap_mscoco_category = True

    im_pil = Image.open(args.image).convert("RGB")
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]], device=device)

    tfm = T.Compose([T.Resize((640, 640)), T.ToTensor()])
    x = tfm(im_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(x)
        results = post(outputs, orig_size)  # list[dict]

    img_bgr = cv2.cvtColor(np.array(im_pil), cv2.COLOR_RGB2BGR)
    det = results[0]
    labels = det["labels"].detach().cpu().numpy()
    boxes = det["boxes"].detach().cpu().numpy()
    scores = det["scores"].detach().cpu().numpy()
    keypoints = det.get("keypoints", None)
    if keypoints is not None:
        keypoints = keypoints.detach().cpu().numpy()  # [Q,17,3]

    # filter persons
    person_mask = np.isin(labels, [0, 1])  # robust across remap/non-remap configs
    keep = (scores >= args.score_thr) & person_mask
    idxs = np.where(keep)[0]
    if idxs.size > 0:
        idxs = idxs[np.argsort(scores[idxs])[::-1]]
        idxs = idxs[: args.max_persons]
    elif args.force_top1 and np.any(person_mask):
        # Pick best person even if below threshold (debug)
        cand = np.where(person_mask)[0]
        best = cand[int(np.argmax(scores[cand]))]
        idxs = np.array([best], dtype=np.int64)

    if args.verbose:
        best_idx = int(np.argmax(scores)) if scores.size > 0 else -1
        best_score = float(scores[best_idx]) if best_idx >= 0 else float("nan")
        best_label = int(labels[best_idx]) if best_idx >= 0 else -1
        best_person_score = float(np.max(scores[person_mask])) if np.any(person_mask) else float("nan")
        print(
            f"[infer_image] total={scores.size} best=(label={best_label}, score={best_score:.3f}) "
            f"best_person_score={best_person_score:.3f} score_thr={float(args.score_thr):.3f} "
            f"drawing={int(len(idxs))}"
        )

    for j in idxs:
        if keypoints is not None:
            kpt = keypoints[j]
            if args.swap_lr_kpts:
                kpt = kpt[COCO_KEYPOINT_FLIP_INDEX, :]
            draw_pose(
                img_bgr,
                boxes[j],
                kpt,
                kpt_thr=float(args.kpt_thr),
                show_kpt_idx=bool(args.show_kpt_idx),
            )
        else:
            x1, y1, x2, y2 = boxes[j].astype(int).tolist()
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            img_bgr,
            f"person {scores[j]:.2f}",
            (int(boxes[j][0]), max(0, int(boxes[j][1]) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    cv2.imwrite(args.out, img_bgr)
    print(f"✅ Saved: {args.out}")


if __name__ == "__main__":
    main()


