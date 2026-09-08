#!/usr/bin/env python3
"""
Evaluate a D-FINE pose checkpoint on COCO val (person_keypoints_val2017) using official COCO keypoint AP.

This is meant to answer: "Which checkpoint is actually best?" without relying on visual inspection.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

# Ensure local imports work when run as a script
current_dir = os.path.dirname(os.path.abspath(__file__))  # .../pose_estimation_berna
repo_root_dir = os.path.dirname(current_dir)  # .../D-FINE-NEWBRINGER
src_dir = os.path.join(repo_root_dir, "src")
for p in [repo_root_dir, current_dir, src_dir]:
    if p and p not in sys.path:
        sys.path.insert(0, p)

from pose_estimation_berna.core.datasets import create_coco_pose_dataset  # noqa: E402
from pose_estimation_berna.train import deep_merge_dict, load_config, pose_collate_fn  # noqa: E402


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


def _load_state(path: str) -> dict:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict):
        if "ema" in state and isinstance(state["ema"], dict) and "module" in state["ema"]:
            return state["ema"]["module"]
        if "model" in state and isinstance(state["model"], dict):
            return state["model"]
    return state


def _load_state_forgiving(model: torch.nn.Module, path: str) -> dict:
    """
    Load a checkpoint but drop keys with mismatched shapes.
    Prevents eval crashes when the wrong dfine-config-path is used.
    """
    sd = _load_state(path)
    if not isinstance(sd, dict):
        return sd
    msd = model.state_dict()
    kept = {}
    dropped = 0
    for k, v in sd.items():
        if k in msd and hasattr(v, "shape") and hasattr(msd[k], "shape") and v.shape == msd[k].shape:
            kept[k] = v
        else:
            dropped += 1
    if dropped > 0:
        print(f"🧩 Forgiving load: keeping {len(kept)}/{len(sd)} keys (dropped {dropped} mismatched)")
    return kept


def _remap_labels_to_coco_category(labels: torch.Tensor) -> torch.Tensor:
    # D-FINE labels are contiguous 0..79; COCO evaluator expects category_id (person==1).
    from src.data.dataset import mscoco_label2category

    flat = labels.flatten()
    mapped = torch.tensor([mscoco_label2category[int(x.item())] for x in flat], device=labels.device)
    return mapped.view_as(labels)


def _postprocess_raw(
    outputs: dict,
    orig_target_sizes: torch.Tensor,
    num_classes: int = 80,
    num_top_queries: int = 300,
    remap_mscoco_category: bool = True,
) -> list[dict]:
    """
    Like DFINEPostProcessor, but keeps bbox-relative keypoints (pred_keypoints) so we can
    do flip-TTA and GT-bbox oracle projection during evaluation.
    Returns list of dicts with keys: labels, boxes (xyxy px), scores, keypoints_rel (bbox-rel, optional).
    """
    logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
    keypoints_rel = outputs.get("pred_keypoints", None)
    pose_quality = outputs.get("pred_pose_quality", None)

    bbox_px = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
    bbox_px = bbox_px * orig_target_sizes.repeat(1, 2).unsqueeze(1)

    scores_all = torch.sigmoid(logits)  # [B,Q,C]
    scores, flat_index = torch.topk(scores_all.flatten(1), int(num_top_queries), dim=-1)
    labels = torch.remainder(flat_index, int(num_classes))
    q_index = flat_index // int(num_classes)

    boxes_sel = bbox_px.gather(dim=1, index=q_index.unsqueeze(-1).repeat(1, 1, 4))
    kpt_sel = None
    if keypoints_rel is not None:
        kpt_sel = keypoints_rel.gather(
            dim=1,
            index=q_index.unsqueeze(-1).unsqueeze(-1).repeat(
                1, 1, keypoints_rel.shape[-2], keypoints_rel.shape[-1]
            ),
        )
    if pose_quality is not None:
        pq_sel = pose_quality.gather(
            dim=1, index=q_index.unsqueeze(-1).repeat(1, 1, pose_quality.shape[-1])
        )
        scores = scores * torch.sigmoid(pq_sel.squeeze(-1))

    if remap_mscoco_category:
        labels = _remap_labels_to_coco_category(labels)

    # clip boxes to image bounds
    w_img = orig_target_sizes[:, 0].to(boxes_sel.dtype).view(-1, 1)
    h_img = orig_target_sizes[:, 1].to(boxes_sel.dtype).view(-1, 1)
    x1 = boxes_sel[..., 0].clamp(min=0.0)
    y1 = boxes_sel[..., 1].clamp(min=0.0)
    x2 = boxes_sel[..., 2].clamp(min=0.0)
    y2 = boxes_sel[..., 3].clamp(min=0.0)
    x1 = torch.minimum(x1, w_img)
    x2 = torch.minimum(x2, w_img)
    y1 = torch.minimum(y1, h_img)
    y2 = torch.minimum(y2, h_img)
    x_min = torch.minimum(x1, x2)
    y_min = torch.minimum(y1, y2)
    x_max = torch.maximum(x1, x2)
    y_max = torch.maximum(y1, y2)
    boxes_sel = torch.stack([x_min, y_min, x_max, y_max], dim=-1)

    out: list[dict] = []
    for bi in range(labels.shape[0]):
        d = {"labels": labels[bi], "boxes": boxes_sel[bi], "scores": scores[bi]}
        if kpt_sel is not None:
            d["keypoints_rel"] = kpt_sel[bi]
        out.append(d)
    return out


def _decode_keypoints_abs(
    boxes_xyxy_px: torch.Tensor, keypoints_rel: torch.Tensor, orig_target_size_wh: torch.Tensor
) -> torch.Tensor:
    x1y1 = boxes_xyxy_px[:, None, :2]
    wh = (boxes_xyxy_px[:, None, 2:] - boxes_xyxy_px[:, None, :2]).clamp(min=1.0)
    kpt_xy = x1y1 + keypoints_rel[..., :2] * wh
    kpt_score = torch.sigmoid(keypoints_rel[..., 2])

    w = orig_target_size_wh[0].to(kpt_xy.dtype).clamp(min=1.0)
    h = orig_target_size_wh[1].to(kpt_xy.dtype).clamp(min=1.0)
    kx = torch.minimum(kpt_xy[..., 0].clamp(min=0.0), w)
    ky = torch.minimum(kpt_xy[..., 1].clamp(min=0.0), h)
    kpt_xy = torch.stack([kx, ky], dim=-1)
    return torch.cat([kpt_xy, kpt_score[..., None]], dim=-1)


def _gt_boxes_xyxy_px(target: dict, orig_target_size_wh: torch.Tensor) -> torch.Tensor:
    gt_cxcywh = target["boxes"]
    gt_xyxy = torchvision.ops.box_convert(gt_cxcywh, in_fmt="cxcywh", out_fmt="xyxy")
    gt_xyxy = gt_xyxy * orig_target_size_wh.repeat(2)
    return gt_xyxy


def _apply_oracle_gt_boxes(
    pred_boxes_xyxy_px: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_keypoints_rel: torch.Tensor | None,
    target: dict,
    orig_target_size_wh: torch.Tensor,
    iou_thr: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    gt_boxes = _gt_boxes_xyxy_px(target, orig_target_size_wh)
    if gt_boxes.numel() == 0 or pred_boxes_xyxy_px.numel() == 0:
        if pred_keypoints_rel is None:
            return pred_boxes_xyxy_px, None
        return pred_boxes_xyxy_px, _decode_keypoints_abs(
            pred_boxes_xyxy_px, pred_keypoints_rel, orig_target_size_wh
        )

    ious = torchvision.ops.box_iou(pred_boxes_xyxy_px, gt_boxes)  # [P,G]
    order = torch.argsort(pred_scores, descending=True)
    used_gt = torch.zeros((gt_boxes.shape[0],), dtype=torch.bool, device=gt_boxes.device)

    boxes_out = pred_boxes_xyxy_px.clone()
    kpts_abs = (
        None
        if pred_keypoints_rel is None
        else _decode_keypoints_abs(boxes_out, pred_keypoints_rel, orig_target_size_wh)
    )

    for pi in order.tolist():
        gi = int(torch.argmax(ious[pi]).item())
        if used_gt[gi]:
            continue
        if float(ious[pi, gi].item()) < float(iou_thr):
            continue
        used_gt[gi] = True
        boxes_out[pi] = gt_boxes[gi]
        if pred_keypoints_rel is not None:
            kpts_abs[pi : pi + 1] = _decode_keypoints_abs(
                boxes_out[pi : pi + 1], pred_keypoints_rel[pi : pi + 1], orig_target_size_wh
            )

    return boxes_out, kpts_abs


@torch.no_grad()
def evaluate_one(
    cfg: dict,
    checkpoint_path: str,
    device: torch.device,
    num_workers: int | None,
    max_val_samples: int | None,
    max_val_steps: int | None,
    flip_tta: bool = False,
    oracle_gt_boxes: bool = False,
    oracle_iou_thr: float = 0.5,
    nms_iou_thr: float = 0.6,
) -> float:
    from src.core import YAMLConfig
    from faster_coco_eval import COCO
    from src.data.dataset.coco_eval import CocoEvaluator

    base_dir = Path(__file__).resolve().parent
    repo_root = base_dir.parent

    dfine_cfg = Path(cfg["dfine"]["config_path"])
    if not dfine_cfg.is_absolute():
        dfine_cfg = base_dir / dfine_cfg

    df_cfg = YAMLConfig(str(dfine_cfg))
    if "HGNetv2" in df_cfg.yaml_cfg:
        df_cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    model = df_cfg.model.to(device).eval()
    # Allow variable-resolution eval by disabling fixed eval_spatial_size (pos-embed cache).
    for m in model.modules():
        if hasattr(m, "eval_spatial_size"):
            try:
                m.eval_spatial_size = None
            except Exception:
                pass
    try:
        state = _load_state(checkpoint_path)
        model.load_state_dict(state, strict=False)
    except RuntimeError as e:
        msg = str(e)
        if "size mismatch" in msg or "Error(s) in loading state_dict" in msg:
            print("⚠️ Checkpoint/model shape mismatch; falling back to forgiving load. (Fix by passing --dfine-config-path)")
            state = _load_state_forgiving(model, checkpoint_path)
            model.load_state_dict(state, strict=False)
        else:
            raise

    root_dir = cfg["dataset"]["root_dir"]
    root_dir = str((repo_root / root_dir) if not str(root_dir).startswith("/") else root_dir)

    val_ds = create_coco_pose_dataset(
        root_dir=root_dir,
        split="val",
        image_size=cfg["dataset"]["image_size"],
        num_keypoints=cfg["dataset"]["num_keypoints"],
        tier=str(cfg.get("tier", "standard")),
    )
    if max_val_samples is not None:
        n = max(0, int(max_val_samples))
        val_ds = torch.utils.data.Subset(val_ds, list(range(min(n, len(val_ds)))))

    val_bs = int(cfg.get("training", {}).get("val_batch_size", cfg.get("training", {}).get("batch_size", 4)))
    nw = int(num_workers) if num_workers is not None else int(cfg.get("system", {}).get("num_workers", 2))
    use_pin_memory = bool(cfg.get("system", {}).get("pin_memory", True)) and device.type == "cuda"
    val_loader = DataLoader(
        val_ds,
        batch_size=val_bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=use_pin_memory,
        collate_fn=pose_collate_fn,
    )

    coco_val_ann = Path(root_dir) / "annotations" / "person_keypoints_val2017.json"
    coco_gt = COCO(str(coco_val_ann))
    coco_evaluator = CocoEvaluator(coco_gt, iou_types=["keypoints"])
    coco_evaluator.cleanup()

    coco_predictions: dict[int, dict] = {}
    total = len(val_loader)
    if max_val_steps is not None:
        total = min(total, int(max_val_steps))

    for step, (images, targets) in enumerate(tqdm(val_loader, total=total, desc="COCOeval")):
        images = images.to(device, non_blocking=(device.type == "cuda"))
        outputs = model(images)
        if flip_tta:
            images_f = torch.flip(images, dims=[-1])
            out_f = model(images_f)
            # unflip normalized outputs
            if "pred_boxes" in out_f and out_f["pred_boxes"] is not None:
                out_f["pred_boxes"] = out_f["pred_boxes"].clone()
                out_f["pred_boxes"][..., 0] = 1.0 - out_f["pred_boxes"][..., 0]
            if "pred_keypoints" in out_f and out_f["pred_keypoints"] is not None:
                k = out_f["pred_keypoints"].clone()
                # Two supported formats:
                # - DFINE: [B,Q,K,3] bbox-relative
                # - DETRPose-style: [B,Q,2K] image-normalized
                if k.ndim == 4:
                    k[..., 0] = 1.0 - k[..., 0]
                    k = k[..., COCO_KEYPOINT_FLIP_INDEX, :]
                elif k.ndim == 3:
                    # flip x for each (x,y) pair in the flattened vector
                    k[..., 0::2] = 1.0 - k[..., 0::2]
                    k = k.view(k.shape[0], k.shape[1], -1, 2)
                    k = k[:, :, COCO_KEYPOINT_FLIP_INDEX, :]
                    k = k.flatten(-2)
                else:
                    raise ValueError(f"Unexpected pred_keypoints shape for flip-TTA: {tuple(k.shape)}")
                out_f["pred_keypoints"] = k

            outputs = dict(outputs)
            for kk in ["pred_logits", "pred_boxes", "pred_keypoints", "pred_pose_quality"]:
                if kk in outputs and kk in out_f and outputs[kk] is not None and out_f[kk] is not None:
                    outputs[kk] = torch.cat([outputs[kk], out_f[kk]], dim=1)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device)
        num_top = int(df_cfg.yaml_cfg.get("DFINEPostProcessor", {}).get("num_top_queries", 300))
        if "pred_boxes" in outputs and outputs.get("pred_boxes", None) is not None:
            preds = _postprocess_raw(
                outputs,
                orig_target_sizes=orig_target_sizes,
                num_classes=int(cfg.get("dfine", {}).get("num_classes", 80)),
                num_top_queries=num_top,
                remap_mscoco_category=True,
            )
            preds_are_pose_only = False
        else:
            from pose_estimation_berna.core.postprocess_detrpose import DETRPosePostProcessor

            detrpose_num_classes = int(df_cfg.yaml_cfg.get("num_classes", cfg.get("dfine", {}).get("num_classes", 80)))
            preds = DETRPosePostProcessor(
                num_classes=detrpose_num_classes,
                num_keypoints=int(cfg.get("dataset", {}).get("num_keypoints", 17)),
                num_top_queries=num_top,
                remap_mscoco_category=True,
            ).to(device).eval()(outputs, orig_target_sizes)
            preds_are_pose_only = True

        for t, p in zip(targets, preds):
            img_id = int(t["image_id"].reshape(-1)[0].detach().cpu().item())
            boxes = p["boxes"].detach().cpu()
            scores = p["scores"].detach().cpu()
            labels = p["labels"].detach().cpu()
            kpts_rel = p.get("keypoints_rel", None)
            kpts_abs = p.get("keypoints", None)
            if kpts_rel is not None:
                kpts_rel = kpts_rel.detach().cpu()
            if kpts_abs is not None:
                kpts_abs = kpts_abs.detach().cpu()

            # Keep only person detections (COCO category_id == 1)
            person_mask = labels == 1
            boxes = boxes[person_mask]
            scores = scores[person_mask]
            labels = labels[person_mask]
            if kpts_rel is not None:
                kpts_rel = kpts_rel[person_mask]
            if kpts_abs is not None:
                kpts_abs = kpts_abs[person_mask]

            # Merge duplicates (especially important for flip-TTA) using NMS.
            if boxes.numel() > 0 and float(nms_iou_thr) > 0:
                keep = torchvision.ops.nms(boxes, scores, float(nms_iou_thr))
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]
                if kpts_rel is not None:
                    kpts_rel = kpts_rel[keep]
                if kpts_abs is not None:
                    kpts_abs = kpts_abs[keep]

            if preds_are_pose_only:
                kpts = (
                    torch.empty((0, cfg["dataset"]["num_keypoints"], 3), dtype=torch.float32)
                    if kpts_abs is None
                    else kpts_abs
                )
            else:
                if kpts_rel is None:
                    kpts = torch.empty((0, cfg["dataset"]["num_keypoints"], 3), dtype=torch.float32)
                else:
                    if oracle_gt_boxes:
                        boxes, kpts = _apply_oracle_gt_boxes(
                            pred_boxes_xyxy_px=boxes,
                            pred_scores=scores,
                            pred_keypoints_rel=kpts_rel,
                            target=t,
                            orig_target_size_wh=t["orig_size"].detach().cpu(),
                            iou_thr=float(oracle_iou_thr),
                        )
                        if kpts is None:
                            kpts = torch.empty((0, cfg["dataset"]["num_keypoints"], 3), dtype=torch.float32)
                    else:
                        kpts = _decode_keypoints_abs(boxes, kpts_rel, t["orig_size"].detach().cpu())
            # evaluator expects keypoints as [x,y,v]; force v=1 to avoid treating low confidence as invisibility
            if kpts.numel() > 0 and kpts.shape[-1] == 3:
                kpts = kpts.clone()
                kpts[..., 2] = 1.0

            coco_predictions[img_id] = {
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
                "keypoints": kpts,
            }

        if max_val_steps is not None and (step + 1) >= int(max_val_steps):
            break

    coco_evaluator.update(coco_predictions)
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    stats = getattr(coco_evaluator.coco_eval["keypoints"], "stats", None)
    ap = float(stats[0]) if stats is not None and len(stats) > 0 else float("nan")
    return ap


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tier", default="standard", choices=["lightweight", "standard", "advanced"])
    p.add_argument("--config-dir", default="pose_estimation_berna/configs")
    p.add_argument(
        "--dfine-config-path",
        default=None,
        help="Override cfg.dfine.config_path (must match checkpoint architecture).",
    )
    p.add_argument("--checkpoint", "-r", nargs="+", required=True, help="One or more .pth checkpoints to evaluate")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--max-val-samples", type=int, default=None)
    p.add_argument("--max-val-steps", type=int, default=None)
    p.add_argument("--flip-tta", action="store_true", help="Enable horizontal flip test-time augmentation.")
    p.add_argument(
        "--oracle-gt-boxes",
        action="store_true",
        help="Diagnostic: match preds to GT boxes by IoU and re-project keypoints using GT boxes.",
    )
    p.add_argument("--oracle-iou-thr", type=float, default=0.5, help="IoU threshold for GT-box oracle matching.")
    p.add_argument("--nms-iou", type=float, default=0.6, help="Box NMS IoU threshold (use 0 to disable).")
    p.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Override config dataset.image_size (square resize). Example: --image-size 896",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.tier, args.config_dir)
    cfg["tier"] = args.tier
    if args.dfine_config_path is not None:
        cfg.setdefault("dfine", {})
        cfg["dfine"]["config_path"] = str(args.dfine_config_path)
        print(f"🧾 Overriding dfine.config_path -> {cfg['dfine']['config_path']}")
    if args.image_size is not None:
        cfg.setdefault("dataset", {})
        cfg["dataset"]["image_size"] = int(args.image_size)
        print(f"🖼️ Overriding dataset.image_size -> {cfg['dataset']['image_size']}")
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"🚀 Evaluating on device: {device}")

    for ckpt in args.checkpoint:
        ckpt_path = Path(ckpt)
        if not ckpt_path.is_absolute():
            ckpt_path = Path(repo_root_dir) / ckpt_path
        ap = evaluate_one(
            cfg=cfg,
            checkpoint_path=str(ckpt_path),
            device=device,
            num_workers=args.num_workers,
            max_val_samples=args.max_val_samples,
            max_val_steps=args.max_val_steps,
            flip_tta=bool(args.flip_tta),
            oracle_gt_boxes=bool(args.oracle_gt_boxes),
            oracle_iou_thr=float(args.oracle_iou_thr),
            nms_iou_thr=float(args.nms_iou),
        )
        print(f"✅ {ckpt_path}: COCO-AP(kpt)={ap:.3f}")


if __name__ == "__main__":
    main()
