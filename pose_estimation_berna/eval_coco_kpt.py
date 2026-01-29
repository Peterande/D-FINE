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


def _load_state(path: str) -> dict:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict):
        if "ema" in state and isinstance(state["ema"], dict) and "module" in state["ema"]:
            return state["ema"]["module"]
        if "model" in state and isinstance(state["model"], dict):
            return state["model"]
    return state


@torch.no_grad()
def evaluate_one(
    cfg: dict,
    checkpoint_path: str,
    device: torch.device,
    num_workers: int | None,
    max_val_samples: int | None,
    max_val_steps: int | None,
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
    state = _load_state(checkpoint_path)
    model.load_state_dict(state, strict=False)

    post = df_cfg.postprocessor.to(device).eval()
    post.remap_mscoco_category = True  # person category_id == 1

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
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device)
        preds = post(outputs, orig_target_sizes)  # list[dict]

        for t, p in zip(targets, preds):
            img_id = int(t["image_id"].reshape(-1)[0].detach().cpu().item())
            boxes = p["boxes"].detach().cpu()
            scores = p["scores"].detach().cpu()
            labels = p["labels"].detach().cpu()
            kpts = p.get("keypoints", None)
            if kpts is None:
                kpts = torch.empty((0, cfg["dataset"]["num_keypoints"], 3), dtype=torch.float32)
            else:
                kpts = kpts.detach().cpu()
            # evaluator expects keypoints as [x,y,v]; force v=1 to avoid treating low confidence as invisibility
            if kpts.numel() > 0 and kpts.shape[-1] == 3:
                kpts = kpts.clone()
                kpts[..., 2] = 1.0

            person_mask = labels == 1
            coco_predictions[img_id] = {
                "boxes": boxes[person_mask],
                "scores": scores[person_mask],
                "labels": labels[person_mask],
                "keypoints": kpts[person_mask],
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
    p.add_argument("--checkpoint", "-r", nargs="+", required=True, help="One or more .pth checkpoints to evaluate")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--max-val-samples", type=int, default=None)
    p.add_argument("--max-val-steps", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.tier, args.config_dir)
    cfg["tier"] = args.tier
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
        )
        print(f"✅ {ckpt_path}: COCO-AP(kpt)={ap:.3f}")


if __name__ == "__main__":
    main()


