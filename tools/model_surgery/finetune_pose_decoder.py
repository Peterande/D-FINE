#!/usr/bin/env python3
"""Fine-tune pose_decoder to work with det-encoder features.

After model surgery (build_pose_seg_singlepass_x.py), the pose_decoder sees
encoder features it was never trained on (det-encoder vs pose-encoder).
This script fixes that by fine-tuning ONLY pose_decoder on COCO keypoints,
while keeping backbone, encoder, det_decoder, and seg_head completely frozen.

Usage:
    python tools/model_surgery/finetune_pose_decoder.py \
        --det-config segmentation_sivert/base_dfine/dfine_hgnetv2_x_obj2coco.yml \
        --pose-config pose_estimation_berna/base_dfine/dfine_hgnetv2_x_obj2coco_detrpose_paper.yml \
        --det-ckpt outputs/standard/dfine_0.73.pth \
        --pose-ckpt outputs/standard/detrpose_paper_x_v1/best.pth \
        --seg-ckpt outputs/standard/dfine_0.73.pth \
        --coco-root /path/to/coco \
        --epochs 15 \
        --out outputs/finetuned_merged.pth
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Repo paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[2]

for p in [REPO, REPO / "src", REPO / "segmentation_sivert",
          REPO / "pose_estimation_berna", REPO / "tools"]:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)


def parse_args():
    ap = argparse.ArgumentParser(description="Fine-tune pose_decoder on merged model")
    # Model / checkpoint paths
    ap.add_argument("--det-config", required=True, help="Det/seg YAML config")
    ap.add_argument("--pose-config", required=True, help="Pose YAML config")
    ap.add_argument("--det-ckpt", required=True, help="Det/seg checkpoint (.pth)")
    ap.add_argument("--pose-ckpt", required=True, help="Pose checkpoint (.pth)")
    ap.add_argument("--seg-ckpt", required=True, help="Seg checkpoint (.pth) — can be same as det-ckpt")
    ap.add_argument("--seg-num-classes", type=int, default=7)
    ap.add_argument("--seg-feature-dim", type=int, default=384)
    ap.add_argument("--seg-dropout", type=float, default=0.1)
    ap.add_argument("--image-size", type=int, default=640)

    # Dataset
    ap.add_argument("--coco-root", required=True, help="Path to COCO dataset root (contains train2017/, annotations/)")

    # Training
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=5e-5, help="Learning rate for pose_decoder")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--val-every", type=int, default=3, help="Run COCO validation every N epochs")
    ap.add_argument("--amp", action="store_true", help="Use automatic mixed precision")
    ap.add_argument("--device", default="auto")

    # Phase 2: optional encoder unfreeze
    ap.add_argument("--unfreeze-encoder", action="store_true",
                    help="Also fine-tune encoder with low LR (phase 2)")
    ap.add_argument("--encoder-lr", type=float, default=5e-6,
                    help="LR for encoder when --unfreeze-encoder is set")

    # Output
    ap.add_argument("--out", required=True, help="Output checkpoint path")
    return ap.parse_args()


def build_merged_model(args):
    """Build SharedBackboneDualDecoder and load weights from separate checkpoints."""
    from src.core import YAMLConfig
    from tools.model_surgery.shared_arch import (
        SegmentationHead,
        SharedBackboneDualDecoder,
        filter_and_strip,
        infer_feature_dim_from_seg_ckpt,
        load_any_state,
        smart_load,
    )

    det_cfg = YAMLConfig(str(args.det_config))
    pose_cfg = YAMLConfig(str(args.pose_config))

    det_model = det_cfg.model
    pose_model = pose_cfg.model

    # Load det weights
    det_state_raw = load_any_state(args.det_ckpt)
    det_state = filter_and_strip("dfine_model.", det_state_raw) or det_state_raw
    smart_load(det_model, det_state, "det_model")

    # Load pose weights
    pose_state = load_any_state(args.pose_ckpt)
    smart_load(pose_model, pose_state, "pose_model")

    # Build seg_head
    det_model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, 3, args.image_size, args.image_size)
        feats = det_model.backbone(dummy)
    in_channels = [int(f.shape[1]) for f in feats]

    seg_state_raw = load_any_state(args.seg_ckpt)
    seg_prefixed = {k: v for k, v in seg_state_raw.items() if k.startswith("seg_head.")}
    feature_dim = infer_feature_dim_from_seg_ckpt(seg_prefixed, default=args.seg_feature_dim)

    seg_head = SegmentationHead(in_channels, args.seg_num_classes, feature_dim, args.seg_dropout)
    seg_state = filter_and_strip("seg_head.", seg_state_raw)
    smart_load(seg_head, seg_state, "seg_head")

    # Assemble merged model
    model = SharedBackboneDualDecoder(
        backbone=det_model.backbone,
        encoder=det_model.encoder,
        det_decoder=det_model.decoder,
        pose_decoder=pose_model.decoder,
        seg_head=seg_head,
    )
    return model


def freeze_module(module: nn.Module, name: str):
    """Freeze all parameters in a module."""
    for p in module.parameters():
        p.requires_grad = False
    module.eval()
    n = sum(p.numel() for p in module.parameters())
    print(f"  [frozen] {name}: {n:,} params")


def count_trainable(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def pose_collate_fn(batch):
    """Stack images, keep targets as list of dicts."""
    images = torch.stack([b[0] for b in batch], dim=0)
    targets = [b[1] for b in batch]
    return images, targets


def main():
    args = parse_args()
    os.chdir(REPO)

    # -----------------------------------------------------------------------
    # Device
    # -----------------------------------------------------------------------
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # -----------------------------------------------------------------------
    # Build merged model
    # -----------------------------------------------------------------------
    print("\n=== Building merged model ===")
    model = build_merged_model(args)

    # -----------------------------------------------------------------------
    # Freeze everything except pose_decoder (and optionally encoder)
    # -----------------------------------------------------------------------
    print("\n=== Freeze strategy ===")
    freeze_module(model.backbone, "backbone")
    freeze_module(model.det_decoder, "det_decoder")
    freeze_module(model.seg_head, "seg_head")

    if not args.unfreeze_encoder:
        freeze_module(model.encoder, "encoder")
    else:
        print(f"  [trainable] encoder: {sum(p.numel() for p in model.encoder.parameters()):,} params (lr={args.encoder_lr})")

    model.pose_decoder.train()
    print(f"  [trainable] pose_decoder: {count_trainable(model.pose_decoder):,} params (lr={args.lr})")
    print(f"  Total trainable: {count_trainable(model):,} params")

    model = model.to(device)

    # -----------------------------------------------------------------------
    # Dataset & DataLoader
    # -----------------------------------------------------------------------
    print("\n=== Loading COCO keypoints dataset ===")
    from pose_estimation_berna.core.datasets import CocoKeypointsDataset

    train_ds = CocoKeypointsDataset(
        root_dir=args.coco_root, split="train",
        image_size=args.image_size, flip_prob=0.5,
    )
    val_ds = CocoKeypointsDataset(
        root_dir=args.coco_root, split="val",
        image_size=args.image_size, flip_prob=0.0,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        collate_fn=pose_collate_fn, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        collate_fn=pose_collate_fn, drop_last=False,
    )

    # -----------------------------------------------------------------------
    # Criterion (DETRPose-style: pose-only, no boxes)
    # -----------------------------------------------------------------------
    from pose_estimation_berna.core.losses import create_pose_criterion_detrpose
    criterion = create_pose_criterion_detrpose(num_classes=2)
    criterion = criterion.to(device)

    # -----------------------------------------------------------------------
    # Optimizer — only trainable params
    # -----------------------------------------------------------------------
    param_groups = [{"params": list(model.pose_decoder.parameters()), "lr": args.lr}]
    if args.unfreeze_encoder:
        param_groups.append({"params": list(model.encoder.parameters()), "lr": args.encoder_lr})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # AMP
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    # -----------------------------------------------------------------------
    # COCO evaluator setup
    # -----------------------------------------------------------------------
    from pose_estimation_berna.core.postprocess_detrpose import DETRPosePostProcessor
    pose_post = DETRPosePostProcessor(num_classes=2, num_keypoints=17, num_top_queries=60, remap_mscoco_category=True)
    pose_post = pose_post.to(device).eval()

    coco_evaluator = None
    try:
        from src.data.dataset.coco_eval import CocoEvaluator
        val_ann_path = os.path.join(args.coco_root, "annotations", "person_keypoints_val2017.json")
        if os.path.isfile(val_ann_path):
            try:
                from faster_coco_eval import COCO as FCOCO
                coco_gt = FCOCO(val_ann_path)
            except ImportError:
                from pycocotools.coco import COCO as PyCOCO
                coco_gt = PyCOCO(val_ann_path)
            coco_evaluator = CocoEvaluator(coco_gt, ["keypoints"])
            print(f"COCO evaluator ready: {val_ann_path}")
    except Exception as e:
        print(f"COCO evaluator not available: {e}")

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ap = -1.0
    best_path = out_dir / "best_finetuned.pth"

    print(f"\n=== Training: {args.epochs} epochs, lr={args.lr}, batch_size={args.batch_size} ===\n")

    for epoch in range(args.epochs):
        model.pose_decoder.train()
        if args.unfreeze_encoder:
            model.encoder.train()

        epoch_loss = 0.0
        epoch_steps = 0
        t_epoch = time.time()

        for step, (images, targets) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            targets = [{k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                        for k, v in t.items()} for t in targets]

            with torch.amp.autocast("cuda", enabled=(args.amp and device.type == "cuda")):
                # Forward: backbone + encoder (frozen, no grads) → pose_decoder (trainable)
                with torch.no_grad():
                    feats = model.backbone(images)
                    if not args.unfreeze_encoder:
                        enc = model.encoder(feats)
                    else:
                        enc = None  # encoder gets gradients below

                if args.unfreeze_encoder:
                    enc = model.encoder(feats)

                pose_out = model.pose_decoder(enc, targets)

                # Criterion expects standard keys (pred_logits, pred_keypoints, aux_outputs)
                loss_dict = criterion(pose_out, targets)

            loss = sum(loss_dict.values())

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                trainable_params = list(model.pose_decoder.parameters())
                if args.unfreeze_encoder:
                    trainable_params += list(model.encoder.parameters())
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.detach())
            epoch_steps += 1

            if step % 100 == 0:
                avg = epoch_loss / max(1, epoch_steps)
                loss_parts = {k: f"{v.item():.4f}" for k, v in loss_dict.items()
                              if not k.endswith(("_0", "_1", "_2", "_3", "_4"))}
                print(f"  [epoch {epoch+1}/{args.epochs}] step {step}/{len(train_loader)} "
                      f"loss={avg:.4f} {loss_parts}")

        scheduler.step()

        elapsed = time.time() - t_epoch
        avg_loss = epoch_loss / max(1, epoch_steps)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f} lr={lr_now:.2e} time={elapsed:.0f}s")

        # -------------------------------------------------------------------
        # Save last checkpoint every epoch
        # -------------------------------------------------------------------
        save_dict = {
            "model": model.state_dict(),
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        }
        last_path = out_dir / "last_finetuned.pth"
        torch.save(save_dict, str(last_path))

        # -------------------------------------------------------------------
        # Validation
        # -------------------------------------------------------------------
        if (epoch + 1) % args.val_every == 0 or (epoch + 1) == args.epochs:
            print(f"\n  Validating (epoch {epoch+1})...")
            model.eval()

            val_loss = 0.0
            val_steps = 0
            coco_predictions = {}

            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(device, non_blocking=True)
                    targets = [{k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                                for k, v in t.items()} for t in targets]

                    feats = model.backbone(images)
                    enc = model.encoder(feats)
                    pose_out = model.pose_decoder(enc, targets)

                    loss_dict = criterion(pose_out, targets)
                    val_loss += float(sum(loss_dict.values()))
                    val_steps += 1

                    # COCO predictions
                    if coco_evaluator is not None:
                        orig_sizes = torch.stack([t["orig_size"] for t in targets]).to(device)
                        preds = pose_post(
                            {"pred_logits": pose_out["pred_logits"],
                             "pred_keypoints": pose_out["pred_keypoints"]},
                            orig_sizes,
                        )
                        for t, p in zip(targets, preds):
                            img_id = int(t["image_id"].item())
                            coco_predictions[img_id] = {
                                "boxes": p["boxes"].cpu(),
                                "scores": p["scores"].cpu(),
                                "labels": p["labels"].cpu(),
                                "keypoints": p["keypoints"].cpu(),
                            }

            avg_val_loss = val_loss / max(1, val_steps)
            print(f"  Val loss: {avg_val_loss:.4f}")

            # COCO AP
            val_ap = None
            if coco_evaluator is not None and coco_predictions:
                coco_evaluator.update(coco_predictions)
                coco_evaluator.synchronize_between_processes()
                coco_evaluator.accumulate()
                coco_evaluator.summarize()
                kpt_eval = coco_evaluator.coco_eval.get("keypoints")
                stats = getattr(kpt_eval, "stats", None) if kpt_eval is not None else None
                if stats is not None and len(stats) > 0:
                    val_ap = float(stats[0])
                    print(f"  COCO AP (keypoints): {val_ap:.4f}")
                # Reset for next eval
                coco_evaluator.cleanup()

            # Save best
            metric = val_ap if val_ap is not None else -avg_val_loss
            if metric > best_ap:
                best_ap = metric
                torch.save(save_dict, str(best_path))
                print(f"  New best! saved to {best_path}")

            print()

    # -----------------------------------------------------------------------
    # Save final merged checkpoint (compatible with infer_video_singlepass_x.py)
    # -----------------------------------------------------------------------
    final_path = Path(args.out)
    if not final_path.is_absolute():
        final_path = (REPO / final_path).resolve()
    final_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model": model.state_dict(),
            "meta": {
                "det_config": str(args.det_config),
                "pose_config": str(args.pose_config),
                "det_ckpt": str(args.det_ckpt),
                "pose_ckpt": str(args.pose_ckpt),
                "seg_ckpt": str(args.seg_ckpt),
                "seg_feature_dim": int(args.seg_feature_dim),
                "seg_num_classes": int(args.seg_num_classes),
                "seg_dropout": float(args.seg_dropout),
                "image_size": int(args.image_size),
                "finetuned_epochs": int(args.epochs),
                "finetuned_lr": float(args.lr),
                "unfroze_encoder": bool(args.unfreeze_encoder),
            },
        },
        str(final_path),
    )
    print(f"\n=== Done! Final checkpoint: {final_path} ===")
    print("Use with: python tools/model_surgery/infer_video_singlepass_x.py --merged-ckpt", str(final_path))


if __name__ == "__main__":
    main()