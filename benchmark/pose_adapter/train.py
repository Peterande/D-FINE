#!/usr/bin/env python3
"""Train isolated pose adapters while all shared production modules stay frozen."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from benchmark.harness.infer_crosshair import build_model_from_merged
from benchmark.pose_adapter.model import PoseAdapterModel


def digest_state(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def collate(batch):
    return torch.stack([item[0] for item in batch]), [item[1] for item in batch]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--det-config", required=True)
    parser.add_argument("--pose-config", required=True)
    parser.add_argument("--teacher")
    parser.add_argument("--rtmo-pseudo", type=Path,
                        help="COCO-style RTMO predictions used as pose targets instead of GT.")
    parser.add_argument("--coco-root", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--distill-weight", type=float, default=0.0)
    parser.add_argument("--distill-matching", choices=["query_index", "hungarian_l1"], default="query_index")
    parser.add_argument("--train-pose-decoder", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if not args.rtmo_pseudo and not args.teacher:
        parser.error("--teacher is required unless --rtmo-pseudo is supplied")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda")

    base, _ = build_model_from_merged(det_config=args.det_config, pose_config=args.pose_config,
        merged_ckpt=args.baseline, seg_num_classes=7, seg_feature_dim=384, seg_dropout=0.1, image_size=640)
    student = PoseAdapterModel(base, [384, 384, 384])
    student.freeze_protected_modules(args.train_pose_decoder)
    protected = {name: digest_state(getattr(student, name)) for name in ("backbone", "encoder", "det_decoder", "seg_head")}

    teacher = None
    if not args.rtmo_pseudo:
        from src.core import YAMLConfig
        teacher_cfg = YAMLConfig(args.pose_config); teacher = teacher_cfg.model
        checkpoint = torch.load(args.teacher, map_location="cpu", weights_only=False)
        teacher.load_state_dict(checkpoint["model"], strict=True)
        teacher.requires_grad_(False).eval().to(device)
    student.to(device)

    from pose_estimation_berna.core.datasets import CocoKeypointsDataset
    from pose_estimation_berna.core.losses import create_pose_criterion_detrpose
    dataset = CocoKeypointsDataset(root_dir=args.coco_root, split="train", image_size=640, flip_prob=0.0)
    if args.rtmo_pseudo:
        pseudo_by_image = {}
        pseudo_data = json.loads(args.rtmo_pseudo.read_text())
        predictions = pseudo_data.get("predictions", []) if isinstance(pseudo_data, dict) else pseudo_data
        for prediction in predictions:
            pseudo_by_image.setdefault(int(prediction["image_id"]), []).append(prediction)
        dataset.ids = [image_id for image_id in dataset.ids if image_id in pseudo_by_image]
        base_dataset = dataset
        original_getitem = base_dataset.__class__.__getitem__

        class RTMOPseudoDataset(torch.utils.data.Dataset):
            def __len__(self):
                return len(base_dataset)

            def __getitem__(self, index):
                image, target = original_getitem(base_dataset, index)
                image_id = int(target["image_id"].item())
                orig_w, orig_h = target["orig_size"].tolist()
                poses, boxes = [], []
                for prediction in pseudo_by_image[image_id]:
                    pose = np.asarray(prediction["keypoints"], dtype=np.float32).reshape(17, 3)
                    pose[:, 0] *= 640.0 / orig_w
                    pose[:, 1] *= 640.0 / orig_h
                    visible = pose[:, 2] >= 0.20
                    if visible.sum() < 4:
                        continue
                    xy = pose[visible, :2]
                    x1, y1 = xy.min(0); x2, y2 = xy.max(0)
                    boxes.append([(x1 + x2) / 1280.0, (y1 + y2) / 1280.0,
                                  (x2 - x1) / 640.0, (y2 - y1) / 640.0])
                    pose[:, 2] = np.where(visible, 2.0, 0.0)
                    poses.append(pose)
                if not poses:
                    return self[(index + 1) % len(self)]
                target["boxes"] = torch.tensor(boxes, dtype=torch.float32)
                target["labels"] = torch.zeros(len(poses), dtype=torch.int64)
                target["keypoints"] = torch.tensor(np.stack(poses), dtype=torch.float32)
                return image, target

        dataset = RTMOPseudoDataset()
    generator = torch.Generator().manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
        generator=generator, num_workers=0, drop_last=True, collate_fn=collate)
    criterion = create_pose_criterion_detrpose(num_classes=2).to(device)
    parameters = list(student.pose_adapters.parameters())
    if args.train_pose_decoder: parameters += list(student.pose_decoder.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-4)
    start = 0
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        student.pose_adapters.load_state_dict(saved["pose_adapters"])
        if args.train_pose_decoder: student.pose_decoder.load_state_dict(saved["pose_decoder"])
        optimizer.load_state_dict(saved["optimizer"]); start = int(saved["step"])

    losses = []
    iterator = iter(loader)
    for step in range(start, args.steps):
        try:
            images, targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, targets = next(iterator)
        images = images.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in target.items()} for target in targets]
        student_out = student.pose_forward(images, targets)
        supervised = sum(criterion(student_out, targets).values())
        if args.rtmo_pseudo:
            distill = torch.zeros((), device=device)
        else:
            with torch.no_grad(): teacher_out = teacher(images, targets)
            if args.distill_matching == "hungarian_l1":
                from benchmark.pose_adapter.matching import matched_distillation_loss
                distill = matched_distillation_loss(student_out, teacher_out)
            else:
                distill = torch.nn.functional.smooth_l1_loss(student_out["pred_logits"], teacher_out["pred_logits"]) + torch.nn.functional.smooth_l1_loss(student_out["pred_keypoints"], teacher_out["pred_keypoints"])
        loss = supervised + args.distill_weight * distill
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(parameters, 1.0); optimizer.step()
        losses.append({"step": step + 1, "total": float(loss.detach()), "supervised": float(supervised.detach()), "distill": float(distill.detach())})
        if (step + 1) % 50 == 0 or step + 1 == args.steps:
            print(json.dumps(losses[-1]), flush=True)

    after = {name: digest_state(getattr(student, name)) for name in protected}
    assert protected == after, "protected module changed"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"pose_adapters": student.pose_adapters.state_dict(), "pose_decoder": student.pose_decoder.state_dict(),
        "optimizer": optimizer.state_dict(), "step": args.steps, "args": vars(args), "protected_hashes": protected, "losses": losses}, args.out)
    print(json.dumps({"out": str(args.out), "steps": args.steps, "losses": losses, "protected_unchanged": True}, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
