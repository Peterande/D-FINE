#!/usr/bin/env python3
"""Low-light fine-tuning of only the ModelSurgery segmentation head."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from benchmark.harness.infer_crosshair import build_model_from_merged
from segmentation_sivert.core.datasets import PascalPersonPartsDataset


def digest(module: torch.nn.Module) -> str:
    result = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        result.update(name.encode()); result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def photometric(image: np.ndarray, rng: random.Random) -> tuple[np.ndarray, str]:
    value = image.astype(np.float32)
    choice = rng.random()
    if choice < 0.50:
        return image, "normal"
    if choice < 0.65:
        return np.clip(value / 3.0, 0, 255).astype(np.uint8), "dark_3x"
    if choice < 0.80:
        return np.clip(value / 10.0, 0, 255).astype(np.uint8), "dark_10x"
    if choice < 0.90:
        noise = np.random.normal(0.0, 8.0, value.shape)
        return np.clip(value / 10.0 + noise, 0, 255).astype(np.uint8), "dark_10x_noise"
    cast = np.asarray([1.22, 0.95, 0.72] if rng.random() < 0.5 else [0.72, 0.95, 1.22])
    return np.clip(value * cast, 0, 255).astype(np.uint8), "color_cast"


class LowLightDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, seed: int):
        self.base = PascalPersonPartsDataset(root_dir=root, split="train", image_size=640,
                                             num_classes=7, tier="standard")
        self.seed = seed

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, mask = self.base._load_image_and_mask(index)
        rng = random.Random(self.seed * 1_000_003 + index + random.randint(0, 2**20))
        if rng.random() < 0.5:
            image, mask = image[:, ::-1].copy(), mask[:, ::-1].copy()
        image, condition = photometric(image, rng)
        return self.base.preprocess_image(image), self.base.preprocess_mask(mask), condition


def weighted_dice(logits: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    probabilities = logits.softmax(1)
    one_hot = F.one_hot(target.clamp(0, 6), 7).permute(0, 3, 1, 2).float()
    intersection = (probabilities * one_hot).sum((0, 2, 3))
    denominator = probabilities.sum((0, 2, 3)) + one_hot.sum((0, 2, 3))
    loss = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    return (loss * weights).sum() / weights.sum()


def evaluate(model, dataset, device, darkness: float) -> tuple[float, list[float]]:
    confusion = np.zeros((7, 7), dtype=np.int64)
    with torch.inference_mode():
        for start in range(0, len(dataset), 8):
            batch = []
            masks = []
            for index in range(start, min(start + 8, len(dataset))):
                image, mask = dataset._load_image_and_mask(index)
                if darkness != 1.0:
                    image = np.clip(image.astype(np.float32) * darkness, 0, 255).astype(np.uint8)
                batch.append(dataset.preprocess_image(image)); masks.append(dataset.preprocess_mask(mask))
            images = torch.stack(batch).to(device)
            targets = torch.stack(masks).numpy().reshape(-1)
            features = model.backbone(images)
            logits = model.seg_head(features)
            logits = F.interpolate(logits, (640, 640), mode="bilinear", align_corners=False)
            predictions = logits.argmax(1).cpu().numpy().reshape(-1)
            confusion += np.bincount(7 * targets + predictions, minlength=49).reshape(7, 7)
    intersection = np.diag(confusion)
    union = confusion.sum(0) + confusion.sum(1) - intersection
    iou = intersection / np.maximum(union, 1)
    return float(iou.mean()), iou.tolist()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--det-config", required=True)
    parser.add_argument("--pose-config", required=True)
    parser.add_argument("--merged-ckpt", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda")
    model, _ = build_model_from_merged(args.det_config, args.pose_config, args.merged_ckpt,
                                       7, 384, 0.1, 640)
    protected_names = ("backbone", "encoder", "det_decoder", "pose_decoder")
    protected = {name: digest(getattr(model, name)) for name in protected_names}
    for name in protected_names:
        getattr(model, name).requires_grad_(False).eval()
    model.seg_head.requires_grad_(True).train()
    model.to(device)

    train = LowLightDataset(args.dataset, args.seed)
    val = PascalPersonPartsDataset(root_dir=args.dataset, split="val", image_size=640,
                                   num_classes=7, tier="standard")
    loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True,
                                         num_workers=4, pin_memory=True, drop_last=True)
    weights = torch.tensor([0.35, 1.0, 1.0, 1.25, 2.5, 1.0, 2.5], device=device)
    optimizer = torch.optim.AdamW(model.seg_head.parameters(), lr=args.lr, weight_decay=1e-4)
    history, best = [], None
    for epoch in range(1, args.epochs + 1):
        model.seg_head.train(); losses = []; counts = {}
        for images, targets, conditions in loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            with torch.no_grad():
                features = model.backbone(images)
            logits = model.seg_head(features)
            logits = F.interpolate(logits, targets.shape[-2:], mode="bilinear", align_corners=False)
            loss = F.cross_entropy(logits, targets, weight=weights) + 0.6 * weighted_dice(logits, targets, weights)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.seg_head.parameters(), 1.0); optimizer.step()
            losses.append(float(loss.detach()))
            for condition in conditions: counts[condition] = counts.get(condition, 0) + 1
        model.seg_head.eval()
        normal, normal_classes = evaluate(model, val, device, 1.0)
        dark, dark_classes = evaluate(model, val, device, 0.1)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "normal_miou": normal,
               "dark_10x_miou": dark, "normal_per_class": normal_classes,
               "dark_10x_per_class": dark_classes, "conditions": counts}
        history.append(row); print(json.dumps(row), flush=True)
        score = dark if normal >= 0.731 else -1.0
        if best is None or score > best[0]:
            best = (score, epoch, {k: v.detach().cpu() for k, v in model.seg_head.state_dict().items()}, row)

    after = {name: digest(getattr(model, name)) for name in protected_names}
    assert protected == after, "protected module changed"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"seg_head": best[2], "best_epoch": best[1], "metrics": best[3],
                "history": history, "protected_hashes": protected, "protected_unchanged": True,
                "args": vars(args)}, args.out)
    print(json.dumps({"out": str(args.out), "best_epoch": best[1], "metrics": best[3],
                      "protected_unchanged": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
