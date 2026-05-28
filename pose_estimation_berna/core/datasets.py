#!/usr/bin/env python3
"""
COCO Keypoints Dataset for D-FINE Pose Estimation (COCO-17).

This supports the **per-detection / per-query** approach:
- D-FINE predicts boxes per query
- We predict 17 keypoints per query, **bbox-relative**: [B, Q, 17, 3]
  where 3 = (x_rel, y_rel, visibility_logit)

Targets returned:
- boxes: [N, 4] (cxcywh normalized to [0,1])
- labels: [N] (COCO contiguous label, person is 0)
- keypoints: [N, 17, 3] (x_px, y_px, v) in resized image pixels, v in {0,1,2}
- size: [2] (h, w) in resized image pixels
- orig_size: [2] (w, h) in original image pixels
"""

import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


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


@dataclass
class CocoImageRecord:
    id: int
    file_name: str
    width: int
    height: int


class CocoKeypointsDataset(Dataset):
    def __init__(
        self,
                 root_dir: str,
        split: str = "train",
                 image_size: int = 640,
        num_keypoints: int = 17,
        flip_prob: float = 0.5,
        return_dict: bool = False,
    ):
        assert split in ("train", "val"), "split must be 'train' or 'val'"
        self.root_dir = root_dir
        self.split = split
        self.image_size = int(image_size)
        self.num_keypoints = int(num_keypoints)
        self.flip_prob = float(flip_prob if split == "train" else 0.0)
        self.return_dict = bool(return_dict)

        self.img_dir = os.path.join(root_dir, f"{split}2017")
        self.ann_path = os.path.join(
            root_dir, "annotations", f"person_keypoints_{split}2017.json"
        )

        with open(self.ann_path, "r") as f:
            coco = json.load(f)

        self.images: Dict[int, CocoImageRecord] = {
            img["id"]: CocoImageRecord(
                id=img["id"],
                file_name=img["file_name"],
                width=img["width"],
                height=img["height"],
            )
            for img in coco["images"]
        }

        anns_by_img: Dict[int, List[dict]] = {}
        for ann in coco["annotations"]:
            if ann.get("iscrowd", 0) == 1:
                continue
            if ann.get("category_id") != 1:  # person
                continue
            if "keypoints" not in ann:
                continue
            if int(ann.get("num_keypoints", 0)) <= 0:
                continue
            anns_by_img.setdefault(ann["image_id"], []).append(ann)

        self.ids: List[int] = [img_id for img_id in anns_by_img.keys() if img_id in self.images]
        self.anns_by_img = anns_by_img

        # ImageNet normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

        print(f"📊 Loaded COCO-{split} keypoints: {len(self.ids)} images from {self.ann_path}")
    
    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _xywh_to_xyxy(box: np.ndarray) -> np.ndarray:
        x, y, w, h = box.tolist()
        return np.array([x, y, x + w, y + h], dtype=np.float32)

    @staticmethod
    def _xyxy_to_cxcywh(box: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = box.tolist()
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        w = (x2 - x1)
        h = (y2 - y1)
        return np.array([cx, cy, w, h], dtype=np.float32)
    
    def __getitem__(self, idx: int):
        image_id = self.ids[idx]
        rec = self.images[image_id]

        img_path = os.path.join(self.img_dir, rec.file_name)
        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                orig_w, orig_h = im.size
                scale_x = self.image_size / float(orig_w)
                scale_y = self.image_size / float(orig_h)
                im = im.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
                image = np.array(im, dtype=np.uint8)
        except Exception as e:
            raise FileNotFoundError(f"Could not read image: {img_path} ({e})") from e

        anns = self.anns_by_img.get(image_id, [])
        boxes_xyxy = []
        kpts = []

        for ann in anns:
            bbox_xywh = np.array(ann["bbox"], dtype=np.float32)
            bbox_xywh[[0, 2]] *= scale_x
            bbox_xywh[[1, 3]] *= scale_y
            box_xyxy = self._xywh_to_xyxy(bbox_xywh)

            box_xyxy[0::2] = np.clip(box_xyxy[0::2], 0, self.image_size - 1)
            box_xyxy[1::2] = np.clip(box_xyxy[1::2], 0, self.image_size - 1)

            kp = np.array(ann["keypoints"], dtype=np.float32).reshape(-1, 3)
            kp[:, 0] *= scale_x
            kp[:, 1] *= scale_y

            boxes_xyxy.append(box_xyxy)
            kpts.append(kp)

        boxes_xyxy = np.stack(boxes_xyxy, axis=0).astype(np.float32)  # [N,4]
        kpts = np.stack(kpts, axis=0).astype(np.float32)  # [N,17,3]

        if self.flip_prob > 0 and random.random() < self.flip_prob:
            image = np.ascontiguousarray(image[:, ::-1, :])

            x1 = boxes_xyxy[:, 0].copy()
            x2 = boxes_xyxy[:, 2].copy()
            boxes_xyxy[:, 0] = (self.image_size - 1) - x2
            boxes_xyxy[:, 2] = (self.image_size - 1) - x1

            kpts[:, :, 0] = (self.image_size - 1) - kpts[:, :, 0]
            kpts = kpts[:, COCO_KEYPOINT_FLIP_INDEX, :]

        # boxes to normalized cxcywh
        boxes_cxcywh = np.stack([self._xyxy_to_cxcywh(b) for b in boxes_xyxy], axis=0)
        boxes_cxcywh[:, 0] /= self.image_size
        boxes_cxcywh[:, 1] /= self.image_size
        boxes_cxcywh[:, 2] /= self.image_size
        boxes_cxcywh[:, 3] /= self.image_size
        boxes_cxcywh = np.clip(boxes_cxcywh, 0.0, 1.0)

        image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_t = (image_t - self.mean[:, None, None]) / self.std[:, None, None]

        target = {
            "boxes": torch.from_numpy(boxes_cxcywh).float(),
            "labels": torch.zeros((boxes_cxcywh.shape[0],), dtype=torch.int64),
            "keypoints": torch.from_numpy(kpts).float(),
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "orig_size": torch.tensor([orig_w, orig_h], dtype=torch.int64),
            "size": torch.tensor([self.image_size, self.image_size], dtype=torch.int64),
        }

        if self.return_dict:
            return {"image": image_t, **target}

        return image_t, target


def create_coco_pose_dataset(
                  root_dir: str, 
    split: str = "train",
    image_size: int = 640,
    num_keypoints: int = 17,
    tier: str = "standard",
    return_dict: bool = False,
):
    # tier can drive aug strength later; for now only flip in train
    flip_prob = 0.5 if split == "train" else 0.0
    return CocoKeypointsDataset(
            root_dir=root_dir,
            split=split,
            image_size=image_size,
        num_keypoints=num_keypoints,
        flip_prob=flip_prob,
        return_dict=return_dict,
    )


