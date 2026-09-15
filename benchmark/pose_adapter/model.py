from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualFeatureAdapters(nn.Module):
    """Per-level 1x1 residual projections initialized as an exact identity."""

    def __init__(self, channels: list[int]):
        super().__init__()
        self.projections = nn.ModuleList([nn.Conv2d(c, c, 1, bias=True) for c in channels])
        for projection in self.projections:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(features) != len(self.projections):
            raise ValueError(f"expected {len(self.projections)} feature levels, got {len(features)}")
        return [feature + projection(feature) for feature, projection in zip(features, self.projections)]


class PoseAdapterModel(nn.Module):
    """Insert isolated adapters between a frozen shared encoder and the pose decoder."""

    def __init__(self, base_model: nn.Module, channels: list[int]):
        super().__init__()
        self.backbone = base_model.backbone
        self.encoder = base_model.encoder
        self.det_decoder = base_model.det_decoder
        self.pose_decoder = base_model.pose_decoder
        self.seg_head = base_model.seg_head
        self.pose_adapters = ResidualFeatureAdapters(channels)

    def freeze_protected_modules(self, train_pose_decoder: bool = False) -> None:
        for module in (self.backbone, self.encoder, self.det_decoder, self.seg_head):
            module.requires_grad_(False)
            module.eval()
        self.pose_adapters.requires_grad_(True)
        self.pose_adapters.train()
        self.pose_decoder.requires_grad_(train_pose_decoder)
        self.pose_decoder.train(train_pose_decoder)

    def forward(self, images: torch.Tensor, targets=None):
        features = self.backbone(images)
        encoded = self.encoder(features)
        detection = self.det_decoder(encoded, targets)
        pose = self.pose_decoder(self.pose_adapters(encoded), targets)
        segmentation = self.seg_head(features)
        segmentation = F.interpolate(segmentation, size=images.shape[-2:], mode="bilinear", align_corners=False)
        output = {"seg.logits": segmentation}
        output.update({f"det.{key}": value for key, value in detection.items()})
        output.update({f"pose.{key}": value for key, value in pose.items()})
        return output

    def pose_forward(self, images: torch.Tensor, targets=None):
        """Training path that avoids executing protected task heads."""
        with torch.no_grad():
            encoded = self.encoder(self.backbone(images))
        return self.pose_decoder(self.pose_adapters(encoded), targets)
