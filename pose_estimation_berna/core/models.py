#!/usr/bin/env python3
"""
Core model helpers for D-FINE pose estimation.

Pose is implemented as a per-query keypoint branch inside `DFINETransformer`:
- outputs["pred_boxes"]: [B,Q,4] (cxcywh normalized)
- outputs["pred_keypoints"]: [B,Q,17,3] (x_rel,y_rel,vis_logit), bbox-relative
"""

import os
import sys
from typing import List, Tuple

import torch
import torch.nn as nn

# Ensure `src/` imports work when running from repo root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
src_path = os.path.join(project_root, "src")
if os.path.exists(src_path) and src_path not in sys.path:
    sys.path.insert(0, src_path)
    sys.path.insert(0, project_root)


def load_pretrained_dfine(config_path: str, checkpoint_path: str) -> nn.Module:
    """Load DFINE model from YAMLConfig + checkpoint."""
    from src.core import YAMLConfig

    cfg = YAMLConfig(config_path)
    model = cfg.model

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "ema" in checkpoint and "module" in checkpoint.get("ema", {}):
        state_dict = checkpoint["ema"]["module"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    return model


def freeze_except_keypoints(model: nn.Module) -> None:
    """Freeze all params except pose-related decoder heads (keypoints + optional Pose-LQE)."""
    for p in model.parameters():
        # Only floating/complex params can require gradients.
        if p.is_floating_point() or p.is_complex():
            p.requires_grad = False

    dec = getattr(model, "decoder", None)
    if dec is None:
        return
    kpt_head = getattr(dec, "dec_keypoint_head", None)
    if kpt_head is None:
        return
    for p in kpt_head.parameters():
        if p.is_floating_point() or p.is_complex():
            p.requires_grad = True

    pose_q = getattr(dec, "dec_pose_lqe_head", None)
    if pose_q is not None:
        for p in pose_q.parameters():
            if p.is_floating_point() or p.is_complex():
                p.requires_grad = True

    dn_kpt_proj = getattr(dec, "denoising_kpt_proj", None)
    if dn_kpt_proj is not None:
        for p in dn_kpt_proj.parameters():
            if p.is_floating_point() or p.is_complex():
                p.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze all parameters."""
    for p in model.parameters():
        # Only floating/complex params can require gradients (PyTorch constraint).
        if p.is_floating_point() or p.is_complex():
            p.requires_grad = True


def set_trainable_by_name(model: nn.Module, name_substrings: List[str], trainable: bool) -> int:
    """
    Set requires_grad for parameters whose name contains any of the provided substrings.
    Returns number of parameters affected.
    """
    affected = 0
    for n, p in model.named_parameters():
        if any(s in n for s in name_substrings):
            if p.is_floating_point() or p.is_complex():
                p.requires_grad = trainable
            affected += 1
    return affected


def freeze_backbone(model: nn.Module) -> None:
    """Freeze backbone parameters (if present)."""
    affected = set_trainable_by_name(model, ["backbone."], trainable=False)
    _ = affected


def unfreeze_backbone_stages(model: nn.Module, stage_indices: List[int]) -> int:
    """
    Unfreeze specific HGNetv2 backbone stages by index (e.g. [3] for stage4).
    Works by matching parameter names containing 'backbone.stages.{idx}'.
    """
    subs = [f"backbone.stages.{i}." for i in stage_indices]
    return set_trainable_by_name(model, subs, trainable=True)


def build_param_groups(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    backbone_lr_factor: float = 0.1,
    keypoint_lr_factor: float = 1.0,
):
    """
    Create AdamW param groups with different LRs:
    - backbone.* uses base_lr * backbone_lr_factor
    - decoder.dec_keypoint_head.* uses base_lr * keypoint_lr_factor
    - everything else uses base_lr
    Only includes params with requires_grad=True.
    """
    backbone_params = []
    keypoint_params = []
    other_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_params.append(p)
        elif (
            ("decoder.dec_keypoint_head" in name)
            or ("decoder.dec_pose_lqe_head" in name)
            or ("decoder.denoising_kpt_proj" in name)
        ):
            keypoint_params.append(p)
        else:
            other_params.append(p)

    groups = []
    if other_params:
        groups.append({"params": other_params, "lr": base_lr, "weight_decay": weight_decay})
    if keypoint_params:
        groups.append(
            {
                "params": keypoint_params,
                "lr": base_lr * float(keypoint_lr_factor),
                "weight_decay": weight_decay,
            }
        )
    if backbone_params:
        groups.append(
            {
                "params": backbone_params,
                "lr": base_lr * float(backbone_lr_factor),
                "weight_decay": weight_decay,
            }
        )
    return groups


def get_trainable_params(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def get_device(device: str = "auto"):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def get_actual_backbone_channels(model: nn.Module, input_size: Tuple[int, int] = (640, 640)) -> List[int]:
    """Utility: inspect backbone output channel sizes."""
    model.eval()
    dummy_input = torch.randn(1, 3, *input_size)
    with torch.no_grad():
        feats = model.backbone(dummy_input)
    return [f.shape[1] for f in feats]


