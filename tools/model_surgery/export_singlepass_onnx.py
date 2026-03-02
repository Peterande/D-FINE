#!/usr/bin/env python3
"""Export model-surgery single-pass checkpoint to ONNX.

This exporter targets the merged SharedBackboneDualDecoder checkpoint produced by:
  tools/model_surgery/finetune_pose_decoder.py

Outputs are raw head tensors with stable names:
  - det_pred_logits
  - det_pred_boxes
  - pose_pred_logits
  - pose_pred_keypoints
  - seg_logits
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn


REPO = Path(__file__).resolve().parents[2]
for p in [REPO, REPO / "src", REPO / "tools", REPO / "segmentation_sivert", REPO / "pose_estimation_berna"]:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from src.core import YAMLConfig
from tools.model_surgery.shared_arch import SegmentationHead, SharedBackboneDualDecoder, load_any_state


def parse_args():
    ap = argparse.ArgumentParser(description="Export merged single-pass checkpoint to ONNX")
    ap.add_argument("--det-config", required=True, help="Det/seg YAML config")
    ap.add_argument("--pose-config", required=True, help="Pose YAML config")
    ap.add_argument("--merged-ckpt", required=True, help="Merged checkpoint (.pth)")
    ap.add_argument("--out", required=True, help="Output ONNX path")
    ap.add_argument("--image-size", type=int, default=640, help="Export input size (must match deployment shape)")
    ap.add_argument("--seg-num-classes", type=int, default=7)
    ap.add_argument("--seg-feature-dim", type=int, default=384)
    ap.add_argument("--seg-dropout", type=float, default=0.1)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--check", action="store_true", help="Validate exported ONNX with onnx.checker")
    ap.add_argument("--simplify", action="store_true", help="Run onnxsim simplification")
    ap.add_argument("--dynamic-batch", action="store_true", help="Export with dynamic batch axis")
    ap.add_argument(
        "--use-dynamo",
        action="store_true",
        help="Use new torch.export-based ONNX path (default: off, uses legacy exporter for compatibility).",
    )
    return ap.parse_args()


def _disable_hgnet_pretrained(cfg: YAMLConfig):
    # Prevent remote weight downloads during export.
    if hasattr(cfg, "yaml_cfg") and isinstance(cfg.yaml_cfg, dict) and "HGNetv2" in cfg.yaml_cfg:
        node = cfg.yaml_cfg["HGNetv2"]
        if isinstance(node, dict):
            node["pretrained"] = False


def build_model_from_merged_ckpt(args) -> nn.Module:
    det_cfg = YAMLConfig(str(args.det_config))
    pose_cfg = YAMLConfig(str(args.pose_config))
    _disable_hgnet_pretrained(det_cfg)
    _disable_hgnet_pretrained(pose_cfg)

    det_model = det_cfg.model
    pose_model = pose_cfg.model

    # Infer feature channels for seg head from backbone outputs.
    det_model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, 3, int(args.image_size), int(args.image_size))
        feats = det_model.backbone(dummy)
    in_channels = [int(f.shape[1]) for f in feats]

    # Infer seg feature dim from checkpoint if available.
    ckpt_state = load_any_state(str(args.merged_ckpt))
    fpn_key = "seg_head.fpn.lateral_convs.0.weight"
    feature_dim = int(ckpt_state[fpn_key].shape[0]) if fpn_key in ckpt_state else int(args.seg_feature_dim)

    seg_head = SegmentationHead(in_channels, int(args.seg_num_classes), int(feature_dim), float(args.seg_dropout))
    model = SharedBackboneDualDecoder(
        backbone=det_model.backbone,
        encoder=det_model.encoder,
        det_decoder=det_model.decoder,
        pose_decoder=pose_model.decoder,
        seg_head=seg_head,
    )

    missing, unexpected = model.load_state_dict(ckpt_state, strict=False)
    print(f"[merged-load] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"  missing (first 10): {missing[:10]}")
    if unexpected:
        print(f"  unexpected (first 10): {unexpected[:10]}")

    return model.eval()


class ExportWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images):
        out = self.model(images)
        return (
            out["det.pred_logits"],
            out["det.pred_boxes"],
            out["pose.pred_logits"],
            out["pose.pred_keypoints"],
            out["seg.logits"],
        )


def main():
    args = parse_args()
    os.chdir(REPO)

    model = build_model_from_merged_ckpt(args)
    wrapper = ExportWrapper(model).eval()

    dummy = torch.randn(1, 3, int(args.image_size), int(args.image_size))
    with torch.inference_mode():
        _ = wrapper(dummy)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (REPO / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {
            "images": {0: "N"},
            "det_pred_logits": {0: "N"},
            "det_pred_boxes": {0: "N"},
            "pose_pred_logits": {0: "N"},
            "pose_pred_keypoints": {0: "N"},
            "seg_logits": {0: "N"},
        }

    print(f"[export] writing ONNX to: {out_path}")
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(out_path),
        input_names=["images"],
        output_names=[
            "det_pred_logits",
            "det_pred_boxes",
            "pose_pred_logits",
            "pose_pred_keypoints",
            "seg_logits",
        ],
        dynamic_axes=dynamic_axes,
        opset_version=int(args.opset),
        do_constant_folding=True,
        verbose=False,
        dynamo=bool(args.use_dynamo),
    )

    if args.check:
        import onnx

        onnx_model = onnx.load(str(out_path))
        onnx.checker.check_model(onnx_model)
        print("[export] onnx.checker: OK")

    if args.simplify:
        import onnx
        import onnxsim

        input_shapes = {"images": tuple(dummy.shape)}
        onnx_model_simplify, ok = onnxsim.simplify(str(out_path), test_input_shapes=input_shapes)
        onnx.save(onnx_model_simplify, str(out_path))
        print(f"[export] onnxsim simplify: {ok}")

    print("[done] ONNX export complete")


if __name__ == "__main__":
    main()
