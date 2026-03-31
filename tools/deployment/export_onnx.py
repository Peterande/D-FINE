"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import torch
import torch.nn as nn

from src.core import YAMLConfig, yaml_utils


def main(args):
    """main"""
    update_dict = yaml_utils.parse_cli(args.update) if args.update else {}
    cfg = YAMLConfig(args.config, resume=args.resume, **update_dict)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]

        cfg.model.load_state_dict(state)
    else:
        print("not load model.state_dict, use default init state dict...")

    img_size = cfg.yaml_cfg["eval_spatial_size"]

    class Model(nn.Module):
        def __init__(self, img_size) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            self.register_buffer("orig_target_sizes", torch.tensor([img_size], dtype=torch.int64))

        def forward(self, images):
            # Input: NHWC uint8-range float32 [0, 255] → NCHW float32 [0, 1]
            images = images.permute(0, 3, 1, 2)
            images = images / 255.0
            outputs = self.model(images)
            orig_target_sizes = self.orig_target_sizes.expand(images.shape[0], -1)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return tuple(o.to(torch.float32) for o in outputs)

    model = Model(img_size)

    # Input: NHWC float32, values in [0, 255]
    data = torch.randint(0, 256, (1, *img_size, 3), dtype=torch.float32)
    _ = model(data)

    dynamic_axes = {"images": {0: "N"}, "labels": {0: "N"}, "boxes": {0: "N"}, "scores": {0: "N"}}

    import os
    output_file = os.path.splitext(args.resume)[0] + ".onnx" if args.resume else "model.onnx"

    torch.onnx.export(
        model,
        (data,),
        output_file,
        input_names=["images"],
        output_names=["labels", "boxes", "scores"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        verbose=False,
        do_constant_folding=True,
    )

    if args.check:
        import onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        print("Check export onnx model done...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="configs/dfine/dfine_hgnetv2_l_coco.yml", type=str)
    parser.add_argument("--resume", "-r", type=str)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--check", action="store_true", default=True)
    parser.add_argument("-u", "--update", nargs="+", help="update yaml config")
    args = parser.parse_args()
    main(args)
