"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys

from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import openvino as ov
import nncf
import torch
from torchvision import datasets, transforms
import numpy as np


def main(
    args,
):
    """main"""
    print("Start OpenVINO model convertion...")
    if not args.dynamic and args.batch_size == 1:
        inp = [1, 3, args.img_size[0], args.img_size[1]]
    elif args.batch_size > 1 and args.dynamic:
        inp = [-1, 3, -1, -1]
    elif args.batch_size > 1:
        inp = [-1, 3, args.img_size[0], args.img_size[1]]
    elif args.dynamic:
        inp = [1, 3, -1, -1]

    model = ov.convert_model(
        input_model=str(args.model_path),
        input=inp,
    )
    ov.serialize(model, str(Path(args.model_path).with_suffix(".xml")), str(Path(args.model_path).with_suffix(".bin")))
    print("OpenVINO model exported")

    if args.quantize_nncf :
        print("Start NNCF quantization...")

        # Instantiate your uncompressed model
        model = ov.Core().read_model(str(Path(args.model_path).with_suffix(".xml")))

        # # Provide validation part of the dataset to collect statistics needed for the compression algorithm
        val_dataset = datasets.ImageFolder(args.calibration_dataset_path, transform=transforms.Compose([transforms.ToTensor()]))
        dataset_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1)

        # Step 1: Initialize transformation function
        def transform_fn(data_item):
            images, _ = data_item
            orig_size = np.array([args.img_size[0], args.img_size[0]], dtype=np.int64).reshape(
            1, 2
            )
            inputs = {
                "images": np.array(images),
                "orig_target_sizes": orig_size,
            }
            return inputs

        # Step 2: Initialize NNCF Dataset
        calibration_dataset = nncf.Dataset(dataset_loader, transform_fn)
        # Step 3: Run the quantization pipeline
        quantized_model = nncf.quantize(model, calibration_dataset, preset=nncf.QuantizationPreset.MIXED)
        # Step 4: Save model
        ov.save_model(quantized_model,str(Path(args.model_path.replace(".onnx","_int8.xml"))))
        print("OpenVINO model quantized")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        "-m",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dynamic",
        "-d",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--img_size",
        "-i",
        default=(640,640),
        type=tuple,
    )
    parser.add_argument(
        "--quantize_nncf",
        "-q",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "--calibration_dataset_path",
        "-c",
        type=str,
    )
    args = parser.parse_args()
    main(args)
