""" "
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import PIL
import numpy as np
import torch
import torch.utils.data
import torchvision
from typing import List, Dict

torchvision.disable_beta_transforms_warning()

__all__ = ["show_sample", "save_samples"]

def save_samples(samples: torch.Tensor, targets: List[Dict], output_dir: str, split: str, normalized: bool, box_fmt: str):
    '''
    normalized: whether the boxes are normalized to [0, 1]
    box_fmt: 'xyxy', 'xywh', 'cxcywh', D-FINE uses 'cxcywh' for training, 'xyxy' for validation
    '''
    from torchvision.transforms.functional import to_pil_image
    from torchvision.ops import box_convert
    from pathlib import Path
    from PIL import ImageDraw, ImageFont
    import os

    os.makedirs(Path(output_dir) / Path(f"{split}_samples"), exist_ok=True)
    # Predefined colors (standard color names recognized by PIL)
    BOX_COLORS = [
        "red", "blue", "green", "orange", "purple",
        "cyan", "magenta", "yellow", "lime", "pink",
        "teal", "lavender", "brown", "beige", "maroon",
        "navy", "olive", "coral", "turquoise", "gold"
    ]

    LABEL_TEXT_COLOR = "white"

    font = ImageFont.load_default()
    font.size = 32

    for i, (sample, target) in enumerate(zip(samples, targets)):
        sample_visualization = sample.clone().cpu()
        target_boxes = target["boxes"].clone().cpu()
        target_labels = target["labels"].clone().cpu()
        target_image_id = target["image_id"].item()
        target_image_path = target["image_path"]
        target_image_path_stem = Path(target_image_path).stem

        sample_visualization = to_pil_image(sample_visualization)
        sample_visualization_w, sample_visualization_h = sample_visualization.size

        # normalized to pixel space
        if normalized:
            target_boxes[:, 0] = target_boxes[:, 0] * sample_visualization_w
            target_boxes[:, 2] = target_boxes[:, 2] * sample_visualization_w
            target_boxes[:, 1] = target_boxes[:, 1] * sample_visualization_h
            target_boxes[:, 3] = target_boxes[:, 3] * sample_visualization_h

        # any box format -> xyxy
        target_boxes = box_convert(target_boxes, in_fmt=box_fmt, out_fmt="xyxy")

        # clip to image size
        target_boxes[:, 0] = torch.clamp(target_boxes[:, 0], 0, sample_visualization_w)
        target_boxes[:, 1] = torch.clamp(target_boxes[:, 1], 0, sample_visualization_h)
        target_boxes[:, 2] = torch.clamp(target_boxes[:, 2], 0, sample_visualization_w)
        target_boxes[:, 3] = torch.clamp(target_boxes[:, 3], 0, sample_visualization_h)

        target_boxes = target_boxes.numpy().astype(np.int32)
        target_labels = target_labels.numpy().astype(np.int32)

        draw = ImageDraw.Draw(sample_visualization)

        # draw target boxes
        for box, label in zip(target_boxes, target_labels):
            x1, y1, x2, y2 = box

            # Select color based on class ID
            box_color = BOX_COLORS[int(label) % len(BOX_COLORS)]

            # Draw box (thick)
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)

            label_text = f"{label}"

            # Measure text size
            text_width, text_height = draw.textbbox((0, 0), label_text, font=font)[2:4]

            # Draw text background
            padding = 2
            draw.rectangle(
                [x1, y1 - text_height - padding * 2, x1 + text_width + padding * 2, y1],
                fill=box_color
            )

            # Draw text (LABEL_TEXT_COLOR)
            draw.text((x1 + padding, y1 - text_height - padding), label_text,
                     fill=LABEL_TEXT_COLOR, font=font)

        save_path = Path(output_dir) / f"{split}_samples" / f"{target_image_id}_{target_image_path_stem}.webp"
        sample_visualization.save(save_path)

def show_sample(sample):
    """for coco dataset/dataloader"""
    import matplotlib.pyplot as plt
    from torchvision.transforms.v2 import functional as F
    from torchvision.utils import draw_bounding_boxes

    image, target = sample
    if isinstance(image, PIL.Image.Image):
        image = F.to_image_tensor(image)

    image = F.convert_dtype(image, torch.uint8)
    annotated_image = draw_bounding_boxes(image, target["boxes"], colors="yellow", width=3)

    fig, ax = plt.subplots()
    ax.imshow(annotated_image.permute(1, 2, 0).numpy())
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fig.tight_layout()
    fig.show()
    plt.show()
