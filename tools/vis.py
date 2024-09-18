import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from PIL import Image
import cv2
import argparse
import matplotlib.pyplot as plt
from matplotlib import rcParams
plt.rcParams['font.family'] = 'Times New Roman'
from src.core import YAMLConfig

import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np

# New imports
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageFont

def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'
original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr


def generate_linspace(reg_max, up, reg_scale, ref, deploy=True):
    if deploy:
        upper_bound1 = (abs(up) * abs(reg_scale))
        upper_bound2 = (abs(up) * abs(reg_scale) * 2)
        step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
        left_values = [-(step) ** i + 1 for i in range(reg_max // 2 - 1, 0, -1)]
        right_values = [(step) ** i - 1 for i in range(1, reg_max // 2)]
        values = [-upper_bound2] + left_values + [0] + right_values + [upper_bound2]
        return torch.tensor(values, device=ref.device, dtype=ref.dtype)
    
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w.clamp(min=0.0)), (y_c - 0.5 * h.clamp(min=0.0)),
         (x_c + 0.5 * w.clamp(min=0.0)), (y_c + 0.5 * h.clamp(min=0.0))]
    return torch.stack(b, dim=-1)

from PIL import ImageFont, ImageDraw, Image

def draw_bounding_boxes(image_pil, boxes_first, boxes_last, distributions_first, distributions_last, flag, clip, save_path, image_size, confidence_threshold=0.3):
    """
    Draw bounding boxes and distributions from both first and last layers on the image.

    :param image_pil: PIL Image object
    :param boxes_first: Tensor of first layer bounding boxes (L, 4)
    :param boxes_last: Tensor of last layer bounding boxes (L, 4)
    :param distributions_first: Tensor of first layer distributions (L, 4, 33)
    :param distributions_last: Tensor of last layer distributions (L, 4, 33)
    :param flag: Index of the bounding box to process
    :param save_path: Path to save the image with drawn bounding boxes
    :param image_size: Tuple of (width, height) to convert relative coordinates to absolute
    :param confidence_threshold: Minimum confidence score to draw the bounding box
    """
    # Convert PIL Image to OpenCV image
    image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    original_width, original_height = image_size

    # Resize image if it's smaller than 640 pixels wide
    if original_width < 1280:
        scale = 1280 / original_width
        image = cv2.resize(image, (1280, int(original_height * scale)), interpolation=cv2.INTER_LINEAR)
        width, height = 1280, int(original_height * scale)
    else:
        scale = 1.0
        width, height = original_width, original_height

    # Determine the amount to cut from the image
    cut = height + 300 - 1150
    if clip:
        image = image[cut:, :]
        crop_offset = cut
    else:
        image = image[:-cut, :]
        crop_offset = 0

    # Update image dimensions after cropping
    height, width = image.shape[:2]

    # Adjust the boxes to account for the crop and resizing
    box_first = boxes_first[flag]
    box_last = boxes_last[flag]

    # Scale boxes from normalized coordinates to original image size
    box_first = box_first * torch.tensor([original_width, original_height, original_width, original_height], device=box_first.device)
    box_last = box_last * torch.tensor([original_width, original_height, original_width, original_height], device=box_last.device)

    # Adjust for resizing if applicable
    if scale != 1.0:
        box_first *= scale
        box_last *= scale

    # Adjust for cropping
    box_first[1] -= crop_offset # - 30 000000365208
    box_first[3] -= crop_offset
    box_last[1] -= crop_offset # - 30
    box_last[3] -= crop_offset

    # Clamp box coordinates to the image dimensions
    min_tensor = torch.zeros_like(box_first)
    max_tensor = torch.tensor([width, height, width, height], device=box_first.device)
    box_first = box_first.clamp(min=min_tensor, max=max_tensor)
    box_last = box_last.clamp(min=min_tensor, max=max_tensor)

    box_first = box_first.int().tolist()
    box_last = box_last.int().tolist()
    
    # Draw first layer bounding box in blue
    cv2.rectangle(image, (box_first[0], box_first[1]), (box_first[2], box_first[3]), (0, 0, 255), 4)  # Blue box

    # Draw last layer bounding box in green
    cv2.rectangle(image, (box_last[0], box_last[1]), (box_last[2], box_last[3]), (0, 255, 0), 4)  # Green box

    # Convert OpenCV image back to PIL Image for text drawing
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)

    # Place labels in the upper right corner with background color
    margin_right = 20
    margin_top = 50
    line_height = 50  # Adjust line height based on font size
    x_label = width - margin_right
    y_label = margin_top

    # Define labels and their colors
    labels = [('Initial BBox and Distributions', (255, 0, 0)), ('Last Layer BBox and Distributions', (0, 255, 0))]

    # Load 'Times New Roman' font
    font_path = '/home/pengys/code/rtdetrv2_pytorch/tools/Times New Roman.ttf'  # Replace with the actual path to your 'Times New Roman' font file
    font_size = 32  # Adjust font size as needed
    font = ImageFont.truetype(font_path, font_size)

    for text, color in labels:
        # Get text size using font.getsize()
        bbox = font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Update x coordinate to align text to the right
        x_text = x_label - text_width
        square_size = 30
        x_square = x_text - square_size - 10  # 10 pixels gap between square and text
        y_square = y_label - text_height

        # Coordinates for the background rectangle
        padding = 10  # Padding around the text
        rect_x1 = x_square - padding
        rect_y1 = y_square - padding
        rect_x2 = x_label + padding
        rect_y2 = y_label + padding

        # Draw filled rectangle as background (light gray color)
        draw.rectangle([(rect_x1, rect_y1), (rect_x2, rect_y2)], fill=(220, 220, 220))

        # Draw color square next to label
        draw.rectangle([(x_square, y_square), (x_square + square_size, y_square + square_size)], fill=color)

        # Draw text over the background rectangle
        draw.text((x_text, y_square - 3), text, font=font, fill=(0, 0, 0))

        # Update y coordinate for next label
        y_label += line_height

    # Prepare distributions
    image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    distributions_first_unweighted = distributions_first[flag].reshape(4, 33).softmax(-1).cpu()
    distributions_last_unweighted = distributions_last[flag].reshape(4, 33).softmax(-1).cpu()
    
    # Generate project for weighting
    project = generate_linspace(32, 0.5, 8, distributions_first_unweighted)
    # Apply weighting
    distributions_first_weighted = distributions_first_unweighted * project
    distributions_last_weighted = distributions_last_unweighted * project

    # Now, generate plots for each distribution and overlay them onto the image
    plot_height = 300  # Height of each plot image
    plot_width = image.shape[1] // 4  # Width of each plot image
    num_rows = 2  # Number of rows for unweighted and weighted plots
    total_height = image.shape[0] + plot_height * num_rows  # Total height for the new image

    new_image = np.zeros((total_height, image.shape[1], 3), dtype=np.uint8)
    new_image[0:image.shape[0], :, :] = image

    titles = ['Left', 'Top', 'Right', 'Bottom']
    for row in range(num_rows):
        for i in range(4):
            if row == 0:
                # Unweighted distributions
                dist_first = distributions_first_unweighted[i].numpy()
                dist_last = distributions_last_unweighted[i].numpy()
                row_title = ''
            else:
                # Weighted distributions
                dist_first = distributions_first_weighted[i].numpy()
                dist_last = distributions_last_weighted[i].numpy()
                row_title = ' (Weighted)'
            title = f"{titles[i]}{row_title}"
            fig = Figure(figsize=(plot_width/100, plot_height/100), dpi=100)
            canvas = FigureCanvas(fig)
            ax = fig.gca()
            x = np.arange(len(dist_first))

            # Plot both distributions on the same plot without labels
            ax.plot(x, dist_first, label='Initial BBox and Distributions', color='red', linewidth=2)
            ax.plot(x, dist_last, label='Final BBox and Distributions', color='green', linewidth=2)

            # Set x-axis ticks to 0, 16, and 32
            ax.set_xticks([0, 16, 32])
            ax.set_xticklabels(['0', '16', '32'], fontsize=16)

            # Find min and max values for consistent y-axis scaling
            max_value = max(dist_first.max(), dist_last.max()) * 1.1
            min_value = min(dist_first.min(), dist_last.min()) * 1.1

            # Set y-axis ticks
            ax.set_yticks([min_value, 0, max_value])
            ax.set_yticklabels([f'{min_value:.2f}', '0', f'{max_value:.2f}'], fontsize=16)

            # Set y-axis limits
            ax.set_ylim(min_value, max_value)

            ax.set_title(title, fontsize=18)  # Adjust title font size
            ax.tick_params(axis='both', which='major', labelsize=14)  # Adjust tick label font size

            # Do not add legend in the plot

            fig.tight_layout()
            canvas.draw()
            buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plot_img = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
            plot_img = cv2.resize(plot_img, (plot_width, plot_height))
            x_offset = i * plot_width
            y_offset = image.shape[0] + row * plot_height
            new_image[y_offset:y_offset+plot_height, x_offset:x_offset+plot_width, :] = plot_img

    cv2.imwrite(save_path, new_image)
    print(f"Predicted image saved at {save_path}")




def main(args):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']

        # NOTE load train mode state -> convert to deploy mode
        model_dict = cfg.model.state_dict()
        cfg.model.load_state_dict(state, strict=True)

    else:
        # raise AttributeError('Only support resume to load model.state_dict by now.')
        print('not load model.state_dict, use default init state dict...')

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = cfg.model

        def forward(self, images, labels, orig_target_sizes):
            outputs = self.model(images, labels)
            return outputs

    model = Model().cuda().eval()

    # Define pre-processing transforms
    transform = T.Compose([
        T.Resize((640, 640)),  # Resize to 640x640
        T.ToTensor(),  # Convert image to tensor
        T.ConvertImageDtype(torch.float32),  # Convert to float32
    ])

    # Define the list of image paths and corresponding indices
    image_paths = [
        './saved_filtered_view/data/000000089648.jpg',
        './saved_filtered_view/data/000000132796.jpg',
        './saved_filtered_view/data/000000142971.jpg',
        './saved_filtered_view/data/000000261888.jpg',
        './saved_filtered_view/data/000000267351.jpg',
        './saved_filtered_view/data/000000365208.jpg',
        './saved_filtered_view/data/000000551820.jpg',
    ]

    indices = [4, 1, 4, 0, 0, 0, 9]  # Corresponding indices for each image
    clip_margin = [0, 1, 0, 1, 1, 1, 1]

    for image_path, idx, clip in zip(image_paths, indices, clip_margin):
        # Load and preprocess the image
        image_pil = Image.open(image_path).convert('RGB')  # Load image as PIL Image
        image_tensor = transform(image_pil).unsqueeze(0).cuda()  # Add batch dimension

        orig_target_sizes = torch.tensor([[image_pil.size[1], image_pil.size[0]]]).cuda()

        # Perform inference
        with torch.no_grad():
            outputs = model(image_tensor, None, orig_target_sizes)

        # Process batch output (assuming B=1)
        boxes_first_layer = box_cxcywh_to_xyxy(outputs['aux_outputs'][0]['pred_boxes'])[0]  # First layer boxes
        distributions_first_layer = outputs['aux_outputs'][0]['pred_corners'][0]

        boxes_last_layer = box_cxcywh_to_xyxy(outputs['pred_boxes'])[0]  # Last layer boxes
        distributions_last_layer = outputs['pred_corners'][0]

        scores = outputs['pred_logits'][0].sigmoid()  # Extract from batch, shape (L,)

        print(f"Processing image: {os.path.basename(image_path)}")
        print(f"Detected {boxes_last_layer.size(0)} bounding boxes.")

        # Draw and save result, filtering by confidence threshold
        save_path = os.path.join(os.path.dirname(image_path), 'Pred_' + os.path.basename(image_path).split('.')[0] + '.png')

        # Ensure idx is within valid range
        if idx < 0 or idx >= boxes_last_layer.size(0):
            print(f"Index {idx} is out of range for image {image_path}. Skipping.")
            continue

        draw_bounding_boxes(image_pil, boxes_first_layer, boxes_last_layer, distributions_first_layer, distributions_last_layer, idx, clip, save_path, image_pil.size, confidence_threshold=0.)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='/home/pengys/code/rtdetrv2_pytorch/configs/dfine/dfine_hgnetv2_b4_6x_coco.yml', type=str)
    parser.add_argument('--resume', '-r', default='/home/pengys/code/rtdetrv2_pytorch/weight/b4/ema_0.9997_0.5394.pth', type=str)
    # parser.add_argument('--image_path', '-i', default='./saved_filtered_view/data/000000089648.jpg', type=str, help='Path to the input image')

    args = parser.parse_args()
    main(args)
