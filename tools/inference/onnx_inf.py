"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torchvision.transforms as T
from PIL import Image
import supervision as sv


def resize_with_aspect_ratio(image, size, interpolation=Image.BILINEAR):
    """Resizes an image while maintaining aspect ratio and pads it."""
    original_width, original_height = image.size
    ratio = min(size / original_width, size / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    image = image.resize((new_width, new_height), interpolation)

    # Create a new image with the desired size and paste the resized image onto it
    new_image = Image.new("RGB", (size, size))
    new_image.paste(image, ((size - new_width) // 2, (size - new_height) // 2))
    return new_image, ratio, (size - new_width) // 2, (size - new_height) // 2


def draw(images, labels, boxes, scores, ratios, paddings, thrh=0.4):
    result_images = []
    for i, im in enumerate(images):
        np_image = np.array(im)

        scr = scores[i]
        lab = labels[i]
        box = boxes[i]

        keep_mask = scr > thrh
        scr = scr[keep_mask]
        lab = lab[keep_mask]
        box = box[keep_mask]

        ratio = ratios[i]
        pad_w, pad_h = paddings[i]

        adjusted_boxes = []
        for b in box:
            x1 = (b[0] - pad_w) / ratio
            y1 = (b[1] - pad_h) / ratio
            x2 = (b[2] - pad_w) / ratio
            y2 = (b[3] - pad_h) / ratio
            adjusted_boxes.append([x1, y1, x2, y2])
        adjusted_boxes = np.array(adjusted_boxes)

        detections = sv.Detections(
            xyxy=adjusted_boxes,
            confidence=scr,
            class_id=lab.astype(int),
        )

        height, width = np_image.shape[:2]
        resolution_wh = (width, height)

        text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)
        line_thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)

        box_annotator = sv.BoxAnnotator(thickness=line_thickness)
        label_annotator = sv.LabelAnnotator(text_scale=text_scale, smart_position=True)

        label_texts = [
            f"{class_id} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]

        np_image = box_annotator.annotate(scene=np_image, detections=detections)
        np_image = label_annotator.annotate(
            scene=np_image,
            detections=detections,
            labels=label_texts,
        )

        result_im = Image.fromarray(np_image)
        result_images.append(result_im)

    return result_images


def process_image(sess, im_pil):
    # Resize image while preserving aspect ratio
    resized_im_pil, ratio, pad_w, pad_h = resize_with_aspect_ratio(im_pil, 640)
    orig_size = torch.tensor([[resized_im_pil.size[1], resized_im_pil.size[0]]])

    transforms = T.Compose(
        [
            T.ToTensor(),
        ]
    )
    im_data = transforms(resized_im_pil).unsqueeze(0)

    output = sess.run(
        output_names=None,
        input_feed={"images": im_data.numpy(), "orig_target_sizes": orig_size.numpy()},
    )

    labels, boxes, scores = output

    result_images = draw([im_pil], labels, boxes, scores, [ratio], [(pad_w, pad_h)])
    result_images[0].save("onnx_result.jpg")
    print("Image processing complete. Result saved as 'result.jpg'.")


def process_video(sess, video_path):
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("onnx_result.mp4", fourcc, fps, (orig_w, orig_h))

    frame_count = 0
    print("Processing video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Resize frame while preserving aspect ratio
        resized_frame_pil, ratio, pad_w, pad_h = resize_with_aspect_ratio(frame_pil, 640)
        orig_size = torch.tensor([[resized_frame_pil.size[1], resized_frame_pil.size[0]]])

        transforms = T.Compose(
            [
                T.ToTensor(),
            ]
        )
        im_data = transforms(resized_frame_pil).unsqueeze(0)

        output = sess.run(
            output_names=None,
            input_feed={"images": im_data.numpy(), "orig_target_sizes": orig_size.numpy()},
        )

        labels, boxes, scores = output

        # Draw detections on the original frame
        result_images = draw([frame_pil], labels, boxes, scores, [ratio], [(pad_w, pad_h)])
        frame_with_detections = result_images[0]

        # Convert back to OpenCV image
        frame = cv2.cvtColor(np.array(frame_with_detections), cv2.COLOR_RGB2BGR)

        # Write the frame
        out.write(frame)
        frame_count += 1

        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print("Video processing complete. Result saved as 'result.mp4'.")


def main(args):
    """Main function."""
    # Load the ONNX model
    sess = ort.InferenceSession(args.onnx)
    print(f"Using device: {ort.get_device()}")

    input_path = args.input

    try:
        # Try to open the input as an image
        im_pil = Image.open(input_path).convert("RGB")
        process_image(sess, im_pil)
    except IOError:
        # Not an image, process as video
        process_video(sess, input_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, required=True, help="Path to the ONNX model file.")
    parser.add_argument(
        "--input", type=str, required=True, help="Path to the input image or video file."
    )
    args = parser.parse_args()
    main(args)
