"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import os
import json
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Resize images and update dataset annotations for both train and val sets.')
    parser.add_argument(
        '--base_dir',
        type=str,
        required=True,
        help='Base directory of the dataset, e.g., /data/Objects365/data'
    )
    parser.add_argument(
        '--max_size',
        type=int,
        default=640,
        help='Maximum size for the longer side of the image (default: 640)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of worker threads for parallel processing (default: 4)'
    )
    return parser.parse_args()

def resize_image_and_update_annotations(image_path, annotations, max_size=640):
    try:
        with Image.open(image_path) as img:
            w, h = img.size
            if max(w, h) <= max_size:
                return annotations, w, h, False  # No need to resize

            scale = max_size / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            new_image_path = image_path.replace('.jpg', '_resized.jpg')
            img.save(new_image_path)
            print(f"Resized image saved: {new_image_path}")
            print(f"Original size: ({w}, {h}), New size: ({new_w}, {new_h})")

            # Update annotations
            for ann in annotations:
                ann['area'] = ann['area'] * (scale ** 2)
                ann['bbox'] = [coord * scale for coord in ann['bbox']]
                if 'orig_size' in ann:
                    ann['orig_size'] = (new_w, new_h)
                if 'size' in ann:
                    ann['size'] = (new_w, new_h)

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

    return annotations, new_w, new_h, True

def resize_images_and_update_annotations(base_dir, subset, max_size=640, num_workers=4):
    json_file = os.path.join(base_dir, subset, 'new_zhiyuan_objv2_{}.json'.format(subset))
    if not os.path.isfile(json_file):
        print(f'Error: JSON file not found at {json_file}')
        return

    with open(json_file, 'r') as f:
        data = json.load(f)

    image_annotations = {img['id']: [] for img in data['images']}
    for ann in data['annotations']:
        image_annotations[ann['image_id']].append(ann)

    def process_image(image_info):
        image_path = os.path.join(base_dir, subset, image_info['file_name'])
        results = resize_image_and_update_annotations(image_path, image_annotations[image_info['id']], max_size)
        if results is None:
            updated_annotations, new_w, new_h, resized = None, None, None, None
        else:
            updated_annotations, new_w, new_h, resized = results
        return image_info, updated_annotations, new_w, new_h, resized

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_image, data['images']))

    new_images = []
    new_annotations = []

    for image_info, updated_annotations, new_w, new_h, resized in results:
        if updated_annotations is not None:
            image_info['width'] = new_w
            image_info['height'] = new_h
            image_annotations[image_info['id']] = updated_annotations
            if resized:
                image_info['file_name'] = image_info['file_name'].replace('.jpg', '_resized.jpg')
            new_images.append(image_info)
            new_annotations.extend(updated_annotations)

    new_data = {
        'images': new_images,
        'annotations': new_annotations,
        'categories': data['categories']
    }

    new_json_file = json_file.replace('.json', '_resized.json')
    with open(new_json_file, 'w') as f:
        json.dump(new_data, f)

    print(f'New JSON file saved to {new_json_file}')

def main():
    args = parse_arguments()
    base_dir = args.base_dir
    max_size = args.max_size
    num_workers = args.num_workers

    subsets = ['train', 'val']
    for subset in subsets:
        print(f'Processing subset: {subset}')
        resize_images_and_update_annotations(
            base_dir=base_dir,
            subset=subset,
            max_size=max_size,
            num_workers=num_workers
        )

if __name__ == "__main__":
    main()
