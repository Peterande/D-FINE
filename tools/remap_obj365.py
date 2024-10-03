"""Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import json
import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Split and update dataset annotations.')
    parser.add_argument(
        '--base_dir',
        type=str,
        required=True,
        help='Base directory of the dataset, e.g., /data/Objects365/data'
    )
    parser.add_argument(
        '--new_val_size',
        type=int,
        default=5000,
        help='Number of images to include in the new validation set (default: 5000)'
    )
    parser.add_argument(
        '--output_suffix',
        type=str,
        default='new',
        help='Suffix to add to new annotation files (default: new)'
    )
    return parser.parse_args()

def update_image_paths(images, new_prefix):
    for img in images:
        split = img['file_name'].split('/')[1:]
        img['file_name'] = os.path.join(new_prefix, *split)
    return images

def create_split_annotations(original_annotations, split_image_ids, new_prefix, output_file):
    new_images = [img for img in original_annotations['images'] if img['id'] in split_image_ids]
    if new_prefix is not None:
        new_images = update_image_paths(new_images, new_prefix)
    
    new_annotations = {
        'images': new_images,
        'annotations': [ann for ann in original_annotations['annotations'] if ann['image_id'] in split_image_ids],
        'categories': original_annotations['categories']
    }
    with open(output_file, 'w') as f:
        json.dump(new_annotations, f)
    print(f'Annotation saved to {output_file}')

def main():
    args = parse_arguments()
    base_dir = args.base_dir
    new_val_size = args.new_val_size
    output_suffix = args.output_suffix

    # Define paths based on the base directory
    original_train_ann_file = os.path.join(base_dir, 'train', 'zhiyuan_objv2_train.json')
    original_val_ann_file = os.path.join(base_dir, 'val', 'zhiyuan_objv2_val.json')

    new_val_ann_file = os.path.join(base_dir, 'val', f'{output_suffix}_zhiyuan_objv2_val.json')
    new_train_ann_file = os.path.join(base_dir, 'train', f'{output_suffix}_zhiyuan_objv2_train.json')

    # Check if original annotation files exist
    if not os.path.isfile(original_train_ann_file):
        print(f'Error: Training annotation file not found at {original_train_ann_file}')
        return
    if not os.path.isfile(original_val_ann_file):
        print(f'Error: Validation annotation file not found at {original_val_ann_file}')
        return

    # Load the original training and validation annotations
    with open(original_train_ann_file, 'r') as f:
        train_annotations = json.load(f)
    
    with open(original_val_ann_file, 'r') as f:
        val_annotations = json.load(f)
    
    # Extract image IDs from the original validation set
    val_image_ids = [img['id'] for img in val_annotations['images']]
    
    # Split image IDs for the new training and validation sets
    new_val_image_ids = val_image_ids[:new_val_size]
    new_train_image_ids = val_image_ids[new_val_size:]
    
    # Create new validation annotation file
    create_split_annotations(val_annotations, new_val_image_ids, None, new_val_ann_file)
    
    # Combine the remaining validation images and annotations with the original training data
    new_train_images = [img for img in val_annotations['images'] if img['id'] in new_train_image_ids]
    new_train_images = update_image_paths(new_train_images, 'images_from_val')
    new_train_annotations = [ann for ann in val_annotations['annotations'] if ann['image_id'] in new_train_image_ids]
    
    # Add the original training images and annotations
    new_train_images.extend(train_annotations['images'])
    new_train_annotations.extend(train_annotations['annotations'])
    
    # Create a new training annotation dictionary
    new_train_annotations_dict = {
        'images': new_train_images,
        'annotations': new_train_annotations,
        'categories': train_annotations['categories']
    }
    
    # Save the new training annotations
    with open(new_train_ann_file, 'w') as f:
        json.dump(new_train_annotations_dict, f)
    
    print(f'New training annotations saved to {new_train_ann_file}')
    print(f'New validation annotations saved to {new_val_ann_file}')

if __name__ == '__main__':
    main()
