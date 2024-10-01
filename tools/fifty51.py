"""Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import os
import subprocess

def kill_existing_mongod():
    try:
        result = subprocess.run(['ps', 'aux'], stdout=subprocess.PIPE)
        processes = result.stdout.decode('utf-8').splitlines()

        for process in processes:
            if 'mongod' in process and '--dbpath' in process:
                # find mongod PID
                pid = int(process.split()[1])
                print(f"Killing existing mongod process with PID: {pid}")
                # kill mongod session
                os.kill(pid, 9)
    except Exception as e:
        print(f"Error occurred while killing mongod: {e}")

kill_existing_mongod()

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse

from src.core import YAMLConfig

import torch
import fiftyone.core.models as fom
import fiftyone as fo
import fiftyone.zoo as foz
import torchvision.transforms as transforms
from PIL import Image
import fiftyone.core.labels as fol
import fiftyone.core.fields as fof
from fiftyone import ViewField as F
import time
import tqdm
label_map = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorbike', 5: 'aeroplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'trafficlight',
    11: 'firehydrant', 12: 'streetsign', 13: 'stopsign', 14: 'parkingmeter',
    15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse',
    20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra',
    25: 'giraffe', 26: 'hat', 27: 'backpack', 28: 'umbrella', 29: 'shoe',
    30: 'eyeglasses', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
    35: 'skis', 36: 'snowboard', 37: 'sportsball', 38: 'kite', 39: 'baseballbat',
    40: 'baseballglove', 41: 'skateboard', 42: 'surfboard', 43: 'tennisracket',
    44: 'bottle', 45: 'plate', 46: 'wineglass', 47: 'cup', 48: 'fork',
    49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
    54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hotdog',
    59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'sofa',
    64: 'pottedplant', 65: 'bed', 66: 'mirror', 67: 'diningtable', 68: 'window',
    69: 'desk', 70: 'toilet', 71: 'door', 72: 'tv', 73: 'laptop',
    74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cellphone', 78: 'microwave',
    79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 83: 'blender',
    84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddybear',
    89: 'hairdrier', 90: 'toothbrush', 91: 'hairbrush'
}

class CustomModel(fom.Model):
    def __init__(self, cfg):
        super().__init__()
        self.model = cfg.model.eval().cuda()
        self.postprocessor = cfg.postprocessor.eval().cuda()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((640, 640)),  # Resize to the size expected by your model
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])

    @property
    def media_type(self):
        return "image"

    @property
    def has_logits(self):
        return False

    @property
    def has_embeddings(self):
        return False

    @property
    def ragged_batches(self):
        return False

    @property
    def transforms(self):
        return None

    @property
    def preprocess(self):
        return True

    @preprocess.setter
    def preprocess(self, value):
        pass

    def _convert_predictions(self, predictions):
        class_labels, bboxes, scores = predictions[0]['labels'], predictions[0]['boxes'], predictions[0]['scores']

        detections = []
        for label, bbox, score in zip(class_labels, bboxes, scores):
            detection = fol.Detection(
                label=label_map[label.item()],
                bounding_box=[
                    bbox[0] / 640,  # Normalized coordinates
                    bbox[1] / 640,
                    (bbox[2] - bbox[0]) / 640,
                    (bbox[3] - bbox[1]) / 640
                ],
                confidence=score
            )
            detections.append(detection)

        return fol.Detections(detections=detections)

    def predict(self, image):
        image = Image.fromarray(image).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).cuda()
        outputs = self.model(image_tensor)
        orig_target_sizes = torch.tensor([[640, 640]]).cuda()
        predictions = self.postprocessor(outputs, orig_target_sizes)
        return self._convert_predictions(predictions)

    def predict_all(self, images):
        image_tensors = []
        for image in images:
            image = Image.fromarray(image)
            image_tensor = self.transform(image)
            image_tensors.append(image_tensor)
        image_tensors = torch.stack(image_tensors).cuda()
        outputs = self.model(image_tensors)
        orig_target_sizes = torch.tensor([[640, 640] for image in images]).cuda()
        predictions = self.postprocessor(outputs, orig_target_sizes)
        converted_predictions = [self._convert_predictions(pred) for pred in predictions]

        # Ensure the output is a list of lists of Detections
        return converted_predictions

def filter_by_predictions5_confidence(predictions_view, confidence_threshold=0.3):
    for j, sample in tqdm.tqdm(enumerate(predictions_view), total=len(predictions_view)):
        has_modified = False
        for i, detection in enumerate(sample["predictions0"].detections):

            if "original_confidence" not in detection:
                detection["original_confidence"] = detection["confidence"]
                
            if (detection["confidence"] <= confidence_threshold and sample["predictions5"].detections[i]["confidence"] >= confidence_threshold) or \
               (detection["confidence"] >= confidence_threshold and sample["predictions5"].detections[i]["confidence"] <= confidence_threshold):

                sample["predictions0"].detections[i]["confidence"] = sample["predictions5"].detections[i]["confidence"]
                has_modified = True 
        if has_modified:
            sample.save()


def restore_confidence(predictions_view):
    for j, sample in tqdm.tqdm(enumerate(predictions_view), total=len(predictions_view)):
        for i, detection in enumerate(sample["predictions0"].detections):
            if "original_confidence" in detection:
                detection["confidence"] = detection["original_confidence"]
        sample.save()
      
def fast_iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    xA = max(x1, x2)
    yA = max(y1, y2)
    xB = min(x1 + w1, x2 + w2)
    yB = min(y1 + h1, y2 + h2)
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = w1 * h1
    boxBArea = w2 * h2
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou
    
def assign_iou_diff(predictions_view):
    for sample in predictions_view:
        ious_0 = [detection.eval0_iou if 'eval0_iou' in detection else None for detection in sample["predictions0"].detections]
        ious_5 = [detection.eval5_iou if 'eval5_iou' in detection else None for detection in sample["predictions5"].detections]
        bbox_0 = [detection.bounding_box for detection in sample["predictions0"].detections]
        bbox_5 = [detection.bounding_box for detection in sample["predictions5"].detections]
        # iou_diffs = [abs(iou_5 - iou_0) if iou_0 is not None and iou_5 is not None else -1 for iou_0, iou_5 in zip(ious_0, ious_5)]
        iou_inter = [fast_iou(b0, b5) for b0, b5 in zip(bbox_0, bbox_5)]
        iou_diffs = [abs(iou_5 - iou_0) if iou_0 is not None and iou_5 is not None and iou_inter > 0.5 else -1 for iou_0, iou_5, iou_inter in zip(ious_0, ious_5, iou_inter)]
        
        for detection, iou_diff in zip(sample["predictions0"].detections, iou_diffs):
            detection["iou_diff"] = iou_diff
        for detection, iou_diff in zip(sample["predictions5"].detections, iou_diffs):
            detection["iou_diff"] = iou_diff
        # for detection, iou_diff in zip(sample["predictions100"].detections, iou_diffs):
        #     detection["iou_diff"] = iou_diff
        sample.save()
       
def main(args):
    print("Shutting down session")
    if 'session' in locals():
        session.close()
    try:
        if os.path.exists("saved_predictions_view") and os.path.exists("saved_filtered_view"):
            print("Loading saved predictions and filtered views...")
            dataset = foz.load_zoo_dataset(
                "coco-2017",
                split="validation",
                dataset_name="evaluate-detections-tutorial",
                dataset_dir="/home/pengys/Data/fiftyone"
            )
            
            dataset.persistent = True
            session = fo.launch_app(dataset)

            predictions_view = fo.Dataset.from_dir(
                dataset_dir="saved_predictions_view",
                dataset_type=fo.types.FiftyOneDataset
            ).view()
            filtered_view = fo.Dataset.from_dir(
                dataset_dir="saved_filtered_view",
                dataset_type=fo.types.FiftyOneDataset
            ).view()
        else:
            dataset = foz.load_zoo_dataset(
                "coco-2017",
                split="validation",
                dataset_name="evaluate-detections-tutorial",
                dataset_dir="data/fiftyone"
            )
            
            dataset.persistent = True
            
            session = fo.launch_app(dataset)
            cfg = YAMLConfig(args.config, resume=args.resume)
            if args.resume:
                checkpoint = torch.load(args.resume, map_location='cpu') 
                if 'ema' in checkpoint:
                    state = checkpoint['ema']['module']
                else:
                    state = checkpoint['model']
            else:
                raise AttributeError('only support resume to load model.state_dict by now.')

            # NOTE load train mode state -> convert to deploy mode
            cfg.model.load_state_dict(state)
            predictions_view = dataset.take(5000, seed=51)

            # Apply models and save predictions in different label fields
            for i in [0, 5]:
                model = CustomModel(cfg)
                model.model.decoder.decoder.eval_idx = i
                label_field = "predictions{:d}".format(i)
                predictions_view.apply_model(model, label_field=label_field)

            # 过滤
            filter_by_predictions5_confidence(predictions_view, confidence_threshold=0.3)
            for i in [0, 5]:
                label_field = "predictions{:d}".format(i)
                predictions_view = predictions_view.filter_labels(label_field, F("confidence") > 0.3, only_matches=False)
                eval_key = "eval{:d}".format(i)
                _ = predictions_view.evaluate_detections(
                    label_field,
                    gt_field="ground_truth",
                    eval_key=eval_key,
                    compute_mAP=True,
                )
            
            assign_iou_diff(predictions_view)

            filtered_view = predictions_view.filter_labels("predictions0", F("iou_diff") > 0.05, only_matches=True)
            filtered_view = filtered_view.filter_labels("predictions5", F("iou_diff") > 0.05, only_matches=True)
            restore_confidence(filtered_view)

            predictions_view.export(
                export_dir="saved_predictions_view",
                dataset_type=fo.types.FiftyOneDataset
            )
            filtered_view.export(
                export_dir="saved_filtered_view",
                dataset_type=fo.types.FiftyOneDataset
            )

        # Display the filtered view
        session.view = filtered_view

        # Keep the session open
        while True:
            time.sleep(1)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Shutting down session")
        if 'session' in locals():
            session.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default= "configs/dfine/dfine_hgnetv2_l_6x_coco.yml")
    parser.add_argument('--resume', '-r', type=str, default= "weight/l/best.pth")
    args = parser.parse_args()

    main(args)
