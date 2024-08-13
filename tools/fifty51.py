import os
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

def main(args):
    try:
        dataset = foz.load_zoo_dataset(
            "coco-2017",
            split="validation",
            dataset_name="evaluate-detections-tutorial",
            dataset_dir="/home/pengys/Data/fiftyone"
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

        model = CustomModel(cfg)
        predictions_view = dataset.take(100, seed=51)
        predictions_view.apply_model(model, label_field="predictions")
        high_conf_view = predictions_view.filter_labels("predictions", F("confidence") > 0.5, only_matches=False)
        results = high_conf_view.evaluate_detections(
        "predictions",
        gt_field="ground_truth",
        eval_key="eval",
        compute_mAP=True,
        )
        # eval_view = dataset.load_evaluation_view("eval")
        session.view = high_conf_view.sort_by("eval_fp", reverse=True)

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
    parser.add_argument('--config', '-c', type=str, default= \
                        "/home/pengys/code/dfine_pytorch/configs/dfine/dfine_hgnetv2_b4_6x_coco.yml")
    parser.add_argument('--resume', '-r', type=str, default= \
                        "/home/pengys/code/dfine_pytorch/log/merge+allious/best.pth")
    args = parser.parse_args()

    main(args)
