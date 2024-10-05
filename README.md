<!--# [D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement](https://arxiv.org/abs/xxxxxx) -->
<h2 align="center">
  D-FINE: Redefine Regression Task of DETRs as Fine&#8209;grained&nbsp;Distribution&nbsp;Refinement
</h2>

<p align="center">
    <!-- <a href="https://github.com/lyuwenyu/RT-DETR/blob/main/LICENSE">
        <img alt="license" src="https://img.shields.io/badge/LICENSE-Apache%202.0-blue">
    </a> -->
    <a href="https://github.com/Peterande/D-FINE/blob/master/LICENSE">
        <img alt="license" src="https://img.shields.io/github/license/Peterande/D-FINE">
    </a>
    <a href="https://github.com/Peterande/D-FINE/pulls">
        <img alt="prs" src="https://img.shields.io/github/issues-pr/Peterande/D-FINE">
    </a>
    <a href="https://github.com/Peterande/D-FINE/issues">
        <img alt="issues" src="https://img.shields.io/github/issues/Peterande/D-FINE?color=pink">
    </a>
    <a href="https://github.com/Peterande/D-FINE">
        <img alt="issues" src="https://img.shields.io/github/stars/Peterande/D-FINE">
    </a>
    <a href="https://arxiv.org/abs/xxx.xxx">
        <img alt="arXiv" src="https://img.shields.io/badge/arXiv-xxx.xxx-red">
    </a>
    <a href="mailto: pengyansong@mail.ustc.edu.cn">
        <img alt="emal" src="https://img.shields.io/badge/contact_me-email-yellow">
    </a>
</p>

<p align="center">
    📄 This is the official implementation of the paper:
    <br>
    <a href="https://arxiv.org/abs/xxxxxx">D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement</a>
</p>


<p align="center">
Yansong Peng, Hebei Li, Peixi Wu, Yueyi Zhang, Xiaoyan Sun, and Feng Wu
</p>

<!-- <table><tr>
<td><img src=https://github.com/Peterande/storage/blob/main/latency.png border=0 width=333></td>
<td><img src=https://github.com/Peterande/storage/blob/main/params.png border=0 width=333></td>
<td><img src=https://github.com/Peterande/storage/blob/main/flops.png border=0 width=333></td>
</tr></table> -->

<table><tr>
<td><img src=https://github.com/Peterande/storage/blob/main/stats_padded.png border=0 width=1000></td>
</tr></table>


## 🚀 Updates
- [x] **\[2024.10.3\]** Release D-FINE series.
<!-- - 🔜 **\[Next\]** Release D-FINE series pretrained on Objects365. -->

## 🔍 Discover the Key Innovations Behind D-FINE
English | [简体中文](README_cn.md)
<details open>
<summary> Introduction </summary>

## D-FINE: Redefining Regression in Object Detection

D-FINE redefines the regression task in DETR-based object detectors. 

**Unlike traditional methods, our FDR method decomposes the detection box generation process into two key steps:**

1. **Initial Box Prediction**: Similar to conventional methods, initial bounding boxes are predicted.
2. **Fine-grained Distribution Refinement**: Decoder layers iteratively refine four sets of probability distribution functions. These distributions allow for both precise fine-tuning and larger adjustments to the four edges of the initial bounding box.

### Key Advantages of FDR:
1. **Simplified Supervision**: The residual between the predictions and the Ground Truth (GT) is used to optimize these probability distributions. This allows each decoder layer to focus more effectively on solving the specific localization errors it faces at that stage. As the network deepens, the supervision at each layer becomes progressively simpler, simplifying the overall optimization.

2. **Robustness in Complex Scenarios**: The probability distributions inherently represent the confidence level of different "fine-tuning" adjustments for each boundary. This allows the system to independently model the uncertainty of each edge at each stage, enabling it to handle complex real-world scenarios like occlusion, motion blur, and low-light conditions with greater robustness compared to directly regressing four fixed values.

   
4. **Flexible Refinement Mechanism**: The probability distributions are transformed into final box offsets through a weighted sum. The carefully designed weighting function ensures fine-grained adjustments when the initial box is accurate and larger corrections when necessary.

   
6. **Research Potential and Extensibility**: By transforming the regression task into a probability distribution prediction problem, similar to classification tasks, this framework not only improves compatibility with other tasks but also opens up new research opportunities. It provides a unified and flexible foundation for future innovations in areas such as multi-task learning and distribution modeling.


<!-- Insert figure explaining FDR process -->
<p align="center">
    <img src="https://github.com/Peterande/storage/blob/main/fdr.png" alt="Fine-grained Distribution Refinement Process" width="777">
</p>

## GO-LSD: Extending FDR to Knowledge Distillation

GO-LSD (Global Optimal Localization Self-Distillation) builds upon FDR by enabling localization knowledge distillation across network layers. With the introduction of FDR, the regression task is now a probability distribution prediction, which offers two key benefits:

1. **Knowledge Transfer**: Probability distributions naturally carry localization knowledge, which can be distilled from deeper layers to earlier layers through computing KLD loss. This is something that traditional fixed box representations (Dirac delta functions) cannot achieve.
   
3. **Consistent Optimization Goals**: Since each layer shares a common goal — reducing the residual between the initial bounding box and the ground truth box — the refined probability distributions from the final layer can be used to guide earlier layers through distillation. This creates a synergistic effect: as training progresses, the final layer's predictions become more accurate, and the soft labels it generates help earlier layers improve their predictions. In turn, the earlier layers learn to localize more quickly, simplifying the optimization tasks of the deeper layers, and leading to further improvements in overall accuracy.


<!-- Insert figure explaining GO-LSD process -->
<p align="center">
    <img src="https://github.com/Peterande/storage/blob/main/go_lsd.png" alt="GO-LSD Process" width="777">
</p>

### Will FDR and GO-LSD increase the inference cost?
No, FDR has almost no difference in speed, parameter size, or computational complexity compared to the original prediction method, making it a seamless replacement.

### Will FDR and GO-LSD increase the training cost?
The increased training cost mainly comes from generating the distribution labels. We have optimized this process, keeping the training time increase within 6% and memory consumption within 2%, making the cost almost negligible.


### Visualization of D-FINE Predictions

The following visualization demonstrates D-FINE's predictions in various complex detection scenarios. These include cases with occlusion, low-light conditions, motion blur, depth of field effects, and densely populated scenes. Despite these challenges, D-FINE consistently produces accurate localization results.

<!-- Insert visualization of predictions in challenging scenarios -->
<p align="center">
    <img src="https://github.com/Peterande/storage/blob/main/hard_case.png" alt="D-FINE Predictions in Challenging Scenarios" width="777">
</p>

</details>

## Model Zoo

### Base models
| Model | Dataset | AP<sup>val</sup> | #Params | FPS | GFLOPs | config | checkpoint | log |
| :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
**D-FINE-S** | COCO | **48.5** |  10M | 287 | 25 | [config](./configs/dfine/dfine_hgnetv2_s_coco.yml) | [48.5](https://github.com/Peterande/storage/releases/download/dfinev1/dfine_s_coco.pth) |
**D-FINE-M** | COCO | **52.3** |  19M | 180 | 57 | [config](./configs/dfine/dfine_hgnetv2_m_coco.yml) | [52.3](https://github.com/Peterande/storage/releases/download/dfinev1/dfine_m_coco.pth) |
**D-FINE-L** | COCO | **54.0** |  31M | 129 | 91 | [config](./configs/dfine/dfine_hgnetv2_l_coco.yml) | [54.0](https://github.com/Peterande/storage/releases/download/dfinev1/dfine_l_coco.pth) |
**D-FINE-X** | COCO | **55.8** |  62M | 81 | 202 | [config](./configs/dfine/dfine_hgnetv2_x_coco.yml) | [55.8](https://github.com/Peterande/storage/releases/download/dfinev1/dfine_x_coco.pth) |
**D-FINE-S** | COCO+Objects365 | **50.3** |  10M | 287 | 25 | [config](./configs/dfine/objects365/dfine_hgnetv2_s_obj2coco.yml) | []() |
**D-FINE-M** | COCO+Objects365 | **55.0** |  19M | 180 | 57 | [config](./configs/dfine/objects365/dfine_hgnetv2_m_obj2coco.yml) | []() |
**D-FINE-L** | COCO+Objects365 | **56.9** |  31M | 129 | 91 | [config](./configs/dfine/objects365/dfine_hgnetv2_l_obj2coco.yml) | []() |
**D-FINE-X** | COCO+Objects365 | **59.0** |  62M | 81 | 202 | [config](./configs/dfine/objects365/dfine_hgnetv2_x_obj2coco.yml) | []() |

**Notes:**
- `AP` is evaluated on *MSCOCO val2017* dataset.
- `FPS` is evaluated on a single T4 GPU with $batch\\_size = 1$, $fp16$, and $TensorRT==10.4.0$.
- `COCO+Objects365` in the table means finetuned model on `COCO` using pretrained weights trained on `Objects365`.
<!-- - `Stage 1`: AP<sup>val</sup> before tuning off advanced augmentations in the final few epochs (Objects365 AP<sup>val</sup> if dataset is `COCO+365`). \
These ckpts offering better generalization.
- `Stage 2`: Best AP<sup>val</sup> after disabling advanced augmentations in the final few epochs. (COCO AP<sup>val</sup> if dataset is `COCO+365`) -->

## Quick start

<details open>
<summary> Setup </summary>
  
```shell

pip install -r requirements.txt
```

</details>


<details>
  
<summary> COCO2017 dataset </summary>

1. Download COCO2017 from [OpenDataLab](https://opendatalab.com/OpenDataLab/COCO_2017). 
1. Modify paths in [coco_detection.yml](./configs/dataset/coco_detection.yml)

    ```yaml
    train_dataloader: 
        img_folder: /data/COCO2017/train2017/
        ann_file: /data/COCO2017/annotations/instances_train2017.json
    val_dataloader:
        img_folder: /data/COCO2017/val2017/
        ann_file: /data/COCO2017/annotations/instances_val2017.json
    ```
      
</details>

<details>
<summary> Objects365 dataset </summary>

1. Download Objects365 from [OpenDataLab](https://opendatalab.com/OpenDataLab/Objects365). 

2. Set the Base Directory:
```shell
export BASE_DIR=/data/Objects365/data
```

3. Create a New Directory to Store Images from the Validation Set:
```shell
mkdir -p ${BASE_DIR}/train/images_from_val
```

3. Copy the v1 and v2 folders from the val directory into the train/images_from_val directory
```shell
cp -r ${BASE_DIR}/val/images/v1 ${BASE_DIR}/train/images_from_val/
cp -r ${BASE_DIR}/val/images/v2 ${BASE_DIR}/train/images_from_val/
```

4. Directory structure after copying:

```shell
${BASE_DIR}/train
├── images_from_val
├── images
│   ├── v1
│   │   ├── patch0
│   │   │   ├── 000000000.jpg
│   │   │   ├── 000000001.jpg
│   │   │   └── ... (more images)
│   ├── v2
│   │   ├── patchx
│   │   │   ├── 000000000.jpg
│   │   │   ├── 000000001.jpg
│   │   │   └── ... (more images)
├── zhiyuan_objv2_train.json
```

```shell
${BASE_DIR}/val
├── images
│   ├── v1
│   │   ├── patch0
│   │   │   ├── 000000000.jpg
│   │   │   └── ... (more images)
│   ├── v2
│   │   ├── patchx
│   │   │   ├── 000000000.jpg
│   │   │   └── ... (more images)
├── zhiyuan_objv2_val.json
```

5. Run remap_obj365.py to merge a subset of the validation set into the training set. Specifically, this script moves samples with indices between 5000 and 800000 from the validation set to the training set.
```shell
python tools/remap_obj365.py --base_dir ${BASE_DIR}
```


6. Run the resize_obj365.py script to resize any images in the dataset where the maximum edge length exceeds 640 pixels. Use the updated JSON file generated in Step 2 to process the sample data. Ensure that you resize images in both the train and val datasets to maintain consistency.
```shell
python tools/resize_obj365.py --base_dir ${BASE_DIR}
```

7. Modify paths in [obj365_detection.yml](./configs/dataset/obj365_detection.yml)

    ```yaml
    train_dataloader: 
        img_folder: /data/Objects365/data/train
        ann_file: /data/Objects365/data/train/new_zhiyuan_objv2_train_resized.json
    val_dataloader:
        img_folder:  /data/Objects365/data/val/
        ann_file:  /data/Objects365/data/val/new_zhiyuan_objv2_val_resized.json
    ```


</details>

<details>
<summary>Custom dataset</summary>

To train on your custom dataset, you need to organize it in the COCO format. Follow the steps below to prepare your dataset:

1. **Set `remap_mscoco_category` to `False`:**

    This prevents the automatic remapping of category IDs to match the MSCOCO categories.

    ```yaml
    remap_mscoco_category: False
    ```

2. **Organize Images:**

    Structure your dataset directories as follows:

    ```shell
    dataset/
    ├── images/
    │   ├── train/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   ├── val/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    └── annotations/
        ├── instances_train.json
        ├── instances_val.json
        └── ...
    ```

    - **`images/train/`**: Contains all training images.
    - **`images/val/`**: Contains all validation images.
    - **`annotations/`**: Contains COCO-formatted annotation files.

3. **Convert Annotations to COCO Format:**

    If your annotations are not already in COCO format, you'll need to convert them. You can use the following Python script as a reference or utilize existing tools:

    ```python
    import json

    def convert_to_coco(input_annotations, output_annotations):
        # Implement conversion logic here
        pass

    if __name__ == "__main__":
        convert_to_coco('path/to/your_annotations.json', 'dataset/annotations/instances_train.json')
    ```

4. **Update Configuration Files:**

    Modify your [custom_detection.yml](./configs/dataset/custom_detection.yml).

    ```yaml
    task: detection
    
    evaluator:
      type: CocoEvaluator
      iou_types: ['bbox', ]

    num_classes: 777 # your dataset classes
    remap_mscoco_category: False
    
    train_dataloader: 
      type: DataLoader
      dataset: 
        type: CocoDetection
        img_folder: /data/yourdataset/train
        ann_file: /data/yourdataset/train/train.json
        return_masks: False
        transforms:
          type: Compose
          ops: ~
      shuffle: True
      num_workers: 4
      drop_last: True 
      collate_fn:
        type: BatchImageCollateFuncion
    
    val_dataloader:
      type: DataLoader
      dataset: 
        type: CocoDetection
        img_folder: /data/yourdataset/val
        ann_file: /data/yourdataset/val/ann.json
        return_masks: False
        transforms:
          type: Compose
          ops: ~ 
      shuffle: False
      num_workers: 4
      drop_last: False
      collate_fn:
        type: BatchImageCollateFuncion
    ```

</details>

## Usage
<details>
<summary> COCO2017 </summary>

<!-- <summary>1. Training </summary> -->
1. Set Model:
```shell
export model=l
```

2. Training
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=777 --nproc_per_node=4 train.py -c configs/dfine/dfine_hgnetv2_${model}_coco.yml --use-amp --seed=0
```

<!-- <summary>2. Testing </summary> -->
3. Testing
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=777 --nproc_per_node=4 train.py -c configs/dfine/dfine_hgnetv2_${model}_coco.yml -r model.pth --test-only
```

<!-- <summary>3. Tuning </summary> -->
4. Tuning
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=777 --nproc_per_node=4 train.py -c configs/dfine/dfine_hgnetv2_${model}_coco.yml -t model.pth --use-amp --seed=0
```
</details>


<details>
<summary> Objects to COCO </summary>

1. Set Model:
```shell
export model=l
```

2. Training on Objects365
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=777 --nproc_per_node=4 train.py -c configs/dfine/objects365/dfine_hgnetv2_${model}_obj365.yml --use-amp --seed=0
```

3. Turning on COCO2017
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=777 --nproc_per_node=4 train.py -c configs/dfine/objects365/dfine_hgnetv2_${model}_obj2coco.yml --use-amp --seed=0 -t model.pth
```

<!-- <summary>2. Testing </summary> -->
4. Testing
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=777 --nproc_per_node=4 train.py -c configs/dfine/dfine_hgnetv2_${model}_coco.yml -r model.pth --test-only
```
</details>


<details>
<summary> Custom dataset </summary>

1. Set Model:
```shell
export model=l
```

2. Training on Custom dataset
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=777 --nproc_per_node=4 train.py -c configs/dfine/custom/dfine_hgnetv2_${model}_custom.yml --use-amp --seed=0
```
<!-- <summary>2. Testing </summary> -->
3. Testing
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=777 --nproc_per_node=4 train.py -c configs/dfine/custom/dfine_hgnetv2_${model}_custom.yml -r model.pth --test-only
```
</details>


<details>
<summary> Deployment </summary>

<!-- <summary>4. Export onnx </summary> -->
1. Setup:
```shell
export model=l
pip install onnx onnxsim
```

2. Export onnx and tensorrt
```shell
python tools/export_onnx.py --check -c configs/dfine/dfine_hgnetv2_${model}_coco.yml -r model.pth
```

3. Export [tensorrt](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)
```shell
trtexec --onnx="model.onnx" --saveEngine="model.engine" --fp16
```

</details>

<details>
<summary> Inference / Benchmark / Visualization </summary>


1. Setup:
```shell
export model=l
pip install -r benchmark/requirements.txt
```


<!-- <summary>5. Inference </summary> -->
2. Inference (onnxruntime / tensorrt / torch)
```shell
python benchmark/inference/onnx_inf.py --onnx-file model.onnx --im-file image.jpg
python benchmark/inference/trt_inf.py --trt-file model.trt --im-file image.jpg
python benchmark/inference/torch_inf.py -c configs/dfine/dfine_hgnetv2_${model}_coco.yml -r model.pth --im-file image.jpg --device cuda:0
```

<!-- <summary>6. Benchmark </summary> -->
3. Benchmark (Params. / GFLOPs / Latency)
```shell
python benchmark/get_info.py -c configs/dfine/dfine_hgnetv2_${model}_coco.yml
python benchmark/trt_benchmark.py --COCO_dir path/to/COCO2017 --engine_dir model.engine
```

4. Voxel51 Fiftyone Visualization ([fiftyone](https://github.com/voxel51/fiftyone))
```shell
pip install fiftyone
python tools/fiftyone.py -c configs/dfine/dfine_hgnetv2_${model}_coco.yml -r model.pth
```
</details>



## Citation
If you use `D-FINE` in your work, please use the following BibTeX entries:

<summary> bibtex </summary>

```latex

```
</details>

## Acknowledgement
Our work is built upon [RT-DETR](https://github.com/lyuwenyu/RT-DETR).
Thanks to the inspirations from [RT-DETR](https://github.com/lyuwenyu/RT-DETR), [GFocal](https://github.com/implus/GFocal), [LD](https://github.com/HikariTJU/LD), and [YOLOv9](https://github.com/WongKinYiu/yolov9).

✨ Feel free to contribute and reach out if you have any questions! ✨