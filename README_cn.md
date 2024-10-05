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

## 🔍 探索D-FINE背后的关键创新
[English](README.md) | 简体中文
<details open>
<summary> 简介 </summary>

## D-FINE：重新定义目标检测中的回归任务

D-FINE重新定义了基于DETR的目标检测器中的回归任务。

**与传统方法不同，我们的FDR方法将检测框的生成过程分解为两个关键步骤：**

1. **初始框预测**：与传统方法类似，首先生成初始边界框。
2. **精细分布优化**：模型解码层迭代地对四组概率分布函数进行逐层迭代优化。通过这些分布对初始边界框地上下左右边缘进行细微调节或较大调整。

### FDR的主要优势：
1. **简化的监督**：在优化最终框的同时，可以用标签和预测结果之间的残差作为这些概率分布函数的优化目标。这使每个解码层能够更有效地集中解决其当前面临的特定定位误差，随着层数加深，其监督也变得越来越简单，从而简化了整体优化过程。

2. **复杂场景下的鲁棒性**：这些概率分布本质上代表了对每个边界“微调”的自信程度。这使系统能够独立建模每个边界在各个阶段的不确定性，从而在遮挡、运动模糊和低光照等复杂的实际场景下表现出更强的鲁棒性，相比直接回归四个固定值要更为稳健。

   
4. **灵活的优化机制**：概率分布通过加权求和转化为最终的边界框偏移值。精心设计的加权函数确保在初始框准确时进行细微调整，而在必要时则提供较大的修正。

   
6. **研究潜力与可扩展性**：通过将回归任务转变为类似分类任务的概率分布预测问题，这一框架不仅提高了与其他任务的兼容性，它还使得目标检测模型可以受益于知识蒸馏、多任务学习和分布建模等更多领域的创新，为未来的研究打开了新的大门。


<!-- 插入解释FDR过程的图 -->
<p align="center">
    <img src="https://github.com/Peterande/storage/blob/main/fdr.png" alt="精细分布优化过程" width="777">
</p>

## GO-LSD：将FDR扩展到知识蒸馏

GO-LSD（全局最优定位自蒸馏）基于FDR，通过在网络层间实现定位知识蒸馏，进一步扩展了FDR的能力。随着FDR的引入，回归任务现在变成了概率分布预测，这带来了两个主要优势：

1. **知识传递**：概率分布天然携带定位知识，可以通过计算KLD损失从深层传递到浅层。这是传统固定框表示（狄拉克δ函数）无法实现的。
   
3. **一致的优化目标**：由于每一层都共享一个共同目标：减少初始边界框与真实边界框之间的残差；因此最后一层生成的精确概率分布可以通过蒸馏引导前几层。这产生了一种双赢的协同效应：随着训练的进行，最后一层的预测变得越来越准确，其生成的软标签更好地帮助前几层提高预测准确性。反过来，前几层学会更快地定位到准确位置，简化了深层的优化任务，进一步提高了整体准确性。


<!-- 插入解释GO-LSD过程的图 -->
<p align="center">
    <img src="https://github.com/Peterande/storage/blob/main/go_lsd.png" alt="GO-LSD过程" width="777">
</p>

### D-FINE预测的可视化

以下可视化展示了D-FINE在各种复杂检测场景中的预测结果。这些场景包括遮挡、低光照、运动模糊、景深效果和密集场景。尽管面对这些挑战，D-FINE依然能够产生准确的定位结果。

<!-- 插入复杂场景中的预测可视化图 -->
<p align="center">
    <img src="https://github.com/Peterande/storage/blob/main/hard_case.png" alt="D-FINE在复杂场景中的预测" width="777">
</p>

### FDR和GO-LSD会带来更多的推理成本吗？
并不会，FDR和原始的预测几乎没有在速度、参数量和计算复杂度上的任何区别，完全是无感替换。

### FDR和GO-LSD会带来更多的训练成本吗？
训练成本的增加主要来源于如何生成分布的标签。我们已经对该过程进行了优化，将训练时长和显存占用控制在了6%和2%，几乎无感。

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