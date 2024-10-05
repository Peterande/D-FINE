<!--# [D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement](https://arxiv.org/abs/xxxxxx) -->
[English](README.md) | 简体中文

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
    📄 这是该文章的官方实现:
    <br>
    <a href="https://arxiv.org/abs/xxxxxx">D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement</a>
</p>


<p align="center">
彭岩松，李和倍，吴沛熹，张越一，孙晓艳，吴枫
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



<p align="center">
    <img src="https://github.com/Peterande/storage/blob/main/fdr.png" alt="精细分布优化过程" width="777">
</p>

## GO-LSD：将FDR扩展到知识蒸馏

GO-LSD（全局最优定位自蒸馏）基于FDR，通过在网络层间实现定位知识蒸馏，进一步扩展了FDR的能力。随着FDR的引入，回归任务现在变成了概率分布预测，这带来了两个主要优势：

1. **知识传递**：概率分布天然携带定位知识，可以通过计算KLD损失从深层传递到浅层。这是传统固定框表示（狄拉克δ函数）无法实现的。
   
3. **一致的优化目标**：由于每一层都共享一个共同目标：减少初始边界框与真实边界框之间的残差；因此最后一层生成的精确概率分布可以通过蒸馏引导前几层。这产生了一种双赢的协同效应：随着训练的进行，最后一层的预测变得越来越准确，其生成的软标签更好地帮助前几层提高预测准确性。反过来，前几层学会更快地定位到准确位置，简化了深层的优化任务，进一步提高了整体准确性。



<p align="center">
    <img src="https://github.com/Peterande/storage/blob/main/go_lsd.png" alt="GO-LSD过程" width="777">
</p>

### FDR和GO-LSD会带来更多的推理成本吗？
并不会，FDR和原始的预测几乎没有在速度、参数量和计算复杂度上的任何区别，完全是无感替换。

### FDR和GO-LSD会带来更多的训练成本吗？
训练成本的增加主要来源于如何生成分布的标签。我们已经对该过程进行了优化，将训练时长和显存占用控制在了6%和2%，几乎无感。

### D-FINE预测的可视化

以下可视化展示了D-FINE在各种复杂检测场景中的预测结果。这些场景包括遮挡、低光照、运动模糊、景深效果和密集场景。尽管面对这些挑战，D-FINE依然能够产生准确的定位结果。


<p align="center">
    <img src="https://github.com/Peterande/storage/blob/main/hard_case.png" alt="D-FINE在复杂场景中的预测" width="777">
</p>


</details>

## Model Zoo

### 模型库
| 模型 | 数据集 | AP<sup>val</sup> | 参数量 | FPS | GFLOPs | 配置 | 检查点 | 日志 |
| :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
**D-FINE-S** | COCO | **48.5** |  10M | 287 | 25 | [config](./configs/dfine/dfine_hgnetv2_s_coco.yml) | [48.5](https://github.com/Peterande/storage/releases/download/dfinev1/dfine_s_coco.pth) |
**D-FINE-M** | COCO | **52.3** |  19M | 180 | 57 | [config](./configs/dfine/dfine_hgnetv2_m_coco.yml) | [52.3](https://github.com/Peterande/storage/releases/download/dfinev1/dfine_m_coco.pth) |
**D-FINE-L** | COCO | **54.0** |  31M | 129 | 91 | [config](./configs/dfine/dfine_hgnetv2_l_coco.yml) | [54.0](https://github.com/Peterande/storage/releases/download/dfinev1/dfine_l_coco.pth) |
**D-FINE-X** | COCO | **55.8** |  62M | 81 | 202 | [config](./configs/dfine/dfine_hgnetv2_x_coco.yml) | [55.8](https://github.com/Peterande/storage/releases/download/dfinev1/dfine_x_coco.pth) |
**D-FINE-S** | COCO+Objects365 | **50.3** |  10M | 287 | 25 | [config](./configs/dfine/objects365/dfine_hgnetv2_s_obj2coco.yml) | []() |
**D-FINE-M** | COCO+Objects365 | **55.0** |  19M | 180 | 57 | [config](./configs/dfine/objects365/dfine_hgnetv2_m_obj2coco.yml) | []() |
**D-FINE-L** | COCO+Objects365 | **56.9** |  31M | 129 | 91 | [config](./configs/dfine/objects365/dfine_hgnetv2_l_obj2coco.yml) | []() |
**D-FINE-X** | COCO+Objects365 | **59.0** |  62M | 81 | 202 | [config](./configs/dfine/objects365/dfine_hgnetv2_x_obj2coco.yml) | []() |

**注意：**
- `AP` 是在 *MSCOCO val2017* 数据集上评估的。
- `FPS` 是在单张 T4 GPU 上以 $batch\\_size = 1$, $fp16$, 和 $TensorRT==10.4.0$ 评估的。
- 表中的 `COCO+Objects365` 表示使用在 `Objects365` 上预训练的权重在 `COCO` 上微调的模型。
<!-- - `Stage 1`: AP<sup>val</sup> before tuning off advanced augmentations in the final few epochs (Objects365 AP<sup>val</sup> if dataset is `COCO+365`). \
These ckpts offering better generalization.
- `Stage 2`: Best AP<sup>val</sup> after disabling advanced augmentations in the final few epochs. (COCO AP<sup>val</sup> if dataset is `COCO+365`) -->

<!-- - `Stage 1`: AP<sup>val</sup> before tuning off advanced augmentations in the final few epochs (Objects365 AP<sup>val</sup> if dataset is `COCO+365`). \
These ckpts offering better generalization.
- `Stage 2`: Best AP<sup>val</sup> after disabling advanced augmentations in the final few epochs. (COCO AP<sup>val</sup> if dataset is `COCO+365`) -->

## 快速开始

<details open>
<summary> Setup </summary>
  
```shell

pip install -r requirements.txt
```

</details>


<details>
  
<summary> COCO2017 数据集 </summary>

1. 从 [OpenDataLab](https://opendatalab.com/OpenDataLab/COCO_2017) 下载 COCO2017。 
1.修改 [coco_detection.yml](./configs/dataset/coco_detection.yml) 中的路径。

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
<summary> Objects365 数据集 </summary>

1. 从 [OpenDataLab](https://opendatalab.com/OpenDataLab/Objects365) 下载 Objects365。

2. 设置数据集的基础目录：
```shell
export BASE_DIR=/data/Objects365/data
```

3. 创建一个新目录来存储验证集中的图像：
```shell
mkdir -p ${BASE_DIR}/train/images_from_val
```

3. 将 val 目录中的 v1 和 v2 文件夹复制到 train/images_from_val 目录中
```shell
cp -r ${BASE_DIR}/val/images/v1 ${BASE_DIR}/train/images_from_val/
cp -r ${BASE_DIR}/val/images/v2 ${BASE_DIR}/train/images_from_val/
```

4. 复制后的目录结构应该如下所示：

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

5. 运行 remap_obj365.py 将验证集中的部分样本合并到训练集中。具体来说，该脚本将索引在 5000 到 800000 之间的样本从验证集移动到训练集。
```shell
python tools/remap_obj365.py --base_dir ${BASE_DIR}
```


6. 运行 resize_obj365.py 脚本，将数据集中任何最大边长超过 640 像素的图像进行大小调整。使用步骤 5 中生成的更新后的 JSON 文件处理样本数据。
```shell
python tools/resize_obj365.py --base_dir ${BASE_DIR}
```

7. 修改 [obj365_detection.yml](./configs/dataset/obj365_detection.yml) 中的路径。

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
<summary>自定义数据集</summary>

要在您的自定义数据集上训练，您需要将其组织为 COCO 格式。请按照以下步骤准备您的数据集：

1. **将 `remap_mscoco_category` 设置为 `False`:**

    这可以防止类别 ID 自动映射以匹配 MSCOCO 类别。

    ```yaml
    remap_mscoco_category: False
    ```

2. **组织图像：**

    按以下结构组织您的数据集目录：

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

    - **`images/train/`**: 包含所有训练图像。
    - **`images/val/`**: 包含所有验证图像。
    - **`annotations/`**: 包含 COCO 格式的注释文件。

3. **将注释转换为 COCO 格式：**

    如果您的注释尚未为 COCO 格式，您需要进行转换。您可以参考以下 Python 脚本或使用现有工具：

    ```python
    import json

    def convert_to_coco(input_annotations, output_annotations):
        # Implement conversion logic here
        pass

    if __name__ == "__main__":
        convert_to_coco('path/to/your_annotations.json', 'dataset/annotations/instances_train.json')
    ```

4. **更新配置文件：**

    修改您的 [custom_detection.yml](./configs/dataset/custom_detection.yml)。

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

## 使用方法
<details>
<summary> COCO2017 </summary>

<!-- <summary>1. Training </summary> -->
1. 设置模型
```shell
export model=l
```

2. 训练
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=777 --nproc_per_node=4 train.py -c configs/dfine/dfine_hgnetv2_${model}_coco.yml --use-amp --seed=0
```

<!-- <summary>2. Testing </summary> -->
3. 测试
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=777 --nproc_per_node=4 train.py -c configs/dfine/dfine_hgnetv2_${model}_coco.yml -r model.pth --test-only
```

<!-- <summary>3. Tuning </summary> -->
4. 微调
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=777 --nproc_per_node=4 train.py -c configs/dfine/dfine_hgnetv2_${model}_coco.yml -t model.pth --use-amp --seed=0
```
</details>


<details>
<summary> 在 Objects365 上训练，在COCO2017上微调 </summary>

1. 设置模型
```shell
export model=l
```

2. 在 Objects365 上训练
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=777 --nproc_per_node=4 train.py -c configs/dfine/objects365/dfine_hgnetv2_${model}_obj365.yml --use-amp --seed=0
```

3. 在 COCO2017 上微调
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=777 --nproc_per_node=4 train.py -c configs/dfine/objects365/dfine_hgnetv2_${model}_obj2coco.yml --use-amp --seed=0 -t model.pth
```

<!-- <summary>2. Testing </summary> -->
4. 测试
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=777 --nproc_per_node=4 train.py -c configs/dfine/dfine_hgnetv2_${model}_coco.yml -r model.pth --test-only
```
</details>


<details>
<summary> 自定义数据集 </summary>

1. 设置模型
```shell
export model=l
```

2. 在自定义数据集上训练
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=777 --nproc_per_node=4 train.py -c configs/dfine/custom/dfine_hgnetv2_${model}_custom.yml --use-amp --seed=0
```
<!-- <summary>2. Testing </summary> -->
3. 测试
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=777 --nproc_per_node=4 train.py -c configs/dfine/custom/dfine_hgnetv2_${model}_custom.yml -r model.pth --test-only
```
</details>


<details>
<summary> 部署 </summary>

<!-- <summary>4. Export onnx </summary> -->
1. 设置
```shell
export model=l
pip install onnx onnxsim
```

2. 导出 onnx
```shell
python tools/export_onnx.py --check -c configs/dfine/dfine_hgnetv2_${model}_coco.yml -r model.pth
```

3. 导出 [tensorrt](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)
```shell
trtexec --onnx="model.onnx" --saveEngine="model.engine" --fp16
```

</details>

<details>
<summary> 推理 / 基准测试 / 可视化 </summary>


1. 设置
```shell
export model=l
pip install -r benchmark/requirements.txt
```


<!-- <summary>5. Inference </summary> -->
2. 推理 (onnxruntime / tensorrt / torch)
```shell
python benchmark/inference/onnx_inf.py --onnx-file model.onnx --im-file image.jpg
python benchmark/inference/trt_inf.py --trt-file model.trt --im-file image.jpg
python benchmark/inference/torch_inf.py -c configs/dfine/dfine_hgnetv2_${model}_coco.yml -r model.pth --im-file image.jpg --device cuda:0
```

<!-- <summary>6. Benchmark </summary> -->
3. 基准测试 (参数量 / GFLOPs / 延迟)
```shell
python benchmark/get_info.py -c configs/dfine/dfine_hgnetv2_${model}_coco.yml
python benchmark/trt_benchmark.py --COCO_dir path/to/COCO2017 --engine_dir model.engine
```

4. Voxel51 Fiftyone 可视化 ([fiftyone](https://github.com/voxel51/fiftyone))
```shell
pip install fiftyone
python tools/fiftyone.py -c configs/dfine/dfine_hgnetv2_${model}_coco.yml -r model.pth
```
</details>



## Citation
如果您在工作中使用了 `D-FINE`，请使用以下 BibTeX 条目：

<summary> bibtex </summary>

```latex

```
</details>

## 致谢
我们的工作基于 [RT-DETR](https://github.com/lyuwenyu/RT-DETR)。
感谢 [RT-DETR](https://github.com/lyuwenyu/RT-DETR), [GFocal](https://github.com/implus/GFocal), [LD](https://github.com/HikariTJU/LD), 和 [YOLOv9](https://github.com/WongKinYiu/yolov9) 的启发。

✨ 欢迎贡献并在有任何问题时联系我！ ✨