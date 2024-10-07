<!--# [D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement](https://arxiv.org/abs/xxxxxx) -->
[English](README.md) | 简体中文 | [博客](src/zoo/dfine/blog_cn.md)

<h2 align="center">
  D-FINE: Redefine Regression Task of DETRs as Fine&#8209;grained&nbsp;Distribution&nbsp;Refinement
</h2>

<p align="center">
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

<p align="center">
中国科学技术大学
</p>
<!-- <table><tr>
<td><img src=https://github.com/Peterande/storage/blob/master/latency.png border=0 width=333></td>
<td><img src=https://github.com/Peterande/storage/blob/master/params.png border=0 width=333></td>
<td><img src=https://github.com/Peterande/storage/blob/master/flops.png border=0 width=333></td>
</tr></table> -->

<table><tr>
<td><img src=https://github.com/Peterande/storage/blob/master/stats_padded.png border=0 width=1000></td>
</tr></table>


## 🚀 Updates
- [x] **\[2024.10.3\]** 发布 D-FINE 系列。
<!-- - 🔜 **\[Next\]** Release D-FINE series pretrained on Objects365. -->


## 模型库

### 基础模型
| 模型 | 数据集 | AP<sup>val</sup> | 参数量 | FPS | GFLOPs | 配置 | 检查点 | 日志 |
| :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
**D-FINE-S** | COCO | **48.5** |  10M | 287 | 25 | [cfg](./configs/dfine/dfine_hgnetv2_s_coco.yml) | [48.5](https://github.com/Peterande/storage/releases/download/dfinev1/dfine_s_coco.pth) |
**D-FINE-M** | COCO | **52.3** |  19M | 180 | 57 | [cfg](./configs/dfine/dfine_hgnetv2_m_coco.yml) | [52.3](https://github.com/Peterande/storage/releases/download/dfinev1/dfine_m_coco.pth) |
**D-FINE-L** | COCO | **54.0** |  31M | 129 | 91 | [cfg](./configs/dfine/dfine_hgnetv2_l_coco.yml) | [54.0](https://github.com/Peterande/storage/releases/download/dfinev1/dfine_l_coco.pth) |
**D-FINE-X** | COCO | **55.8** |  62M | 81 | 202 | [cfg](./configs/dfine/dfine_hgnetv2_x_coco.yml) | [55.8](https://github.com/Peterande/storage/releases/download/dfinev1/dfine_x_coco.pth) |
**D-FINE-S** | COCO+Objects365 | **50.3** |  10M | 287 | 25 | [cfg](./configs/dfine/objects365/dfine_hgnetv2_s_obj2coco.yml) | []() |
**D-FINE-M** | COCO+Objects365 | **55.0** |  19M | 180 | 57 | [cfg](./configs/dfine/objects365/dfine_hgnetv2_m_obj2coco.yml) | []() |
**D-FINE-L** | COCO+Objects365 | **56.9** |  31M | 129 | 91 | [cfg](./configs/dfine/objects365/dfine_hgnetv2_l_obj2coco.yml) | []() |
**D-FINE-X** | COCO+Objects365 | **59.0** |  62M | 81 | 202 | [cfg](./configs/dfine/objects365/dfine_hgnetv2_x_obj2coco.yml) | []() |

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

### 设置
  
```shell

pip install -r requirements.txt
```

</details>



### 数据集准备


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

3. 解压并整理目录结构如下：

```shell
${BASE_DIR}/train
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


4. 创建一个新目录来存储验证集中的图像：
```shell
mkdir -p ${BASE_DIR}/train/images_from_val
```

5. 将 val 目录中的 v1 和 v2 文件夹复制到 train/images_from_val 目录中
```shell
cp -r ${BASE_DIR}/val/images/v1 ${BASE_DIR}/train/images_from_val/
cp -r ${BASE_DIR}/val/images/v2 ${BASE_DIR}/train/images_from_val/
```


6. 运行 remap_obj365.py 将验证集中的部分样本合并到训练集中。具体来说，该脚本将索引在 5000 到 800000 之间的样本从验证集移动到训练集。
```shell
python tools/remap_obj365.py --base_dir ${BASE_DIR}
```


7. 运行 resize_obj365.py 脚本，将数据集中任何最大边长超过 640 像素的图像进行大小调整。使用步骤 5 中生成的更新后的 JSON 文件处理样本数据。
```shell
python tools/resize_obj365.py --base_dir ${BASE_DIR}
```

8. 修改 [obj365_detection.yml](./configs/dataset/obj365_detection.yml) 中的路径。

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

要在你的自定义数据集上训练，你需要将其组织为 COCO 格式。请按照以下步骤准备你的数据集：

1. **将 `remap_mscoco_category` 设置为 `False`:**

    这可以防止类别 ID 自动映射以匹配 MSCOCO 类别。

    ```yaml
    remap_mscoco_category: False
    ```

2. **组织图像：**

    按以下结构组织你的数据集目录：

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

    如果你的注释尚未为 COCO 格式，你需要进行转换。你可以参考以下 Python 脚本或使用现有工具：

    ```python
    import json

    def convert_to_coco(input_annotations, output_annotations):
        # Implement conversion logic here
        pass

    if __name__ == "__main__":
        convert_to_coco('path/to/your_annotations.json', 'dataset/annotations/instances_train.json')
    ```

4. **更新配置文件：**

    修改你的 [custom_detection.yml](./configs/dataset/custom_detection.yml)。

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
<summary> 自定义批次大小 </summary>

例如，如果你想在训练 D-FINE-L 时将 COCO2017 的总批次大小增加一倍，请按照以下步骤操作：

1. **修改你的 [dataloader.yml](./configs/dfine/include/dataloader.yml)**，增加 `total_batch_size`：

    ```yaml
    train_dataloader: 
        total_batch_size: 64  # 原来是 32，现在增加了一倍
    ```

2. **修改你的 [dfine_hgnetv2_l_coco.yml](./configs/dfine/dfine_hgnetv2_l_coco.yml)**。

    ```yaml
    optimizer:
    type: AdamW
    params: 
        - 
        params: '^(?=.*backbone)(?!.*norm|bn).*$'
        lr: 0.000025  # 翻倍，线性缩放原则
        - 
        params: '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn)).*$'
        weight_decay: 0.

    lr: 0.0005  # 翻倍，线性缩放原则
    betas: [0.9, 0.999]
    weight_decay: 0.0000625  # 减半，但可能需要网格搜索找到最优值

    ema:  # 添加 EMA 设置
        decay: 0.9998  # 根据 1 - (1 - decay) * 2 调整
        warmups: 500  # 减半

    lr_warmup_scheduler:
        warmup_duration: 250  # 减半
    ```

</details>



## 工具

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
<summary> 推理 </summary>


1. 设置
```shell
export model=l
pip install -r tools/inference/requirements.txt
```


<!-- <summary>5. Inference </summary> -->
2. 推理 (onnxruntime / tensorrt / torch)
```shell
python tools/inference/onnx_inf.py --onnx-file model.onnx --im-file image.jpg
python tools/inference/trt_inf.py --trt-file model.trt --im-file image.jpg
python tools/inference/torch_inf.py -c configs/dfine/dfine_hgnetv2_${model}_coco.yml -r model.pth --im-file image.jpg --device cuda:0
```
</details>

<details>
<summary> 基准测试  </summary>

1. 设置
```shell
export model=l
pip install -r tools/benchmark/requirements.txt
```

<!-- <summary>6. Benchmark </summary> -->
2. 模型 FLOPs、MACs、参数量
```shell
python tools/benchmark/get_info.py -c configs/dfine/dfine_hgnetv2_${model}_coco.yml
```

2. TensorRT 延迟
```shell
python tools/benchmark/trt_benchmark.py --COCO_dir path/to/COCO2017 --engine_dir model.engine
```
</details>

<details>
<summary> Voxel51 Fiftyone 可视化  </summary>

1. 设置
```shell
export model=l
pip install fiftyone
```
4. Voxel51 Fiftyone 可视化 ([fiftyone](https://github.com/voxel51/fiftyone))
```shell
python tools/visualization/fiftyone_vis.py -c configs/dfine/dfine_hgnetv2_${model}_coco.yml -r model.pth
```
</details>

## 引用
如果你在工作中使用了 `D-FINE` 或其方法，请引用以下 BibTeX 条目：
<details open>
<summary> bibtex </summary>

```latex

```
</details>

## 致谢
我们的工作基于 [RT-DETR](https://github.com/lyuwenyu/RT-DETR)。
感谢 [RT-DETR](https://github.com/lyuwenyu/RT-DETR), [GFocal](https://github.com/implus/GFocal), [LD](https://github.com/HikariTJU/LD), 和 [YOLOv9](https://github.com/WongKinYiu/yolov9) 的启发。

✨ 欢迎贡献并在有任何问题时联系我！ ✨