<!--# [D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement](https://arxiv.org/abs/xxxxxx) -->
<h2 align="center">D-FINE: Redefine Regression Task of DETRs as&nbsp;Fine-grained&nbsp;Distribution Refinement</h2>


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

This is the official implementation of paper:
- [D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement](https://arxiv.org/abs/xxxxxx)

<table><tr>
<td><img src=https://github.com/Peterande/storage/blob/main/latency.png border=0 width=333></td>
<td><img src=https://github.com/Peterande/storage/blob/main/params.png border=0 width=333></td>
<td><img src=https://github.com/Peterande/storage/blob/main/flops.png border=0 width=333></td>
</tr></table>

## 🚀 Updates
- \[2024.10.3\] Release D-FINE series.
- \[Next\] Release D-FINE series pretrained on Objects365.
  
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

<details>
<summary>Setup</summary>

```shell

pip install -r requirements.txt
```

</details>

## Dataset Prepare
<details>
<summary> COCO </summary>

1. Download COCO2017 from [OpenDataLab](https://opendatalab.com/OpenDataLab/COCO_2017). 
1. Modify paths in [coco_detection.yml](./configs/dataset/coco_detection.yml)

</details>

<details>
<summary> Objects365 </summary>

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


</details>

## Usage
<details>
<summary> COCO </summary>

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
<summary> Objects365 to COCO </summary>

1. Set Model:
```shell
export model=l
```

2. Training on Objects365
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=777 --nproc_per_node=4 train.py -c configs/dfine/objects365/dfine_hgnetv2_${model}_obj365.yml --use-amp --seed=0
```

3. Turning on COCO
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=777 --nproc_per_node=4 train.py -c configs/dfine/objects365/dfine_hgnetv2_${model}_obj2coco.yml --use-amp --seed=0 -t model.pth
```
</details>

<details>
<summary> Deployment and Benchmark </summary>

<!-- <summary>4. Export onnx </summary> -->
1. Set Model:
```shell
export model=l
```

2. Export onnx and tensorrt
```shell
python tools/export_onnx.py -c configs/dfine/dfine_hgnetv2_${model}_coco.yml -r model.pth --check
trtexec --onnx="./model.onnx" --saveEngine="./model.engine" --fp16
```

<!-- <summary>5. Inference </summary> -->
3. Inference

Support torch, onnxruntime, tensorrt and openvino, see details in *benchmark/inference*
```shell
python benchmark/inference/onnx_inf.py --onnx-file model.onnx --im-file image.jpg
python benchmark/inference/trt_inf.py --trt-file model.trt --im-file image.jpg
python benchmark/inference/torch_inf.py -c configs/dfine/dfine_hgnetv2_${model}_coco.yml -r model.pth --im-file image.jpg --device cuda:0
```

<!-- <summary>6. Benchmark </summary> -->
4. Benchmark (Params. / GFLOPs / Latency)
```shell
pip install -r benchmark/requirements.txt
python benchmark/get_info.py -c configs/dfine/dfine_hgnetv2_${model}_coco.yml
python benchmark/TRT/trt_benchmark.py --COCO_dir path/to/COCO2017 --engine_dir model.engine
```

</details>



## Citation
If you use `D-FINE` in your work, please use the following BibTeX entries:

<details>
<summary> bibtex </summary>

```latex

```
</details>
