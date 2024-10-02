<!-- ## 🚀 Updates
- \[2024.10.7\] Release D-FINE series. -->


This is the official implementation of papers 
- [D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement](https://arxiv.org/abs/xxxxxx)

<summary>Fig</summary>

<table><tr>
<td><img src=https://github.com/Peterande/storage/blob/main/latency.png border=0 width=333></td>
<td><img src=https://github.com/Peterande/storage/blob/main/params.png border=0 width=333></td>
<td><img src=https://github.com/Peterande/storage/blob/main/flops.png border=0 width=333></td>
</tr></table>

## Quick start

<details open>
<summary>Setup</summary>

```shell

pip install -r requirements.txt
```

## Model Zoo

### Base models
| Model | Dataset | AP<sup>val</sup> | #Params | FPS | GFLOPs | config | checkpoint |
| :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: |
**D-FINE-S** | COCO | **48.5** |  10M | 287 | 25 | [config](./configs/dfine/dfine_hgnetv2_s_coco.yml) | [48.5](xxx.pth)
**D-FINE-M** | COCO | **52.3** |  19M | 180 | 57 | [config](./configs/dfine/dfine_hgnetv2_m_coco.yml) | [52.3](xxx.pth)
**D-FINE-L** | COCO | **54.0** |  31M | 129 | 91 | [config](./configs/dfine/dfine_hgnetv2_l_coco.yml) | [54.0](xxx.pth)
**D-FINE-X** | COCO | **55.8** |  62M | 81 | 202 | [config](./configs/dfine/dfine_hgnetv2_x_coco.yml) | [55.8](xxx.pth)
<!-- **D-FINE-S** | COCO+365 | **48.5** |  10M | 287 | 25 | [config](./configs/dfine/objects365/dfine_hgnetv2_s_obj2coco.yml) | [48.1](xxx.pth) | [48.5](xxx.pth)
**D-FINE-M** | COCO+365 | **52.3** |  19M | 180 | 57 | [config](./configs/dfine/objects365/dfine_hgnetv2_m_obj2coco.yml) | [52.1](xxx.pth) | [52.3](xxx.pth)
**D-FINE-L** | COCO+365 | **53.9** |  31M | 129 | 91 | [config](./configs/dfine/objects365/dfine_hgnetv2_l_obj2coco.yml) | [53.8](xxx.pth) | [53.9](xxx.pth)
**D-FINE-X** | COCO+365 | **55.8** |  62M | 81 | 202 | [config](./configs/dfine/objects365/dfine_hgnetv2_x_obj2coco.yml) | [55.6](xxx.pth) | [55.8](xxx.pth) -->

**Notes:**
- `AP` is evaluated on *MSCOCO val2017* dataset.
- `FPS` is evaluated on a single T4 GPU with $batch\\_size = 1$, $fp16$, and $TensorRT==10.4.0$.
- `COCO+365` in the table means finetuned model on `COCO` using pretrained weights trained on `Objects365`.
<!-- - `Stage 1`: AP<sup>val</sup> before tuning off advanced augmentations in the final few epochs (Objects365 AP<sup>val</sup> if dataset is `COCO+365`). \
These ckpts offering better generalization.
- `Stage 2`: Best AP<sup>val</sup> after disabling advanced augmentations in the final few epochs. (COCO AP<sup>val</sup> if dataset is `COCO+365`) -->

## Usage
<details>
<summary> COCO </summary>

<!-- <summary>1. Training </summary> -->
```shell
model=l 
```
1. Training
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=777 --nproc_per_node=4 train.py -c configs/dfine/dfine_hgnetv2_${model}_coco.yml --use-amp --seed=0
```

<!-- <summary>2. Testing </summary> -->
2. Testing
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=777 --nproc_per_node=4 train.py -c configs/dfine/dfine_hgnetv2_${model}_coco.yml -r model.pth --test-only
```

<!-- <summary>3. Tuning </summary> -->
3. Tuning
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=777 --nproc_per_node=4 tools/train.py -c configs/dfine/dfine_hgnetv2_${model}_coco.yml -t model.pth --use-amp --seed=0
```
</details>

<!-- <details>
<summary> Objects365 to COCO </summary>
1. Download Objects365 from [OpenDataLab](https://opendatalab.com/OpenDataLab/Objects365/cli/main).
After decompressing the dataset, make sure to copy the contents of val/v1 and val/v2 into train/images_from_val to prepare for the next step.

```shell
/data/username/Objects365/data/train
├── images_from_val
├── images
│   ├── v1
│   │   ├── patch0
│   │   │   ├── 000000000.jpg
│   ├── v2
│   │   ├── patchx
│   │   │   ├── 000000000.jpg
├── /data/Objects365/data/train/zhiyuan_objv2_train.json
```

```shell
/data/username/Objects365/data/val
├── images
│   ├── v1
│   │   ├── patch0
│   │   │   ├── 000000000.jpg
│   ├── v2
│   │   ├── patchx
│   │   │   ├── 000000000.jpg
├── /data/Objects365/data/val/zhiyuan_objv2_val.json
```

2. Once all the files are decompressed and organized, run the remap_obj365.py script. This script will merge samples with indices between 5000 and 800000 from the validation set into the training set.
```shell
python tools/remap_obj365.py
```


3. Next, run the resize_obj365.py script to resize the dataset images that have a maximum edge length greater than 640 pixels. Make sure to use the updated JSON file created in Step 2 to read the sample data. Resize the samples in both the train and val datasets to ensure consistency.
```shell
python tools/resize_obj365.py
```

4. Training on Objects365
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port=777 --nproc_per_node=8 tools/train.py -c configs/dfine/objects365/dfine_hgnetv2_${model}_obj365.yml --use-amp --seed=0
```

5. Turning on COCO
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port=777 --nproc_per_node=8 tools/train.py -c configs/dfine/objects365/dfine_hgnetv2_${model}_obj2coco.yml --use-amp --seed=0 -t model.pth
```
</details> -->

<details>
<summary> Deployment and Benchmark </summary>

<!-- <summary>4. Export onnx </summary> -->
```shell
model=l 
```
1. Export onnx and tensorrt
```shell
python tools/export_onnx.py -c configs/dfine/dfine_hgnetv2_${model}_coco.yml -r model.pth --check
trtexec --onnx="./model.onnx" --saveEngine="./model.engine" --fp16
```

<!-- <summary>5. Inference </summary> -->
2. Inference

Support torch, onnxruntime, tensorrt and openvino, see details in *benchmark/inference*
```shell
python benchmark/inference/onnx_inf.py --onnx-file model.onnx --im-file image.jpg
python benchmark/inference/trt_inf.py --trt-file model.trt --im-file image.jpg
python benchmark/inference/torch_inf.py -c configs/dfine/dfine_hgnetv2_${model}_coco.yml -r model.pth --im-file image.jpg --device cuda:0
```

<!-- <summary>6. Benchmark </summary> -->
3. Benchmark (Params. / GFLOPs / Latency)
```shell
pip install -r benchmark/requirements.txt
python benchmark/get_info.py -c configs/dfine/dfine_hgnetv2_${model}_coco.yml
python benchmark/TRT/trt_benchmark.py --COCO_dir path/to/COCO2017 --engine_dir model.engine
```

</details>



## Citation
If you use `DFINE` in your work, please use the following BibTeX entries:

<details>
<summary> bibtex </summary>

```latex

```
</details>
