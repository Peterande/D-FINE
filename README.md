<!-- ## 🚀 Updates
- \[2024.10.7\] Release D-FINE series. -->


This is the official implementation of papers 
- [D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement](https://arxiv.org/abs/xxxxxx)

<summary>Fig</summary>

<table><tr>
<td><img src=https://github.com/lyuwenyu/RT-DETR/assets/77494834/0ede1dc1-a854-43b6-9986-cf9090f11a61 border=0 width=333></td>
<td><img src=https://github.com/user-attachments/assets/437877e9-1d4f-4d30-85e8-aafacfa0ec56 border=0 width=333></td>
<td><img src=https://github.com/user-attachments/assets/437877e9-1d4f-4d30-85e8-aafacfa0ec56 border=0 width=333></td>
</tr></table>

## Quick start

<details open>
<summary>Setup</summary>

```shell

pip install -r requirements.txt
```

## Model Zoo

### Base models

| Model | Dataset | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | #Params(M) | FPS | GFLOPs | config | Stage 1 | Stage 2 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
**D-FINE-S** | COCO | **48.5** | **65.5** | 10 | 287 | 25 | [config](./configs/dfine/dfine_hgnetv2_s_10x_coco.yml) | [48.1](xxx.pth) | [48.5](xxx.pth)
**D-FINE-M** | COCO | **52.3** | **69.8** | 19 | 180 | 57 | [config](./configs/dfine/dfine_hgnetv2_m_10x_coco.yml) | [52.1](xxx.pth) | [52.3](xxx.pth)
**D-FINE-L** | COCO | **53.9** | **71.6** | 31 | 129 | 91 | [config](./configs/dfine/dfine_hgnetv2_l_6x_coco.yml) | [53.8](xxx.pth) | [53.9](xxx.pth)
**D-FINE-X** | COCO | **55.8** | **73.7** | 62 | 81 | 202 | [config](./configs/dfine/dfine_hgnetv2_x_6x_coco.yml) | [55.6](xxx.pth) | [55.8](xxx.pth)


**Notes:**
- `AP` is evaluated on *MSCOCO val2017* dataset.
- `FPS` is evaluated on a single T4 GPU with $batch\\_size = 1$, $fp16$, and $TensorRT==10.4.0$.
<!-- - `COCO + Objects365` in the table means finetuned model on `COCO` using pretrained weights trained on `Objects365`. -->


## Usage
<details>
<summary> Details </summary>

<!-- <summary>1. Training </summary> -->
1. Training
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=777 --nproc_per_node=4 tools/train.py -c configs/dfine/xxx_coco --use-amp --seed=0
```

<!-- <summary>2. Testing </summary> -->
2. Testing
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=777 --nproc_per_node=4 tools/train.py -c configs/dfine/xxx_coco -r path/to/checkpoint --test-only
```

<!-- <summary>3. Tuning </summary> -->
3. Tuning
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=777 --nproc_per_node=4 tools/train.py -c configs/dfine/xxx_coco -t path/to/checkpoint --use-amp --seed=0
```
</details>

<details>
<summary> Objects365 Prepare </summary>
1. Download Objects365 from [OpenDataLab](https://opendatalab.com/OpenDataLab/Objects365/cli/main).
After decompressing the dataset, make sure to copy the contents of val/v1 and val/v2 into train/images_from_val to prepare for the next step.

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

/data/username/Objects365/data/val
├── images
│   ├── v1
│   │   ├── patch0
│   │   │   ├── 000000000.jpg
│   ├── v2
│   │   ├── patchx
│   │   │   ├── 000000000.jpg
├── /data/Objects365/data/val/zhiyuan_objv2_val.json

2. Once all the files are decompressed and organized, run the remap_obj365.py script. This script will merge samples with indices between 5000 and 800000 from the validation set into the training set.
```shell
python configs/dataset/remap_obj365.py
```


3. Next, run the resize_obj365.py script to resize the dataset images that have a maximum edge length greater than 640 pixels. Make sure to use the updated JSON file created in Step 2 to read the sample data. Resize the samples in both the train and val datasets to ensure consistency.
```shell
python configs/dataset/resize_obj365.py
```

4. Training on Objects365
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port=777 --nproc_per_node=8 tools/train.py -c configs/dfine/objects365/xxx_obj365 --use-amp --seed=0
```

5. Turning on COCO
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port=777 --nproc_per_node=8 tools/train.py -c configs/dfine/objects365/xxx_obj2coco --use-amp --seed=0 -t path/to/checkpoint
```
</details>

<details>
<summary> Deployment and Benchmark </summary>
<!-- <summary>4. Export onnx </summary> -->
1. Export onnx and tensorrt
```shell
python tools/export_onnx.py -c path/to/config -r path/to/checkpoint --check
trtexec --onnx=".model.onnx" --saveEngine="./model.engine" --fp16
```

<!-- <summary>5. Inference </summary> -->
2. Inference

Support torch, onnxruntime, tensorrt and openvino, see details in *benchmark/inference*
```shell
python benchmark/inference/onnx_inf.py --onnx-file=model.onnx --im-file=xxxx
python benchmark/inference/trt_inf.py --trt-file=model.trt --im-file=xxxx
python benchmark/inference/torch_inf.py -c path/to/config -r path/to/checkpoint --im-file=xxxx --device=cuda:0
```

<!-- <summary>6. Benchmark </summary> -->
3. Benchmark (Params. / GFLOPs / Latency)
```shell
pip install -r benchmark/requirements.txt
python benchmark/get_info.py -c path/to/config
python benchmark/TRT/trt_benchmark_.py --COCO_dir path/to/COCO --engine_dir path/to/engine
```

</details>



## Citation
If you use `DFINE` in your work, please use the following BibTeX entries:

<details>
<summary> bibtex </summary>

```latex

```
</details>
