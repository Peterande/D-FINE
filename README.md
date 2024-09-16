
## Quick start

<details open>
<summary>Setup</summary>

```shell

pip install -r requirements.txt
```

## Model Zoo

### Base models

| Model | Dataset | Input Size | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | #Params(M) | FPS | config | checkpoint (Stage 1) | checkpoint (Stage 2) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |

**DFINE-X** | COCO | 640 | **55.8** | **73.7** | 62 | xxx | [config](./configs/dfine/dfine_hgnetv2_b5_6x_coco.yml) | [url](xxx.pth) | [url](xxx.pth)


**Notes:**
- `AP` is evaluated on *MSCOCO val2017* dataset.
- `FPS` is evaluated on a single T4 GPU with $batch\\_size = 1$, $fp16$, and $TensorRT>=8.5.1$.
- `COCO + Objects365` in the table means finetuned model on `COCO` using pretrained weights trained on `Objects365`.


## Usage
<details>
<summary> details </summary>

<!-- <summary>1. Training </summary> -->
1. Training
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=9909 --nproc_per_node=4 tools/train.py -c path/to/config --use-amp --seed=0 &> log.txt 2>&1 &
```

<!-- <summary>2. Testing </summary> -->
2. Testing
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=9909 --nproc_per_node=4 tools/train.py -c path/to/config -r path/to/checkpoint --test-only
```

<!-- <summary>3. Tuning </summary> -->
3. Tuning
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=9909 --nproc_per_node=4 tools/train.py -c path/to/config -t path/to/checkpoint --use-amp --seed=0 &> log.txt 2>&1 &
```

<!-- <summary>4. Export onnx </summary> -->
4. Export onnx
```shell
python tools/export_onnx.py -c path/to/config -r path/to/checkpoint --check
```

<!-- <summary>5. Inference </summary> -->
5. Inference

Support torch, onnxruntime, tensorrt and openvino, see details in *references/deploy*
```shell
python references/deploy/dfine_onnx.py --onnx-file=model.onnx --im-file=xxxx
python references/deploy/dfine_tensorrt.py --trt-file=model.trt --im-file=xxxx
python references/deploy/dfine_torch.py -c path/to/config -r path/to/checkpoint --im-file=xxx --device=cuda:0
```
</details>



## Citation
If you use `DFINE` or `DFINEv2` in your work, please use the following BibTeX entries:

<details>
<summary> bibtex </summary>

```latex

```
</details>
