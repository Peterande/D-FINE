
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
<summary> details </summary>

<!-- <summary>1. Training </summary> -->
1. Training
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=9909 --nproc_per_node=4 tools/train.py -c path/to/config --use-amp --seed=0
```

<!-- <summary>2. Testing </summary> -->
2. Testing
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=9909 --nproc_per_node=4 tools/train.py -c path/to/config -r path/to/checkpoint --test-only
```

<!-- <summary>3. Tuning </summary> -->
3. Tuning
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=9909 --nproc_per_node=4 tools/train.py -c path/to/config -t path/to/checkpoint --use-amp --seed=0
```

<!-- <summary>4. Export onnx </summary> -->
4. Export onnx and tensorrt
```shell
python tools/export_onnx.py -c path/to/config -r path/to/checkpoint --check
trtexec --onnx=".model.onnx" --saveEngine="./model.engine" --fp16
```

<!-- <summary>5. Inference </summary> -->
5. Inference

Support torch, onnxruntime, tensorrt and openvino, see details in *benchmark/inference*
```shell
python benchmark/inference/onnx_inf.py --onnx-file=model.onnx --im-file=xxxx
python benchmark/inference/trt_inf.py --trt-file=model.trt --im-file=xxxx
python benchmark/inference/torch_inf.py -c path/to/config -r path/to/checkpoint --im-file=xxxx --device=cuda:0
```

<!-- <summary>6. Benchmark </summary> -->
5. Benchmark (Params. / GFLOPs / Latency)
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
