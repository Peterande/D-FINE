#!/usr/bin/env python3
"""Compare inference latency for PyTorch, ONNX Runtime and TensorRT on the same video frames."""

import argparse
import os
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.core import YAMLConfig
from tools.inference.trt_inf import TRTInference
from tools.model_surgery.shared_arch import SegmentationHead, SharedBackboneDualDecoder, load_any_state

try:
    import onnxruntime as ort
except Exception:
    ort = None


@dataclass
class BackendResult:
    name: str
    frames: int
    mean_ms: float
    p50_ms: float
    p95_ms: float


def resize_with_aspect_ratio(image: Image.Image, size: int) -> Tuple[Image.Image, int, int]:
    original_width, original_height = image.size
    ratio = min(size / original_width, size / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    image = image.resize((new_width, new_height), Image.BILINEAR)

    padded = Image.new("RGB", (size, size))
    pad_w = (size - new_width) // 2
    pad_h = (size - new_height) // 2
    padded.paste(image, (pad_w, pad_h))
    return padded, new_width, new_height


def load_video_inputs(video_path: str, input_size: int, max_frames: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    to_tensor = T.ToTensor()
    images_np: List[np.ndarray] = []
    orig_sizes_np: List[np.ndarray] = []

    while True:
        if max_frames > 0 and len(images_np) >= max_frames:
            break

        ok, frame = cap.read()
        if not ok:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        resized, new_w, new_h = resize_with_aspect_ratio(frame_pil, input_size)
        x = to_tensor(resized).unsqueeze(0).numpy().astype(np.float32)
        sz = np.array([[new_h, new_w]], dtype=np.float32)

        images_np.append(x)
        orig_sizes_np.append(sz)

    cap.release()
    if not images_np:
        raise RuntimeError("No frames decoded from video")
    return images_np, orig_sizes_np


def measure_backend(
    name: str,
    run_one: Callable[[np.ndarray, np.ndarray], None],
    images_np: List[np.ndarray],
    orig_sizes_np: List[np.ndarray],
    warmup: int,
) -> BackendResult:
    if warmup > 0:
        for i in range(min(warmup, len(images_np))):
            run_one(images_np[i], orig_sizes_np[i])

    times_ms: List[float] = []
    for x_np, sz_np in zip(images_np, orig_sizes_np):
        t0 = time.perf_counter()
        run_one(x_np, sz_np)
        dt = (time.perf_counter() - t0) * 1000.0
        times_ms.append(dt)

    return BackendResult(
        name=name,
        frames=len(times_ms),
        mean_ms=float(statistics.fmean(times_ms)),
        p50_ms=float(np.percentile(np.array(times_ms), 50)),
        p95_ms=float(np.percentile(np.array(times_ms), 95)),
    )


def build_torch_runner(config: str, resume: str, device: str):
    cfg = YAMLConfig(config, resume=resume)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    checkpoint = torch.load(resume, map_location="cpu")
    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
    else:
        state = checkpoint["model"]

    incompatible = cfg.model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print("[torch] WARNING: non-strict checkpoint load")

    model = cfg.model.deploy().to(device).eval()

    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()

    @torch.inference_mode()
    def run_one(x_np: np.ndarray, sz_np: np.ndarray):
        x = torch.from_numpy(x_np).to(device)
        # torch deploy model ignores orig size; keep signature consistent with others
        if use_cuda:
            torch.cuda.synchronize()
        _ = model(x)
        if use_cuda:
            torch.cuda.synchronize()

    return run_one


def _disable_hgnet_pretrained(cfg: YAMLConfig):
    if hasattr(cfg, "yaml_cfg") and isinstance(cfg.yaml_cfg, dict) and "HGNetv2" in cfg.yaml_cfg:
        node = cfg.yaml_cfg["HGNetv2"]
        if isinstance(node, dict):
            node["pretrained"] = False


def build_torch_multimodel_runner(
    det_config: str,
    pose_config: str,
    merged_ckpt: str,
    device: str,
    input_size: int = 640,
    seg_num_classes: int = 7,
    seg_feature_dim: int = 384,
    seg_dropout: float = 0.1,
):
    det_cfg = YAMLConfig(det_config)
    pose_cfg = YAMLConfig(pose_config)
    _disable_hgnet_pretrained(det_cfg)
    _disable_hgnet_pretrained(pose_cfg)

    det_model = det_cfg.model
    pose_model = pose_cfg.model

    det_model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, 3, int(input_size), int(input_size))
        feats = det_model.backbone(dummy)
    in_channels = [int(f.shape[1]) for f in feats]

    ckpt_state = load_any_state(merged_ckpt)
    fpn_key = "seg_head.fpn.lateral_convs.0.weight"
    feature_dim = int(ckpt_state[fpn_key].shape[0]) if fpn_key in ckpt_state else int(seg_feature_dim)
    seg_head = SegmentationHead(in_channels, int(seg_num_classes), int(feature_dim), float(seg_dropout))

    model = SharedBackboneDualDecoder(
        backbone=det_model.backbone,
        encoder=det_model.encoder,
        det_decoder=det_model.decoder,
        pose_decoder=pose_model.decoder,
        seg_head=seg_head,
    )
    missing, unexpected = model.load_state_dict(ckpt_state, strict=False)
    if missing or unexpected:
        print(
            f"[torch-multimodel] WARNING non-strict load: "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )

    model = model.to(device).eval()
    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()

    @torch.inference_mode()
    def run_one(x_np: np.ndarray, sz_np: np.ndarray):
        x = torch.from_numpy(x_np).to(device)
        if use_cuda:
            torch.cuda.synchronize()
        _ = model(x)
        if use_cuda:
            torch.cuda.synchronize()

    return run_one


def build_onnx_runner(onnx_path: str, use_gpu: bool):
    if ort is None:
        raise RuntimeError("onnxruntime is not installed")

    available = list(ort.get_available_providers())
    providers = ["CPUExecutionProvider"]
    if use_gpu:
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            print(
                "[onnx] WARNING: CUDAExecutionProvider is not available in this environment. "
                f"Falling back to CPU. available={available}"
            )

    providers = [p for p in providers if p in available]
    if not providers:
        raise RuntimeError(f"No usable ONNX Runtime providers found. available={available}")

    try:
        sess = ort.InferenceSession(onnx_path, providers=providers)
    except Exception as e:
        msg = str(e)
        if "GatherElementsAxis1Plugin" in msg:
            raise RuntimeError(
                "ONNX model includes TensorRT-only custom op 'GatherElementsAxis1Plugin'. "
                "Use an ORT-compatible ONNX (before TRT plugin rewrite), e.g. "
                "'outputs/phase2_run/singlepass_best_dynamo.onnx'."
            ) from e
        raise
    print(f"[onnx] providers={sess.get_providers()}")
    input_names = {i.name for i in sess.get_inputs()}

    def run_one(x_np: np.ndarray, sz_np: np.ndarray):
        feed: Dict[str, np.ndarray] = {}
        if "images" in input_names:
            feed["images"] = x_np
        if "orig_target_sizes" in input_names:
            feed["orig_target_sizes"] = sz_np
        if not feed:
            # fallback for unnamed/custom single-input graphs
            first_name = sess.get_inputs()[0].name
            feed[first_name] = x_np
        _ = sess.run(None, feed)

    return run_one


def build_trt_runner(engine_path: str, device: str, plugin_libs: List[str]):
    trt_runner = TRTInference(
        engine_path,
        device=device,
        max_batch_size=1,
        plugin_libs=plugin_libs,
    )

    def run_one(x_np: np.ndarray, sz_np: np.ndarray):
        blob: Dict[str, torch.Tensor] = {}
        x_t = torch.from_numpy(x_np).to(device)
        sz_t = torch.from_numpy(sz_np).to(device)

        if "images" in trt_runner.input_names:
            blob["images"] = x_t
        if "orig_target_sizes" in trt_runner.input_names:
            blob["orig_target_sizes"] = sz_t

        # Fallback if engine uses unnamed/custom input names.
        for n in trt_runner.input_names:
            if n not in blob:
                blob[n] = x_t if len(blob) == 0 else sz_t

        _ = trt_runner(blob)
        trt_runner.synchronize()

    return run_one


def print_table(results: List[BackendResult]):
    print("\nLatency (samme video):")
    print(f"{'Modell':<12} {'Frames':>8} {'Mean (ms)':>12} {'P50 (ms)':>10} {'P95 (ms)':>10}")
    for r in results:
        print(f"{r.name:<12} {r.frames:>8d} {r.mean_ms:>12.2f} {r.p50_ms:>10.2f} {r.p95_ms:>10.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to input video")
    ap.add_argument("--input-size", type=int, default=640)
    ap.add_argument("--max-frames", type=int, default=300, help="0 = all frames")
    ap.add_argument("--warmup", type=int, default=30)

    ap.add_argument("--torch-config", type=str, default=None)
    ap.add_argument("--torch-resume", type=str, default=None)
    ap.add_argument("--torch-device", type=str, default="cuda:0")
    ap.add_argument("--torch-det-config", type=str, default=None, help="Detection config for multimodel torch benchmark")
    ap.add_argument("--torch-pose-config", type=str, default=None, help="Pose config for multimodel torch benchmark")
    ap.add_argument("--torch-merged-ckpt", type=str, default=None, help="Merged model-surgery checkpoint (.pth)")
    ap.add_argument("--torch-seg-num-classes", type=int, default=7)
    ap.add_argument("--torch-seg-feature-dim", type=int, default=384)
    ap.add_argument("--torch-seg-dropout", type=float, default=0.1)

    ap.add_argument("--onnx", type=str, default=None)
    ap.add_argument("--onnx-gpu", action="store_true")

    ap.add_argument("--trt", type=str, default=None)
    ap.add_argument("--trt-device", type=str, default="cuda:0")
    ap.add_argument("--plugin-lib", action="append", default=[])

    args = ap.parse_args()

    max_frames = args.max_frames if args.max_frames > 0 else 10**9
    images_np, orig_sizes_np = load_video_inputs(args.video, args.input_size, max_frames)
    print(f"Loaded {len(images_np)} frames from {args.video}")

    results: List[BackendResult] = []
    failures: List[str] = []

    if args.torch_det_config and args.torch_pose_config and args.torch_merged_ckpt:
        print("Running PyTorch benchmark (multimodel)...")
        try:
            torch_run = build_torch_multimodel_runner(
                det_config=args.torch_det_config,
                pose_config=args.torch_pose_config,
                merged_ckpt=args.torch_merged_ckpt,
                device=args.torch_device,
                input_size=args.input_size,
                seg_num_classes=args.torch_seg_num_classes,
                seg_feature_dim=args.torch_seg_feature_dim,
                seg_dropout=args.torch_seg_dropout,
            )
            results.append(measure_backend("PyTorch", torch_run, images_np, orig_sizes_np, args.warmup))
        except Exception as e:
            failures.append(f"PyTorch failed: {e}")
    elif args.torch_config and args.torch_resume:
        print("Running PyTorch benchmark...")
        try:
            torch_run = build_torch_runner(args.torch_config, args.torch_resume, args.torch_device)
            results.append(measure_backend("PyTorch", torch_run, images_np, orig_sizes_np, args.warmup))
        except Exception as e:
            failures.append(f"PyTorch failed: {e}")

    if args.onnx:
        print("Running ONNX Runtime benchmark...")
        try:
            onnx_run = build_onnx_runner(args.onnx, args.onnx_gpu)
            results.append(measure_backend("ONNX", onnx_run, images_np, orig_sizes_np, args.warmup))
        except Exception as e:
            failures.append(f"ONNX failed: {e}")

    if args.trt:
        print("Running TensorRT benchmark...")
        try:
            trt_run = build_trt_runner(args.trt, args.trt_device, args.plugin_lib)
            results.append(measure_backend("TensorRT", trt_run, images_np, orig_sizes_np, args.warmup))
        except Exception as e:
            failures.append(f"TensorRT failed: {e}")

    if not results:
        raise RuntimeError(
            "No successful benchmark runs. "
            + (" ".join(failures) if failures else "Provide at least one of --torch-*, --onnx, --trt")
        )

    print_table(results)
    if failures:
        print("\nBackends that failed:")
        for m in failures:
            print(f"- {m}")


if __name__ == "__main__":
    main()
