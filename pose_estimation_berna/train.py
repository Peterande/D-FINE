#!/usr/bin/env python3
"""
Unified D-FINE Pose Estimation Training Script (COCO-17).

Implements the recommended approach:
- Use D-FINE's DETR-style decoder queries (per-detection)
- Predict keypoints per query in a parallel branch (already integrated into DFINETransformer)
- Output: pred_keypoints [B, num_queries, 17, 3] where 3=(x_rel, y_rel, visibility_logit)
  and (x_rel, y_rel) are bbox-relative (0..1 within the predicted bbox).

No *extra* matching is introduced: the keypoint loss uses the same Hungarian indices as boxes.
"""

import argparse
import os
import sys
from pathlib import Path
import signal

import faulthandler
from contextlib import nullcontext
import inspect
import time
from datetime import datetime, timedelta
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

# Ensure local imports work (also under multiprocessing 'spawn')
current_dir = os.path.dirname(os.path.abspath(__file__))          # .../pose_estimation_berna
repo_root_dir = os.path.dirname(current_dir)                      # .../D-FINE-NEWBRINGER
src_dir = os.path.join(repo_root_dir, "src")                      # .../D-FINE-NEWBRINGER/src
for p in [repo_root_dir, current_dir, src_dir]:
    if p and p not in sys.path:
        sys.path.insert(0, p)

from pose_estimation_berna.core.datasets import create_coco_pose_dataset
from pose_estimation_berna.core.losses import create_pose_criterion
from pose_estimation_berna.core.metrics import PoseMetricsTracker
from pose_estimation_berna.core.models import (
    build_param_groups,
    freeze_except_keypoints,
    get_device,
    unfreeze_all,
    unfreeze_backbone_stages,
)

def pose_collate_fn(batch):
    """Pickle-safe collate_fn for (image_tensor, target_dict) samples."""
    images = torch.stack([b[0] for b in batch], dim=0)
    targets = [b[1] for b in batch]
    return images, targets


def dataloader_worker_init_fn(_worker_id: int):
    """Pickle-safe worker init: avoid OpenCV oversubscribing threads if cv2 is installed."""
    try:
        import cv2  # optional dependency in some envs

        cv2.setNumThreads(0)
    except Exception:
        pass


def deep_merge_dict(base_dict: dict, override_dict: dict) -> dict:
    result = base_dict.copy()
    for key, value in override_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def load_config(tier: str, config_dir: str) -> dict:
    cfg_path = os.path.join(config_dir, f"{tier}.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    if "base" in cfg and cfg["base"]:
        base_path = os.path.join(config_dir, cfg["base"])
        with open(base_path, "r") as f:
            base_cfg = yaml.safe_load(f)
        cfg = deep_merge_dict(base_cfg, cfg)
    return cfg


def parse_args():
    p = argparse.ArgumentParser(description="Unified D-FINE Pose Training")
    p.add_argument("--tier", required=True, choices=["lightweight", "standard", "advanced"])
    p.add_argument(
        "--run-name",
        default=None,
        help="Optional output subfolder name (defaults to --tier). Example: --tier standard --run-name standard_lqe_obj365",
    )
    p.add_argument("--config-dir", default="configs")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", type=str, default=None, help="Resume from a pose checkpoint (*.pth)")
    p.add_argument(
        "--freeze-detection",
        action="store_true",
        help="Freeze everything except keypoint head (overrides config training.freeze_detection)",
    )
    p.add_argument(
        "--no-freeze-detection",
        action="store_true",
        help="Do NOT freeze detection/box/class heads (overrides config training.freeze_detection)",
    )
    p.add_argument("--num-workers", type=int, default=None, help="Override DataLoader num_workers from config")
    p.add_argument("--pin-memory", action="store_true", help="Enable DataLoader pin_memory (overrides config)")
    p.add_argument("--no-pin-memory", action="store_true", help="Disable DataLoader pin_memory (overrides config)")
    # Debug / smoke-test knobs
    p.add_argument("--epochs", type=int, default=None, help="Override epochs from config")
    p.add_argument("--batch-size", type=int, default=None, help="Override batch_size from config")
    p.add_argument(
        "--grad-accum-steps",
        type=int,
        default=None,
        help="Override gradient accumulation steps from config (effective batch = batch_size * grad_accum_steps)",
    )
    p.add_argument(
        "--amp",
        action="store_true",
        help="Enable mixed precision (overrides config; recommended on CUDA to reduce memory)",
    )
    p.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable mixed precision (overrides config)",
    )
    p.add_argument("--compile", action="store_true", help="Enable torch.compile() (overrides config)")
    p.add_argument("--no-compile", action="store_true", help="Disable torch.compile() (overrides config)")
    p.add_argument(
        "--channels-last",
        action="store_true",
        help="Use channels_last memory format (overrides config; often faster on CUDA conv backbones)",
    )
    p.add_argument("--no-channels-last", action="store_true", help="Disable channels_last (overrides config)")
    p.add_argument(
        "--val-every",
        type=int,
        default=None,
        help="Run validation every N epochs (default from config, else 1). Speeds up training if >1.",
    )
    p.add_argument(
        "--coco-eval-every",
        type=int,
        default=None,
        help="Run COCO keypoint eval every N epochs (default from config, else 1). Speeds up training if >1.",
    )
    p.add_argument("--max-train-samples", type=int, default=None, help="Limit training dataset size")
    p.add_argument("--max-val-samples", type=int, default=None, help="Limit validation dataset size")
    p.add_argument("--max-train-steps", type=int, default=None, help="Limit training steps per epoch")
    p.add_argument("--max-val-steps", type=int, default=None, help="Limit validation steps")
    p.add_argument("--train-split", default="train", choices=["train", "val"], help="COCO split for training")
    p.add_argument("--val-split", default="val", choices=["train", "val"], help="COCO split for validation")
    p.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Override config dataset.image_size (square resize). Example: --image-size 896",
    )
    p.add_argument(
        "--best-metric",
        default="auto",
        choices=["auto", "coco_ap_keypoints", "oks"],
        help="Which metric controls best.pth. auto = coco_ap_keypoints if COCOeval is available, else oks.",
    )
    return p.parse_args()


def main():
    # Print Python tracebacks on native crashes (segfaults) to help debugging long runs.
    # NOTE: this won't prevent the segfault, but it can reveal *where* it happened.
    try:
        faulthandler.enable(all_threads=True)
    except Exception:
        pass

    args = parse_args()
    cfg = load_config(args.tier, args.config_dir)
    # Optional overrides (keep config files clean)
    if args.image_size is not None:
        cfg.setdefault("dataset", {})
        cfg["dataset"]["image_size"] = int(args.image_size)
        print(f"🖼️ Overriding dataset.image_size -> {cfg['dataset']['image_size']}")

    device = get_device(args.device)
    print(f"🚀 Using device: {device}")
    if device.type == "cuda":
        # safe defaults for speed; no accuracy loss for fp32-accumulated ops
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # fixed-size inputs (e.g., 640x640) benefit from cuDNN autotuning
        cudnn_benchmark = bool(cfg.get("system", {}).get("cudnn_benchmark", True))
        torch.backends.cudnn.benchmark = cudnn_benchmark
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    # AMP dtype (fp16 default; bf16 is often more stable if supported)
    amp_dtype_cfg = str(cfg.get("training", {}).get("amp_dtype", "fp16")).lower()
    if amp_dtype_cfg in ("bf16", "bfloat16"):
        if device.type == "cuda" and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
        else:
            print("⚠️ amp_dtype=bf16 requested but bf16 is not supported on this GPU; falling back to fp16.")
            amp_dtype = torch.float16
    else:
        amp_dtype = torch.float16

    def _to_fp32(x):
        """Recursively cast tensors to fp32 (keeps dict/list/tuple structure)."""
        if isinstance(x, torch.Tensor):
            # Only cast floating tensors; keep int/bool tensors as-is (e.g. dn_meta indices).
            return x.float() if x.is_floating_point() else x
        if isinstance(x, dict):
            return {k: _to_fp32(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_to_fp32(v) for v in x]
        if isinstance(x, tuple):
            return tuple(_to_fp32(v) for v in x)
        return x

    def _check_finite(outputs: dict, targets: list, phase: str):
        """Fail fast with context if model outputs are non-finite."""
        pb = outputs.get("pred_boxes", None)
        if pb is None:
            return
        if not torch.isfinite(pb).all():
            bad = int((~torch.isfinite(pb)).sum().detach().cpu().item())
            img_ids = []
            try:
                img_ids = [int(t["image_id"].reshape(-1)[0].detach().cpu().item()) for t in targets]
            except Exception:
                img_ids = []
            raise RuntimeError(
                f"[{phase}] Non-finite pred_boxes detected (bad_elems={bad}). image_ids={img_ids}. "
                f"Suggested fix: set training.amp_dtype=bf16 (if supported) or disable AMP."
            )

    base_dir = Path(__file__).resolve().parent  # pose_estimation_berna/
    repo_root = base_dir.parent

    # Create DFINE model (must be a config that enables num_keypoints in DFINETransformer)
    dfine_cfg = Path(cfg["dfine"]["config_path"])
    ckpt = Path(cfg["dfine"]["checkpoint_path"])
    if not dfine_cfg.is_absolute():
        dfine_cfg = base_dir / dfine_cfg
    if not ckpt.is_absolute():
        ckpt = base_dir / ckpt
    print(f"🏗️ Loading DFINE: {dfine_cfg}")
    from src.core import YAMLConfig

    df_cfg = YAMLConfig(str(dfine_cfg))
    # avoid re-downloading backbone weights during training script
    if "HGNetv2" in df_cfg.yaml_cfg:
        df_cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    # If we override image_size (e.g. 896), the default DFINE configs still set
    # eval_spatial_size=[640,640]. That caches positional embeddings and decoder
    # anchor masks for 640 and will crash at validation with shape mismatches.
    # Safer: disable the fixed eval_spatial_size cache and always derive shapes
    # from the current feature maps.
    if int(cfg.get("dataset", {}).get("image_size", 640)) != 640 and "eval_spatial_size" in df_cfg.yaml_cfg:
        df_cfg.yaml_cfg["eval_spatial_size"] = None

    model = df_cfg.model.to(device)

    def _load_state(path: str):
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict):
            if "ema" in state and isinstance(state["ema"], dict) and "module" in state["ema"]:
                return state["ema"]["module"]
            if "model" in state and isinstance(state["model"], dict):
                return state["model"]
        return state

    def _load_state_forgiving(model: torch.nn.Module, path: str):
        """
        Load a checkpoint but drop keys with mismatched shapes (e.g. Obj365 365-class heads into COCO 80-class model).
        This keeps backbone/encoder/decoder weights while avoiding load_state_dict size-mismatch errors.
        """
        sd = _load_state(path)
        if not isinstance(sd, dict):
            return sd
        msd = model.state_dict()
        kept = {}
        dropped = 0
        for k, v in sd.items():
            if k in msd and hasattr(v, "shape") and hasattr(msd[k], "shape") and v.shape == msd[k].shape:
                kept[k] = v
            else:
                dropped += 1
        print(f"🧩 Forgiving load: keeping {len(kept)}/{len(sd)} keys (dropped {dropped} mismatched)")
        return kept

    # Load initial weights: resume (pose checkpoint) > base detector checkpoint
    start_epoch = 0
    best_metric = None
    best_metric_name = None
    # Keep both around so we never accidentally compare OKS vs COCO-AP across resumes.
    best_oks = None
    best_coco_ap = None
    resume_obj = None
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = repo_root / resume_path
        resume_obj = torch.load(str(resume_path), map_location="cpu")
        if isinstance(resume_obj, dict) and "model" in resume_obj:
            model.load_state_dict(resume_obj["model"], strict=False)
            start_epoch = int(resume_obj.get("epoch", 0)) + 1
            best_metric = resume_obj.get("best_metric", None)
            best_metric_name = resume_obj.get("best_metric_name", None)
            best_oks = resume_obj.get("best_oks", None)
            best_coco_ap = resume_obj.get("best_coco_ap", None)
            # Back-compat: infer missing explicit fields from legacy best_metric.
            if best_oks is None and best_metric_name == "oks":
                best_oks = best_metric
            if best_coco_ap is None and best_metric_name == "coco_ap_keypoints":
                best_coco_ap = best_metric
        else:
            model.load_state_dict(resume_obj, strict=False)
        print(f"🔁 Resumed from: {resume_path} (start_epoch={start_epoch})")
    else:
        # Base init: allow loading Obj365 checkpoints even if class heads don't match COCO.
        model.load_state_dict(_load_state_forgiving(model, str(ckpt)), strict=False)
    
    # Optional: freeze everything except keypoint head
    if args.freeze_detection and args.no_freeze_detection:
        raise ValueError("Choose only one of --freeze-detection or --no-freeze-detection")
    if args.freeze_detection:
        freeze_detection = True
    elif args.no_freeze_detection:
        freeze_detection = False
    else:
        freeze_detection = bool(cfg.get("training", {}).get("freeze_detection", False))
    if freeze_detection:
        freeze_except_keypoints(model)
        print("🔒 Frozen all params except decoder keypoint head")

    # Freeze→unfreeze schedule (pose warmup): train pose heads first, then unfreeze everything.
    # This is safer than training all heads from step 1 when adapting a detection-pretrained model to pose.
    pose_only_epochs = int(cfg.get("training", {}).get("pose_only_epochs", 0) or 0)
    pose_unfreeze_epoch = None
    # Important: pose warmup should only run on a fresh start; on resume it would re-freeze a trained model.
    if pose_only_epochs > 0 and start_epoch == 0:
        pose_unfreeze_epoch = start_epoch + pose_only_epochs
        # Even if freeze_detection is false, a pose-warmup is allowed to override it.
        if not freeze_detection:
            freeze_except_keypoints(model)
            freeze_detection = True
            print(f"🔒 Pose warmup enabled: training pose-only for {pose_only_epochs} epochs (until epoch {pose_unfreeze_epoch})")
    elif pose_only_epochs > 0 and start_epoch > 0:
        print(f"ℹ️ Skipping pose_only_epochs warmup on resume (start_epoch={start_epoch}).")

    # Optional: channels_last for better throughput on conv-heavy backbones
    cfg_channels_last = bool(cfg.get("system", {}).get("channels_last", False))
    if args.channels_last:
        use_channels_last = True
    elif args.no_channels_last:
        use_channels_last = False
    else:
        use_channels_last = cfg_channels_last and device.type == "cuda"
    if use_channels_last:
        try:
            model = model.to(memory_format=torch.channels_last)
            print("📦 Using channels_last memory format")
        except Exception as e:
            print(f"⚠️ channels_last requested but could not be applied: {e}")
            use_channels_last = False

    # Optional: torch.compile
    cfg_compile = bool(cfg.get("training", {}).get("compile", False))
    if args.compile:
        use_compile = True
    elif args.no_compile:
        use_compile = False
    else:
        use_compile = cfg_compile
    if use_compile:
        if hasattr(torch, "compile"):
            try:
                # Make compile safer: if Dynamo hits unsupported graph/symbolic-shape paths,
                # fall back to eager rather than crashing (may reduce speedup but avoids "tulling").
                try:
                    import torch._dynamo as dynamo  # type: ignore

                    dynamo.config.suppress_errors = True
                except Exception:
                    pass

                # Use conservative defaults unless user overrides via config.
                compile_cfg = cfg.get("training", {}) or {}
                compile_backend = str(compile_cfg.get("compile_backend", "inductor"))
                compile_mode = str(compile_cfg.get("compile_mode", "reduce-overhead"))
                compile_dynamic = bool(compile_cfg.get("compile_dynamic", False))
                compile_fullgraph = bool(compile_cfg.get("compile_fullgraph", False))

                compile_kwargs = {}
                sig = inspect.signature(torch.compile)
                if "backend" in sig.parameters:
                    compile_kwargs["backend"] = compile_backend
                if "mode" in sig.parameters:
                    compile_kwargs["mode"] = compile_mode
                if "dynamic" in sig.parameters:
                    compile_kwargs["dynamic"] = compile_dynamic
                if "fullgraph" in sig.parameters:
                    compile_kwargs["fullgraph"] = compile_fullgraph

                model = torch.compile(model, **compile_kwargs)
                print(
                    f"⚡ torch.compile enabled (backend={compile_backend}, mode={compile_mode}, dynamic={compile_dynamic})"
                )
                print("🛠️ Note: first training step can take a few minutes while torch.compile/inductor builds kernels.")
            except Exception as e:
                print(f"⚠️ torch.compile failed; continuing without compile: {e}")
        else:
            print("⚠️ torch.compile not available in this torch version")

    # Criterion (DFINECriterion + keypoints)
    criterion = create_pose_criterion(num_classes=cfg.get("dfine", {}).get("num_classes", 80)).to(device)
    metrics = PoseMetricsTracker(num_keypoints=cfg["dataset"]["num_keypoints"])

    # Postprocessor + official COCO keypoint evaluator (optional but recommended for "best model")
    postprocessor = df_cfg.postprocessor.to(device).eval()
    postprocessor.remap_mscoco_category = True  # label(0) -> category_id(1)=person
    use_coco_eval = bool(cfg.get("evaluation", {}).get("use_coco_eval", True))
    coco_evaluator = None
    if use_coco_eval:
        try:
            from faster_coco_eval import COCO
            from src.data.dataset.coco_eval import CocoEvaluator

            coco_val_ann = Path(cfg["dataset"]["root_dir"]) / "annotations" / "person_keypoints_val2017.json"
            coco_val_ann = repo_root / coco_val_ann if not str(coco_val_ann).startswith("/") else coco_val_ann
            coco_gt = COCO(str(coco_val_ann))
            coco_evaluator = CocoEvaluator(coco_gt, iou_types=["keypoints"])
        except Exception as e:
            print(f"⚠️ COCOeval disabled (could not initialize): {e}")
            coco_evaluator = None

    # Decide which metric controls best.pth in THIS run.
    # Default is "auto": COCO-AP(kpt) if COCOeval exists, else OKS.
    if args.best_metric == "auto":
        chosen_best_metric = "coco_ap_keypoints" if coco_evaluator is not None else "oks"
    else:
        chosen_best_metric = args.best_metric
        if chosen_best_metric == "coco_ap_keypoints" and coco_evaluator is None:
            print("⚠️ --best-metric=coco_ap_keypoints requested but COCOeval is not available; falling back to OKS.")
            chosen_best_metric = "oks"

    if chosen_best_metric == "coco_ap_keypoints":
        if best_coco_ap is None:
            print("📌 Best checkpoint will be selected by COCO-AP(kpt) in this run.")
        best_metric = best_coco_ap
        best_metric_name = "coco_ap_keypoints"
    else:
        if best_oks is None:
            print("📌 Best checkpoint will be selected by Val OKS in this run.")
        best_metric = best_oks
        best_metric_name = "oks"

    # Data
    train_ds = create_coco_pose_dataset(
        root_dir=str(
            (repo_root / cfg["dataset"]["root_dir"])
            if not str(cfg["dataset"]["root_dir"]).startswith("/")
            else cfg["dataset"]["root_dir"]
        ),
        split=args.train_split,
        image_size=cfg["dataset"]["image_size"],
        num_keypoints=cfg["dataset"]["num_keypoints"],
        tier=args.tier,
    )
    val_ds = create_coco_pose_dataset(
        root_dir=str(
            (repo_root / cfg["dataset"]["root_dir"])
            if not str(cfg["dataset"]["root_dir"]).startswith("/")
            else cfg["dataset"]["root_dir"]
        ),
        split=args.val_split,
        image_size=cfg["dataset"]["image_size"],
        num_keypoints=cfg["dataset"]["num_keypoints"],
        tier=args.tier,
    )
    
    # Optional dataset size limits for quick pipeline tests
    if args.max_train_samples is not None:
        n = max(0, int(args.max_train_samples))
        train_ds = torch.utils.data.Subset(train_ds, list(range(min(n, len(train_ds)))))
    if args.max_val_samples is not None:
        n = max(0, int(args.max_val_samples))
        val_ds = torch.utils.data.Subset(val_ds, list(range(min(n, len(val_ds)))))

    num_workers = int(args.num_workers) if args.num_workers is not None else int(cfg.get("system", {}).get("num_workers", 4))
    persistent_workers = bool(cfg.get("system", {}).get("persistent_workers", True))
    prefetch_factor = int(cfg.get("system", {}).get("prefetch_factor", 2))
    start_method = cfg.get("system", {}).get("start_method", None) or cfg.get("system", {}).get(
        "multiprocessing_context", None
    )
    train_bs = int(args.batch_size) if args.batch_size is not None else int(cfg["training"]["batch_size"])
    val_bs = int(cfg.get("training", {}).get("val_batch_size", train_bs))

    # pin_memory override (sometimes improves stability to disable)
    if args.pin_memory and args.no_pin_memory:
        raise ValueError("Choose only one of --pin-memory or --no-pin-memory")
    if args.pin_memory:
        use_pin_memory = True
    elif args.no_pin_memory:
        use_pin_memory = False
    else:
        use_pin_memory = bool(cfg.get("system", {}).get("pin_memory", True))

    # Build DataLoader kwargs safely across torch versions
    common_loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        worker_init_fn=dataloader_worker_init_fn,
    )
    if num_workers > 0:
        common_loader_kwargs["persistent_workers"] = persistent_workers
        common_loader_kwargs["prefetch_factor"] = max(1, prefetch_factor)
        if start_method:
            # Using "spawn" is often more stable for long runs when native libs are involved.
            common_loader_kwargs["multiprocessing_context"] = str(start_method)
    # NOTE: pin_memory_device is deprecated in recent torch and can spam warnings.
    # We rely on plain pin_memory + non_blocking transfers instead.
    
    train_loader = DataLoader(
        train_ds,
        batch_size=train_bs,
        shuffle=True,
        **common_loader_kwargs,
        collate_fn=pose_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=val_bs,
        shuffle=False,
        **common_loader_kwargs,
        collate_fn=pose_collate_fn,
    )
    
    if args.dry_run:
        print("✅ dry-run OK (model/dataset/criterion constructed)")
        return
    
    # Optimizer (param groups with LR multipliers)
    lr = float(cfg["training"]["learning_rate"])
    wd = float(cfg["training"].get("weight_decay", 1e-4))
    backbone_lr_factor = float(cfg["training"].get("backbone_lr_factor", 0.1))
    keypoint_lr_factor = float(cfg["training"].get("keypoint_lr_factor", 1.0))
    optimizer = optim.AdamW(
        build_param_groups(
            model,
            base_lr=lr,
            weight_decay=wd,
            backbone_lr_factor=backbone_lr_factor,
            keypoint_lr_factor=keypoint_lr_factor,
        )
    )

    epochs = int(args.epochs) if args.epochs is not None else int(cfg["training"]["epochs"])
    if epochs <= int(start_epoch):
        extra = int(start_epoch) + 1 - int(epochs)
        raise ValueError(
            f"--epochs={epochs} is <= start_epoch={start_epoch}, so there are 0 epochs to run. "
            f"Either increase --epochs (e.g. --epochs {start_epoch + 10} to run ~10 more epochs), "
            f"or omit --epochs to use the config value. "
            f"(You are short by at least {extra} epoch(s) just to run one epoch.)"
        )
    grad_accum_steps = (
        int(args.grad_accum_steps)
        if args.grad_accum_steps is not None
        else int(cfg.get("training", {}).get("grad_accum_steps", 1))
    )
    grad_accum_steps = max(1, grad_accum_steps)
    # AMP defaults to config, but only makes sense on CUDA
    cfg_amp = bool(cfg.get("training", {}).get("amp", True))
    if args.amp:
        use_amp = device.type == "cuda"
    elif args.no_amp:
        use_amp = False
    else:
        use_amp = cfg_amp and device.type == "cuda"
    # Use torch.amp if available (torch.cuda.amp is deprecated in newer torch)
    if hasattr(torch, "amp"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        print(
            f"🧠 AMP enabled | batch_size={train_bs} | grad_accum_steps={grad_accum_steps} | effective_batch={train_bs*grad_accum_steps}"
        )
    else:
        print(
            f"🧠 AMP disabled | batch_size={train_bs} | grad_accum_steps={grad_accum_steps} | effective_batch={train_bs*grad_accum_steps}"
        )

    # Resume optimizer / scaler state if available (true continuation of LR/momentum).
    if isinstance(resume_obj, dict):
        try:
            if "optimizer" in resume_obj and isinstance(resume_obj["optimizer"], dict):
                optimizer.load_state_dict(resume_obj["optimizer"])
                print("🔁 Restored optimizer state from checkpoint.")
        except Exception as e:
            print(f"⚠️ Could not restore optimizer state (continuing with fresh optimizer): {e}")
        try:
            if "scaler" in resume_obj and hasattr(scaler, "load_state_dict") and isinstance(resume_obj["scaler"], dict):
                scaler.load_state_dict(resume_obj["scaler"])
                print("🔁 Restored AMP scaler state from checkpoint.")
        except Exception as e:
            print(f"⚠️ Could not restore AMP scaler state: {e}")
    run_name = str(args.run_name) if args.run_name else str(args.tier)
    # IMPORTANT: make output dir stable regardless of current working directory (avoid nested outputs/.../outputs/...).
    out_base = Path(cfg.get("output", {}).get("base_dir", "outputs/pose"))
    if not out_base.is_absolute():
        out_base = (repo_root / out_base).resolve()
    out_dir = out_base / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_every = int(cfg.get("output", {}).get("save_every", 1))
    save_best = bool(cfg.get("output", {}).get("save_best", True))
    save_last_every_steps = int(cfg.get("output", {}).get("save_last_every_steps", 0) or 0)

    # Graceful Ctrl+C: set a flag, then checkpoint at a safe point.
    stop_requested = {"flag": False}

    def _handle_sigint(_signum, _frame):
        stop_requested["flag"] = True
        print("\n🛑 Ctrl+C received — will stop after current step/epoch and save checkpoint.")

    try:
        signal.signal(signal.SIGINT, _handle_sigint)
    except Exception:
        pass

    # Progressive unfreezing knobs
    unfreeze_epoch = cfg.get("training", {}).get("unfreeze_epoch", None)
    unfreeze_epoch1 = cfg.get("training", {}).get("unfreeze_epoch1", None)
    unfreeze_epoch2 = cfg.get("training", {}).get("unfreeze_epoch2", None)

    eval_cfg = cfg.get("evaluation", {}) or {}
    val_every = int(args.val_every) if args.val_every is not None else int(eval_cfg.get("val_every", 1))
    coco_eval_every = (
        int(args.coco_eval_every) if args.coco_eval_every is not None else int(eval_cfg.get("coco_eval_every", 1))
    )
    val_every = max(1, val_every)
    # Allow disabling COCOeval inside training (helps avoid rare segfaults in evaluator/native deps).
    # Use: --coco-eval-every 0
    coco_eval_every = int(coco_eval_every)
    if coco_eval_every < 0:
        coco_eval_every = 0
    if coco_eval_every == 0:
        if coco_evaluator is not None:
            print("⚠️ Disabling COCOeval inside training (coco_eval_every=0). Use eval_coco_kpt.py for AP(kpt).")
        coco_evaluator = None
        if chosen_best_metric == "coco_ap_keypoints":
            print("⚠️ Best-metric was COCO-AP(kpt) but COCOeval is disabled; switching best-metric to OKS.")
            chosen_best_metric = "oks"
            best_metric_name = "oks"
            best_metric = best_oks
    else:
        coco_eval_every = max(1, coco_eval_every)

    # Simple wall-clock tracking for global ETA
    run_start_time = time.time()
    completed_epoch_times_sec = []  # only epochs we actually ran in this process

    for epoch in range(start_epoch, epochs):
        epoch_wall_start = time.time()

        # Global ETA (based on mean epoch time so far, else unknown for first epoch)
        epochs_done = epoch - start_epoch
        epochs_left = max(0, epochs - epoch)
        if completed_epoch_times_sec:
            mean_epoch = sum(completed_epoch_times_sec) / float(len(completed_epoch_times_sec))
            eta_sec = int(mean_epoch * epochs_left)
            eta_str = str(timedelta(seconds=eta_sec))
            eta_finish = datetime.now() + timedelta(seconds=eta_sec)
            print(
                f"⏱️ Overall ETA: ~{eta_str} remaining | "
                f"{epochs_done}/{epochs} epochs done | "
                f"estimated finish ~ {eta_finish.strftime('%Y-%m-%d %H:%M:%S')}"
            )
        else:
            print(f"⏱️ Overall ETA: collecting first epoch timing... | {epochs_done}/{epochs} epochs done")

        # staged unfreezing
        if pose_unfreeze_epoch is not None and epoch == int(pose_unfreeze_epoch):
            print(f"🔓 Pose warmup done — unfreezing all params at epoch {epoch}")
            unfreeze_all(model)
            optimizer = optim.AdamW(
                build_param_groups(
                    model,
                    base_lr=lr,
                    weight_decay=wd,
                    backbone_lr_factor=backbone_lr_factor,
                    keypoint_lr_factor=keypoint_lr_factor,
                )
            )
        if unfreeze_epoch is not None and epoch == int(unfreeze_epoch):
            print(f"🔓 Unfreezing all params at epoch {epoch} (was freeze_detection)")
            unfreeze_all(model)
            optimizer = optim.AdamW(
                build_param_groups(
                    model,
                    base_lr=lr,
                    weight_decay=wd,
                    backbone_lr_factor=backbone_lr_factor,
                    keypoint_lr_factor=keypoint_lr_factor,
                )
            )
        if unfreeze_epoch1 is not None and epoch == int(unfreeze_epoch1):
            n = unfreeze_backbone_stages(model, [3])
            print(f"🔓 Unfroze HGNetv2 stage4 params: {n}")
            optimizer = optim.AdamW(
                build_param_groups(
                    model,
                    base_lr=lr,
                    weight_decay=wd,
                    backbone_lr_factor=backbone_lr_factor,
                    keypoint_lr_factor=keypoint_lr_factor,
                )
            )
        if unfreeze_epoch2 is not None and epoch == int(unfreeze_epoch2):
            n = unfreeze_backbone_stages(model, [2])
            print(f"🔓 Unfroze HGNetv2 stage3 params: {n}")
            optimizer = optim.AdamW(
                build_param_groups(
                    model,
                    base_lr=lr,
                    weight_decay=wd,
                    backbone_lr_factor=backbone_lr_factor,
                    keypoint_lr_factor=keypoint_lr_factor,
                )
            )

        model.train()
        criterion.train()
        metrics.reset()

        # If max steps are requested, make tqdm reflect the true total (otherwise it shows full loader length).
        train_total = len(train_loader)
        if args.max_train_steps is not None:
            train_total = min(train_total, int(args.max_train_steps))
        pbar = tqdm(train_loader, total=train_total, desc=f"Epoch {epoch+1}/{epochs}")
        optimizer.zero_grad(set_to_none=True)
        step = -1
        did_any_train = False
        for step, (images, targets) in enumerate(pbar):
            did_any_train = True
            if stop_requested["flag"]:
                break
            non_blocking = device.type == "cuda" and use_pin_memory
            images = images.to(device, non_blocking=non_blocking)
            if use_channels_last and images.ndim == 4:
                images = images.contiguous(memory_format=torch.channels_last)
            targets = [
                {
                    k: (v.to(device, non_blocking=non_blocking) if isinstance(v, torch.Tensor) else v)
                    for k, v in t.items()
                }
                for t in targets
            ]

            with (
                torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype)
                if hasattr(torch, "amp") and device.type == "cuda"
                else nullcontext()
            ):
                outputs = model(images, targets=targets)

            _check_finite(outputs, targets, phase="train")

            # Compute criterion/matcher in fp32 for stability (even if model forward used AMP).
            loss_dict = criterion(_to_fp32(outputs), targets)
            loss = sum(loss_dict.values())
            loss_to_backprop = loss / float(grad_accum_steps)

            scaler.scale(loss_to_backprop).backward()

            should_step = ((step + 1) % grad_accum_steps) == 0
            if should_step:
                # unscale before clipping
                if float(cfg["training"].get("gradient_clipping", 1.0)) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        float(cfg["training"].get("gradient_clipping", 1.0)),
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # quick OKS on matched pairs (no extra matching)
            indices = criterion.matcher({k: v for k, v in _to_fp32(outputs).items() if "aux" not in k}, targets)[
                "indices"
            ]
            metrics.update(outputs, targets, indices)
            oks = metrics.compute()["OKS"]

            pbar.set_postfix(loss=float(loss.detach().cpu()), OKS=f"{oks:.3f}")
            if args.max_train_steps is not None and (step + 1) >= int(args.max_train_steps):
                break

            # Periodic "last" save to reduce work lost on native crashes.
            # This does not affect training (only writes a checkpoint).
            if save_last_every_steps > 0 and ((step + 1) % save_last_every_steps == 0):
                try:
                    last_path = out_dir / "last.pth"
                    torch.save(
                        {
                            "epoch": epoch,
                            "step": step,
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "best_metric": best_metric,
                            "best_metric_name": best_metric_name,
                        },
                        last_path,
                    )
                except Exception:
                    pass
        # flush remainder if we broke early or epoch length not divisible by grad_accum_steps
        # (only if we have pending grads)
        if did_any_train and ((step + 1) % grad_accum_steps) != 0:
            if float(cfg["training"].get("gradient_clipping", 1.0)) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(cfg["training"].get("gradient_clipping", 1.0)),
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # Always checkpoint "last" right after finishing the TRAIN phase of the epoch.
        # This protects you from losing progress if validation crashes or the job gets preempted.
        last_path = out_dir / "last.pth"
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_metric": best_metric,
                "best_metric_name": best_metric_name,
                "best_oks": best_oks,
                "best_coco_ap": best_coco_ap,
            },
            last_path,
        )

        if stop_requested["flag"]:
            print("🛑 Stop requested — saving interrupt checkpoint and exiting cleanly...")
            interrupt_path = out_dir / "interrupt_last.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_metric": best_metric,
                    "best_metric_name": best_metric_name,
                },
                interrupt_path,
            )
            print(f"💾 Saved {interrupt_path}")
            # Skip validation and exit the epoch loop cleanly.
            epoch_wall_sec = time.time() - epoch_wall_start
            completed_epoch_times_sec.append(epoch_wall_sec)
            print(f"⏲️ Epoch time (interrupted): {timedelta(seconds=int(epoch_wall_sec))}")
            break

        # validation loop (optionally less frequent for speed)
        did_validate = ((epoch - start_epoch) % val_every) == 0 or (epoch + 1) == epochs
        val_ap = None
        val_oks = None
        val_loss = None
        is_best = False
        metric_name = "oks"
        metric_val = None

        if did_validate:
            do_coco = coco_evaluator is not None and (
                ((epoch - start_epoch) % coco_eval_every) == 0 or (epoch + 1) == epochs
            )

            model.eval()
            criterion.eval()
            metrics.reset()
            val_loss_acc = 0.0

            if do_coco:
                coco_evaluator.cleanup()
                coco_predictions = {}  # image_id -> prediction dict

            # Validation in fp32 to avoid rare fp16 overflow/NaN in matcher/box ops.
            with torch.no_grad():
                val_total = len(val_loader)
                if args.max_val_steps is not None:
                    val_total = min(val_total, int(args.max_val_steps))
                for step, (images, targets) in enumerate(tqdm(val_loader, total=val_total, desc="Val")):
                    non_blocking = device.type == "cuda" and use_pin_memory
                    images = images.to(device, non_blocking=non_blocking)
                    if use_channels_last and images.ndim == 4:
                        images = images.contiguous(memory_format=torch.channels_last)
                    targets = [
                        {
                            k: (
                                v.to(device, non_blocking=non_blocking)
                                if isinstance(v, torch.Tensor)
                                else v
                            )
                            for k, v in t.items()
                        }
                        for t in targets
                    ]
                    outputs = model(images, targets=targets)
                    _check_finite(outputs, targets, phase="val")
                    loss_dict = criterion(_to_fp32(outputs), targets)
                    val_loss_acc += float(sum(loss_dict.values()).detach().cpu())
                    indices = criterion.matcher({k: v for k, v in _to_fp32(outputs).items() if "aux" not in k}, targets)[
                        "indices"
                    ]
                    metrics.update(outputs, targets, indices)

                    if do_coco:
                        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device)
                        preds = postprocessor(outputs, orig_target_sizes)  # list[dict]
                        for t, p in zip(targets, preds):
                            img_id = int(t["image_id"].reshape(-1)[0].detach().cpu().item())
                            # COCO keypoint eval is defined for the "person" category_id=1.
                            # Filter predictions to person-only and ensure tensors exist even if empty.
                            boxes = p["boxes"].detach().cpu()
                            scores = p["scores"].detach().cpu()
                            labels = p["labels"].detach().cpu()
                            kpts = p.get("keypoints", None)
                            if kpts is None:
                                kpts = torch.empty((0, cfg["dataset"]["num_keypoints"], 3), dtype=torch.float32)
                            else:
                                kpts = kpts.detach().cpu()
                            # COCOeval expects keypoints as [x, y, v] (v is a visibility flag).
                            # Our postprocessor uses the 3rd channel as a per-keypoint confidence score
                            # (useful for visualization), which can hurt evaluation if interpreted as v.
                            # For evaluation, force v=1 for all predicted keypoints.
                            if kpts.numel() > 0 and kpts.shape[-1] == 3:
                                kpts = kpts.clone()
                                kpts[..., 2] = 1.0
                            person_mask = labels == 1
                            coco_predictions[img_id] = {
                                "boxes": boxes[person_mask],
                                "scores": scores[person_mask],
                                "labels": labels[person_mask],
                                "keypoints": kpts[person_mask],
                            }

                    if args.max_val_steps is not None and (step + 1) >= int(args.max_val_steps):
                        break

            val_loss = val_loss_acc / max(1, len(val_loader))
            val_oks = float(metrics.compute()["OKS"])

            if do_coco and coco_predictions:
                coco_evaluator.update(coco_predictions)
                coco_evaluator.synchronize_between_processes()
                coco_evaluator.accumulate()
                try:
                    coco_evaluator.summarize()
                except Exception:
                    pass
                stats = getattr(coco_evaluator.coco_eval["keypoints"], "stats", None)
                if stats is not None and len(stats) > 0:
                    val_ap = float(stats[0])

            # Best checkpoint selection:
            # - If COCOeval is available, we ONLY select best by COCO-AP(keypoints) (to avoid mixing metrics).
            # - Otherwise, fall back to OKS.
            if coco_evaluator is not None:
                if val_ap is not None:
                    print(f"📊 Val loss: {val_loss:.4f} | Val OKS: {val_oks:.3f} | COCO-AP(kpt): {val_ap:.3f}")
                    metric_name = "coco_ap_keypoints"
                    metric_val = float(val_ap)
                    is_best = best_metric is None or float(metric_val) > float(best_metric)
                    if is_best:
                        best_metric = float(metric_val)
                        best_metric_name = metric_name
                        best_coco_ap = float(metric_val)
                else:
                    # COCOeval enabled but not computed this epoch (e.g., coco_eval_every > 1)
                    if do_coco:
                        print(f"📊 Val loss: {val_loss:.4f} | Val OKS: {val_oks:.3f} | COCO-AP(kpt): n/a")
                    else:
                        print(f"📊 Val loss: {val_loss:.4f} | Val OKS: {val_oks:.3f}")
                    metric_name = best_metric_name or "coco_ap_keypoints"
                    metric_val = best_metric if best_metric is not None else float("nan")
                    is_best = False
            else:
                print(f"📊 Val loss: {val_loss:.4f} | Val OKS: {val_oks:.3f}")
                metric_name = "oks"
                metric_val = float(val_oks)
                is_best = best_metric is None or float(metric_val) > float(best_metric)
                if is_best:
                    best_metric = float(metric_val)
                    best_metric_name = metric_name
                    best_oks = float(metric_val)
        else:
            print(f"⏭️ Skipping validation this epoch (val_every={val_every})")
            metric_name = best_metric_name or "oks"
            metric_val = best_metric if best_metric is not None else float("nan")
            is_best = False

        # Epoch wall-clock summary (always)
        epoch_wall_sec = time.time() - epoch_wall_start
        completed_epoch_times_sec.append(epoch_wall_sec)
        mean_epoch = sum(completed_epoch_times_sec) / float(len(completed_epoch_times_sec))
        total_elapsed = int(time.time() - run_start_time)
        epochs_done_now = (epoch + 1) - start_epoch
        epochs_left_now = max(0, epochs - (epoch + 1))
        eta_sec_now = int(mean_epoch * epochs_left_now)
        print(
            f"⏲️ Epoch time: {timedelta(seconds=int(epoch_wall_sec))} | "
            f"avg/epoch: {timedelta(seconds=int(mean_epoch))} | "
            f"elapsed: {timedelta(seconds=total_elapsed)} | "
            f"ETA left: {timedelta(seconds=eta_sec_now)}"
        )

        # Save last again after (optional) validation so it includes the latest best-metric bookkeeping.
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_metric": best_metric,
                "best_metric_name": best_metric_name or metric_name,
                "best_oks": best_oks,
                "best_coco_ap": best_coco_ap,
            },
            last_path,
        )

        # Save periodic checkpoint (always; independent of validation schedule)
        if save_every > 0 and ((epoch + 1) % save_every == 0):
            ckpt_path = out_dir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_metric": best_metric,
                    "best_metric_name": best_metric_name or metric_name,
                    "best_oks": best_oks,
                    "best_coco_ap": best_coco_ap,
                },
                ckpt_path,
            )

        # Save best checkpoint (only meaningful when we validated and computed controlling metric)
        if save_best and did_validate and is_best:
            best_path = out_dir / "best.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_metric": best_metric,
                    "best_metric_name": best_metric_name or metric_name,
                    "best_oks": best_oks,
                    "best_coco_ap": best_coco_ap,
                },
                best_path,
            )
            # best_metric can be None if we didn't compute the controlling metric this epoch
            try:
                best_str = f"{float(best_metric):.4f}"
            except Exception:
                best_str = "n/a"
            print(f"🏆 New best ({metric_name}={best_str}) saved to {best_path}")


if __name__ == "__main__":
    main()


