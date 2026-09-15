"""Control run: official DETRPose-X through this harness.

The detection numbers in MODEL_BASELINE.md are trustworthy because evaluating
stock D-FINE-X through the same code returned 59.3 AP, exactly matching the
published figure. The pose number (48.2 OKS AP) has no such control, so it
cannot yet be distinguished from a harness fault.

This runs the official DETRPose-X checkpoint, which the authors report at
73.3 AP, along two paths:

  official  official model + the DETRPose repo's own postprocessor
  mine      official model + this project's DETRPosePostProcessor

Reading the result:
  both ~73  scoring and postprocessing are both sound; 48.2 is a real gap
  official ~73, mine ~48   the bug is in our postprocessor, not the model
  both low  the fault is in the COCO scoring path or the data

DETRPose lives outside this worktree and is gitignored; it is imported from the
main checkout read-only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
for p in [REPO, REPO / "src", REPO / "pose_estimation_berna"]:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

COCO_PERSON_ID = 1


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detrpose-root", default="/home/berna/D-FINE-NEWBRINGER/DETRPose")
    ap.add_argument("--config", default="configs/detrpose/detrpose_hgnetv2_x.py",
                    help="Relative to --detrpose-root.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--coco-root", required=True)
    ap.add_argument("--split", default="val2017")
    ap.add_argument("--image-size", type=int, default=640)
    ap.add_argument("--path", choices=["official", "mine", "both"], default="both")
    ap.add_argument("--max-dets", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--swap-backbone-encoder", default=None, metavar="MERGED_CKPT",
                    help="Transplant experiment: keep the official pose decoder but replace "
                         "the backbone and encoder weights with those from the merged "
                         "production checkpoint. Both stacks are architecturally identical "
                         "(650/650 backbone keys, 546/546 encoder keys), so this measures "
                         "how well the official decoder reads D-FINE's features.")
    ap.add_argument("--out-prefix", default="runs/pose_control")
    return ap.parse_args()


def build_official(args, device):
    """Instantiate the official DETRPose model and postprocessor from its own code.

    Both repositories ship a top-level `src` package, so DETRPose has to own the
    name while its model is built: its root goes to the front of sys.path and any
    already-imported `src*` modules are dropped from the cache first. The
    previous state is restored on the way out so our own imports keep working.
    """
    root = Path(args.detrpose_root).resolve()
    saved_path = list(sys.path)
    saved_mods = {k: v for k, v in sys.modules.items() if k == "src" or k.startswith("src.")}
    for k in saved_mods:
        del sys.modules[k]
    sys.path = [str(root)] + [p for p in sys.path if Path(p).resolve() != REPO]

    # DETRPose's package __init__ chain pulls in its CrowdPose dataset, which needs
    # xtcocotools. That wheel does not build here, and nothing on the model-construction
    # path uses it — only dataset loading does, which never runs. Alias the one symbol
    # it imports to pycocotools so the import chain completes.
    if "xtcocotools" not in sys.modules:
        import types
        from pycocotools.coco import COCO as _COCO
        from pycocotools.cocoeval import COCOeval as _COCOeval

        pkg = types.ModuleType("xtcocotools")
        mod_coco = types.ModuleType("xtcocotools.coco")
        mod_coco.COCO = _COCO
        mod_eval = types.ModuleType("xtcocotools.cocoeval")
        mod_eval.COCOeval = _COCOeval
        pkg.coco, pkg.cocoeval = mod_coco, mod_eval
        sys.modules.update({"xtcocotools": pkg,
                            "xtcocotools.coco": mod_coco,
                            "xtcocotools.cocoeval": mod_eval})

    from src.core import LazyConfig, instantiate  # DETRPose's core, not ours

    cfg = LazyConfig.load(str(root / args.config))
    if hasattr(cfg.model.backbone, "pretrained"):
        cfg.model.backbone.pretrained = False

    model = instantiate(cfg.model)
    postprocessor = instantiate(cfg.postprocessor)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]
    model.load_state_dict(state)

    if args.swap_backbone_encoder:
        merged = torch.load(args.swap_backbone_encoder, map_location="cpu", weights_only=False)
        merged = merged.get("model", merged) if isinstance(merged, dict) else merged
        swapped = {}
        for prefix in ("backbone.", "encoder."):
            for k, v in merged.items():
                if k.startswith(prefix):
                    swapped[k] = v
        missing, unexpected = model.load_state_dict(swapped, strict=False)
        replaced = len(swapped) - len(unexpected)
        print(f"[swap] replaced {replaced}/{len(swapped)} backbone+encoder tensors "
              f"from {Path(args.swap_backbone_encoder).name} "
              f"(unmatched={len(unexpected)})")

    model = model.deploy().to(device).eval()
    post = postprocessor.deploy().to(device).eval()

    # Hand the `src` name back to our repo.
    sys.path = saved_path
    for k in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
        del sys.modules[k]
    sys.modules.update(saved_mods)

    return model, post


def score(results, coco, img_ids, tag, out_json):
    from pycocotools.cocoeval import COCOeval

    if not results:
        print(f"[{tag}] no detections — nothing to score")
        return None
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f)
    dt = coco.loadRes(out_json)
    ev = COCOeval(coco, dt, "keypoints")
    ev.params.imgIds = img_ids
    ev.evaluate()
    ev.accumulate()
    print(f"\n===== {tag} =====")
    ev.summarize()
    return float(ev.stats[0])


def main():
    args = parse_args()
    from pycocotools.coco import COCO

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    root = Path(args.coco_root)
    coco = COCO(str(root / "annotations" / f"person_keypoints_{args.split}.json"))
    img_dir = root / args.split
    img_ids = sorted(coco.getImgIds(catIds=[COCO_PERSON_ID]))
    if args.limit:
        img_ids = img_ids[: args.limit]

    model, official_post = build_official(args, device)

    mine_post = None
    if args.path in ("mine", "both"):
        from pose_estimation_berna.core.postprocess_detrpose import DETRPosePostProcessor
        mine_post = DETRPosePostProcessor(
            num_classes=1, num_keypoints=17, num_top_queries=300,
            remap_mscoco_category=False,
        ).to(device).eval()

    tfm = T.Compose([T.Resize((args.image_size, args.image_size)), T.ToTensor()])
    res_official, res_mine = [], []
    t0 = time.perf_counter()

    for i, img_id in enumerate(img_ids):
        info = coco.loadImgs(img_id)[0]
        img = Image.open(img_dir / info["file_name"]).convert("RGB")
        w0, h0 = img.size
        x = tfm(img).unsqueeze(0).to(device)
        orig = torch.tensor([[w0, h0]], device=device)

        with torch.inference_mode():
            raw = model(x)

            if args.path in ("official", "both"):
                scores, labels, kpts = official_post(raw, orig)
                s = scores[0].cpu().numpy()
                k = kpts[0].cpu().numpy()
                for idx in s.argsort()[::-1][: args.max_dets]:
                    flat = []
                    for j in range(k.shape[1]):
                        flat += [float(k[idx, j, 0]), float(k[idx, j, 1]), 1.0]
                    res_official.append({"image_id": int(img_id), "category_id": COCO_PERSON_ID,
                                         "keypoints": flat, "score": float(s[idx])})

            if mine_post is not None:
                # The official model emits keypoints as [B,Q,K,2]; our postprocessor
                # was written for the merged model's flat [B,Q,2K]. Same values,
                # different packing — flatten so the comparison is like for like.
                kp = raw["pred_keypoints"]
                if kp.ndim == 4:
                    kp = kp.flatten(2)
                out = mine_post({"pred_logits": raw["pred_logits"],
                                 "pred_keypoints": kp}, orig)[0]
                s = out["scores"].cpu().numpy()
                k = out["keypoints"].cpu().numpy()
                for idx in s.argsort()[::-1][: args.max_dets]:
                    flat = []
                    for j in range(k.shape[1]):
                        flat += [float(k[idx, j, 0]), float(k[idx, j, 1]), 1.0]
                    res_mine.append({"image_id": int(img_id), "category_id": COCO_PERSON_ID,
                                     "keypoints": flat, "score": float(s[idx])})

        if (i + 1) % 500 == 0:
            print(f"[control] {i + 1}/{len(img_ids)}", flush=True)

    print(f"[control] {len(img_ids)} images in {time.perf_counter() - t0:.1f}s")

    ap_off = ap_mine = None
    if args.path in ("official", "both"):
        ap_off = score(res_official, coco, img_ids, "official postprocessor",
                       f"{args.out_prefix}_official.json")
    if mine_post is not None:
        ap_mine = score(res_mine, coco, img_ids, "our postprocessor",
                        f"{args.out_prefix}_mine.json")

    print("\n----- verdict -----")
    print(f"published DETRPose-X          73.3")
    if ap_off is not None:
        print(f"official postprocessor here   {ap_off * 100:.1f}")
    if ap_mine is not None:
        print(f"our postprocessor here        {ap_mine * 100:.1f}")
    print(f"production merged pose head   48.2")


if __name__ == "__main__":
    main()
