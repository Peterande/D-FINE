#!/usr/bin/env python3
"""Build reproducible encoder-retention candidates from reconstructed phase 1 and deployed phase 2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state(path: Path) -> dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj["model"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1", required=True, type=Path)
    parser.add_argument("--phase2", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--alphas", default="0,0.1,0.25,0.5,0.75,1")
    args = parser.parse_args()
    phase1, phase2 = state(args.phase1), state(args.phase2)
    if phase1.keys() != phase2.keys():
        raise RuntimeError("phase checkpoint state keys differ")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "phase1": {"path": str(args.phase1.resolve()), "sha256": sha256(args.phase1)},
        "phase2": {"path": str(args.phase2.resolve()), "sha256": sha256(args.phase2)},
        "meaning": "encoder = (1-alpha)*phase1 + alpha*phase2; all non-encoder tensors use phase2",
        "candidates": [],
    }
    for alpha in [float(value) for value in args.alphas.split(",")]:
        if not 0 <= alpha <= 1:
            raise ValueError(alpha)
        merged = {}
        for key, value2 in phase2.items():
            if key.startswith("encoder.") and torch.is_floating_point(value2):
                value1 = phase1[key]
                merged[key] = value1.mul(1.0 - alpha).add(value2, alpha=alpha)
            elif key.startswith("encoder."):
                merged[key] = phase1[key] if alpha < 0.5 else value2
            else:
                merged[key] = value2
        name = f"encoder_alpha_{alpha:g}.pth"
        path = args.out_dir / name
        torch.save({"model": merged, "meta": {"encoder_alpha": alpha}}, path)
        manifest["candidates"].append({"alpha": alpha, "path": str(path.resolve()), "sha256": sha256(path)})
        print(name, manifest["candidates"][-1]["sha256"])
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
