#!/usr/bin/env python3
"""Rebuild the seven-class Pascal Person Parts layout used by this project.

The original prepared directory is unavailable. This converter consumes the official PASCAL-Part
MAT annotations and VOC2010 images and writes a new, provenance-recorded dataset. It does not claim
byte identity with the lost preparation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat


LABELS = {
    0: "background",
    1: "head",
    2: "torso",
    3: "arms",
    4: "hands",
    5: "legs",
    6: "feet",
}
PART_TO_LABEL = {
    "hair": 1, "head": 1, "lear": 1, "leye": 1, "lebrow": 1, "mouth": 1,
    "neck": 1, "nose": 1, "rear": 1, "reye": 1, "rebrow": 1,
    "torso": 2,
    "luarm": 3, "llarm": 3, "ruarm": 3, "rlarm": 3,
    "lhand": 4, "rhand": 4,
    "luleg": 5, "llleg": 5, "ruleg": 5, "rlleg": 5,
    "lfoot": 6, "rfoot": 6,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def positive_ids(path: Path) -> list[str]:
    ids = []
    for line in path.read_text().splitlines():
        image_id, value = line.split()
        if int(value) == 1:
            ids.append(image_id)
    return sorted(ids)


def convert(annotation: Path) -> np.ndarray | None:
    anno = loadmat(annotation, squeeze_me=True, struct_as_record=False)["anno"]
    output = None
    found_person = False
    for obj in np.atleast_1d(anno.objects):
        if str(getattr(obj, "class")) != "person":
            continue
        found_person = True
        for part in np.atleast_1d(obj.parts):
            part_name = str(part.part_name)
            label = PART_TO_LABEL.get(part_name)
            if label is None:
                raise ValueError(f"unknown person part {part_name!r} in {annotation}")
            part_mask = np.asarray(part.mask, dtype=bool)
            if output is None:
                output = np.zeros(part_mask.shape, dtype=np.uint8)
            output[part_mask] = label
    return output if found_person else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts-root", required=True, type=Path)
    parser.add_argument("--voc-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--parts-archive-sha256", required=True)
    parser.add_argument("--voc-archive-sha256", required=True)
    args = parser.parse_args()

    annotations = args.parts_root / "Annotations_Part"
    images = args.voc_root / "JPEGImages"
    image_sets = args.voc_root / "ImageSets" / "Main"
    summary = {"schema_version": 1, "labels": LABELS, "splits": {}, "sources": {
        "pascal_parts_trainval_sha256": args.parts_archive_sha256,
        "voc2010_trainval_sha256": args.voc_archive_sha256,
    }}

    split_ids: dict[str, set[str]] = {}
    for split in ("train", "val"):
        candidates = positive_ids(image_sets / f"person_{split}.txt")
        written = []
        image_out = args.out / "images" / split
        mask_out = args.out / "masks" / split
        image_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)
        for image_id in candidates:
            annotation = annotations / f"{image_id}.mat"
            source_image = images / f"{image_id}.jpg"
            if not annotation.is_file() or not source_image.is_file():
                continue
            mask = convert(annotation)
            if mask is None:
                continue
            shutil.copyfile(source_image, image_out / source_image.name)
            Image.fromarray(mask, mode="L").save(mask_out / f"{image_id}.png")
            written.append(image_id)
        split_ids[split] = set(written)
        summary["splits"][split] = {
            "count": len(written),
            "ordered_ids": written,
            "ordered_ids_sha256": hashlib.sha256(("\n".join(written) + "\n").encode()).hexdigest(),
        }

    overlap = sorted(split_ids["train"] & split_ids["val"])
    if overlap:
        raise RuntimeError(f"train/val leakage: {overlap[:10]}")
    summary_path = args.out / "provenance.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote train={summary['splits']['train']['count']} val={summary['splits']['val']['count']}")
    print(f"provenance={summary_path} sha256={sha256(summary_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
