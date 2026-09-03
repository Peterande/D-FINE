#!/usr/bin/env python3
"""Validate the frozen production-baseline manifest without modifying artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REQUIRED_EVIDENCE = {"measured", "artifact-derived", "inferred", "unknown"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "manifest",
        nargs="?",
        default=Path(__file__).with_name("production_manifest.json"),
        type=Path,
    )
    parser.add_argument("--skip-hashes", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    errors: list[str] = []

    if manifest.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if not manifest.get("baseline_id"):
        errors.append("baseline_id is required")

    for artifact in manifest.get("artifacts", []):
        evidence = artifact.get("evidence")
        if evidence not in REQUIRED_EVIDENCE:
            errors.append(f"{artifact.get('role')}: invalid evidence {evidence!r}")
        path_value = artifact.get("path")
        if not path_value:
            continue
        path = Path(path_value)
        if not path.is_file():
            errors.append(f"{artifact.get('role')}: missing {path}")
            continue
        if path.stat().st_size != artifact.get("size_bytes"):
            errors.append(f"{artifact.get('role')}: size mismatch")
        if not args.skip_hashes and sha256(path) != artifact.get("sha256"):
            errors.append(f"{artifact.get('role')}: sha256 mismatch")

    inputs = manifest.get("model_contract", {}).get("inputs", [])
    outputs = manifest.get("model_contract", {}).get("outputs", [])
    if [item.get("name") for item in inputs] != ["images"]:
        errors.append("model input contract changed")
    expected_outputs = [
        "det_pred_logits",
        "det_pred_boxes",
        "pose_pred_logits",
        "pose_pred_keypoints",
        "seg_logits",
    ]
    if [item.get("name") for item in outputs] != expected_outputs:
        errors.append("model output contract changed")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"OK: {manifest['baseline_id']} ({len(manifest['artifacts'])} artifacts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
