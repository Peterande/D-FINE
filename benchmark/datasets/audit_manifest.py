#!/usr/bin/env python3
"""Audit dataset manifests for identity, ordering, leakage, and annotation ranges."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", nargs="?", type=Path, default=Path(__file__).parent / "tagtwo_v1" / "manifest.json")
    parser.add_argument("--skip-hashes", action="store_true")
    args = parser.parse_args()
    data = json.loads(args.manifest.read_text())
    errors: list[str] = []

    sources = {item["id"]: item for item in data.get("sources", [])}
    source_groups: dict[str, str] = {}
    for source in sources.values():
        path = Path(source["path"])
        if not path.is_file():
            errors.append(f"missing source: {path}")
        elif not args.skip_hashes and sha256(path) != source["sha256"]:
            errors.append(f"source hash mismatch: {source['id']}")
        group = source["source_group"]
        split = source["split"]
        if group in source_groups and source_groups[group] != split:
            errors.append(f"split leakage for source group {group}")
        source_groups[group] = split

    for public in data.get("public_validation", []):
        if "annotation_path" in public:
            path = Path(public["annotation_path"])
            if not path.is_file():
                errors.append(f"missing annotations: {path}")
            elif not args.skip_hashes and sha256(path) != public["annotation_sha256"]:
                errors.append(f"annotation hash mismatch: {public['id']}")
        if "provenance_path" in public:
            path = Path(public["provenance_path"])
            if not path.is_file():
                errors.append(f"missing provenance: {path}")
            elif not args.skip_hashes and sha256(path) != public["provenance_sha256"]:
                errors.append(f"provenance hash mismatch: {public['id']}")
            else:
                provenance = json.loads(path.read_text())
                if provenance["splits"]["train"]["ordered_ids_sha256"] != public["train_order_sha256"]:
                    errors.append(f"train order mismatch: {public['id']}")
                if provenance["splits"]["val"]["ordered_ids_sha256"] != public["validation_order_sha256"]:
                    errors.append(f"validation order mismatch: {public['id']}")

    samples = data.get("samples", [])
    ids = [sample["sample_id"] for sample in samples]
    if ids != sorted(ids):
        errors.append("samples are not in declared lexicographic order")
    if len(ids) != len(set(ids)):
        errors.append("duplicate sample_id")

    scenario_counts: Counter[str] = Counter()
    for sample in samples:
        source = sources.get(sample["source_id"])
        if source is None:
            errors.append(f"unknown source for {sample['sample_id']}")
            continue
        if sample["split"] != source["split"]:
            errors.append(f"sample/source split mismatch: {sample['sample_id']}")
        if not 0 <= sample["frame_index"] < source["frame_count"]:
            errors.append(f"frame out of range: {sample['sample_id']}")
        x, y = sample["crosshair_xy"]
        if not (0 <= x < source["width"] and 0 <= y < source["height"]):
            errors.append(f"crosshair out of range: {sample['sample_id']}")
        scenario_counts.update(sample.get("scenarios", []))
        annotation = sample.get("annotation")
        if annotation:
            labels = annotation.get("body_part_labels", [])
            if any(not isinstance(value, int) or not 0 <= value <= 6 for value in labels):
                errors.append(f"body-part label outside 0..6: {sample['sample_id']}")

    required = set(data.get("coverage", {}).get("required", []))
    declared_missing = set(data.get("coverage", {}).get("missing", []))
    actual_missing = {scenario for scenario in required if scenario_counts[scenario] == 0}
    if declared_missing != actual_missing:
        errors.append(f"coverage mismatch: declared={sorted(declared_missing)} actual={sorted(actual_missing)}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    annotation_coverage = data["coverage"]["annotation_coverage"]
    print(f"OK: {data['dataset_id']} samples={len(samples)} missing_required_scenarios={len(actual_missing)} anatomical_masks={annotation_coverage['seven_class_masks']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
