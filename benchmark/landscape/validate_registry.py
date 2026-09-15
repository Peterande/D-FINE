#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def main() -> int:
    path = Path(__file__).with_name("registry.json")
    data = json.loads(path.read_text())
    entries = data["entries"]
    errors: list[str] = []
    counts = Counter(entry["category"] for entry in entries)
    for category in ("detection", "pose", "body_part_segmentation", "multitask"):
        if counts[category] < 5:
            errors.append(f"{category}: expected at least 5 entries, got {counts[category]}")
    required = {"id", "category", "source", "revision", "license", "task", "published", "local", "export", "fit", "decision", "reason"}
    for entry in entries:
        missing = required - entry.keys()
        if missing:
            errors.append(f"{entry.get('id')}: missing {sorted(missing)}")
        if entry.get("decision") not in data["decisions"]:
            errors.append(f"{entry.get('id')}: invalid decision")
        if not str(entry.get("source", "")).startswith(("https://", "local:")):
            errors.append(f"{entry.get('id')}: source must be primary URL or local anchor")
        if entry.get("published") is not None and entry.get("local") is not None and entry["published"] == entry["local"]:
            errors.append(f"{entry.get('id')}: published/local evidence appears conflated")
    if len({entry["id"] for entry in entries}) != len(entries):
        errors.append("duplicate candidate ids")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"OK: {len(entries)} candidates {dict(sorted(counts.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
