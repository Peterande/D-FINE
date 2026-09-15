#!/usr/bin/env python3
"""Run any conforming model adapter and write raw outputs plus a normalized report."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

from benchmark.metrics import evaluate_records, protocol_fingerprint


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def load_adapter(spec: str):
    module_name, separator, symbol = spec.partition(":")
    if not separator:
        raise ValueError("adapter must use module:function syntax")
    return getattr(importlib.import_module(module_name), symbol)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="module:function returning iterable raw records")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--settings", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    dataset = json.loads(args.manifest.read_text())
    model = json.loads(args.model_manifest.read_text())
    settings = json.loads(args.settings.read_text())
    protocol = {
        "schema_version": 1,
        "dataset_id": dataset["dataset_id"],
        "dataset_manifest_sha256": file_sha256(args.manifest),
        "settings": settings,
    }
    fingerprint = protocol_fingerprint(protocol)

    adapter = load_adapter(args.adapter)
    records = list(adapter(dataset=dataset, model=model, settings=settings))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / "raw.jsonl"
    with raw_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    report = {
        "schema_version": 1,
        "protocol_fingerprint": fingerprint,
        "protocol": protocol,
        "model": {
            "id": model["baseline_id"],
            "manifest_sha256": file_sha256(args.model_manifest),
        },
        "raw_output": {"path": raw_path.name, "sha256": file_sha256(raw_path)},
        "code": {"git_revision": git_revision()},
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "metrics": evaluate_records(records),
        "claim_type": "locally_measured",
    }
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"OK: samples={len(records)} protocol={fingerprint} report={args.out_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
