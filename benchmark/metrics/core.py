from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Iterable

import numpy as np


def protocol_fingerprint(protocol: dict[str, Any]) -> str:
    payload = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def _latency(records: list[dict[str, Any]]) -> dict[str, Any]:
    stages = ("preprocess_ms", "inference_ms", "postprocess_ms", "end_to_end_ms")
    output: dict[str, Any] = {}
    for stage in stages:
        values = [float(r["timing"][stage]) for r in records if stage in r.get("timing", {})]
        output[stage] = {
            "count": len(values),
            "p50": _percentile(values, 50),
            "p95": _percentile(values, 95),
            "mean": float(np.mean(values)) if values else None,
        }
    memory = [float(r["timing"]["peak_memory_mb"]) for r in records if "peak_memory_mb" in r.get("timing", {})]
    output["peak_memory_mb"] = max(memory) if memory else None
    return output


def _segmentation(records: list[dict[str, Any]], classes: int = 7) -> dict[str, Any]:
    confusion = np.zeros((classes, classes), dtype=np.int64)
    for record in records:
        pred = record.get("segmentation_pred")
        target = record.get("segmentation_target")
        if pred is None or target is None:
            continue
        pred_arr = np.asarray(pred, dtype=np.int64).reshape(-1)
        target_arr = np.asarray(target, dtype=np.int64).reshape(-1)
        valid = (target_arr >= 0) & (target_arr < classes) & (pred_arr >= 0) & (pred_arr < classes)
        confusion += np.bincount(classes * target_arr[valid] + pred_arr[valid], minlength=classes * classes).reshape(classes, classes)
    intersection = np.diag(confusion)
    union = confusion.sum(1) + confusion.sum(0) - intersection
    iou = np.divide(intersection, union, out=np.full(classes, np.nan), where=union > 0)
    return {
        "confusion_matrix": confusion.tolist(),
        "per_class_iou": [None if np.isnan(v) else float(v) for v in iou],
        "miou_present_classes": float(np.nanmean(iou)) if np.any(union > 0) else None,
        "pixel_accuracy": float(intersection.sum() / confusion.sum()) if confusion.sum() else None,
    }


def _hits(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    correct_part = comparable_part = 0
    for record in records:
        pred = record.get("hit_pred")
        target = record.get("hit_target")
        if pred is None or target is None:
            continue
        pred_hit, target_hit = bool(pred.get("hit")), bool(target.get("hit"))
        if pred_hit and target_hit:
            counts["tp"] += 1
        elif pred_hit:
            counts["fp"] += 1
        elif target_hit:
            counts["fn"] += 1
        else:
            counts["tn"] += 1
        sources[str(pred.get("source") or "unknown")] += 1
        if pred_hit and target_hit and target.get("body_part_id") is not None:
            comparable_part += 1
            correct_part += int(pred.get("body_part_id") == target.get("body_part_id"))
    precision = counts["tp"] / max(1, counts["tp"] + counts["fp"])
    recall = counts["tp"] / max(1, counts["tp"] + counts["fn"])
    return {
        **{key: counts[key] for key in ("tp", "fp", "fn", "tn")},
        "precision": precision,
        "recall": recall,
        "body_part_accuracy": correct_part / comparable_part if comparable_part else None,
        "decision_sources": dict(sorted(sources.items())),
    }


def evaluate_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = list(records)
    return {
        "sample_count": len(materialized),
        "segmentation": _segmentation(materialized),
        "product_hits": _hits(materialized),
        "runtime": _latency(materialized),
        "external_metrics": {
            "detection": "COCO AP and domain recall are imported from a pinned evaluator report",
            "pose": "COCO OKS AP is imported from a pinned evaluator report",
        },
    }
