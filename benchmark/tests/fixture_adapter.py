from __future__ import annotations


def run(*, dataset, model, settings):
    del dataset, model, settings
    return [
        {
            "sample_id": "fixture-1",
            "segmentation_pred": [[0, 1], [2, 2]],
            "segmentation_target": [[0, 1], [2, 1]],
            "hit_pred": {"hit": True, "body_part_id": 2, "source": "mask"},
            "hit_target": {"hit": True, "body_part_id": 2},
            "timing": {"preprocess_ms": 1, "inference_ms": 5, "postprocess_ms": 2, "end_to_end_ms": 8, "peak_memory_mb": 100},
        },
        {
            "sample_id": "fixture-2",
            "segmentation_pred": [[0, 0], [2, 2]],
            "segmentation_target": [[0, 0], [2, 2]],
            "hit_pred": {"hit": False, "body_part_id": None, "source": "mask"},
            "hit_target": {"hit": True, "body_part_id": 1},
            "timing": {"preprocess_ms": 2, "inference_ms": 7, "postprocess_ms": 2, "end_to_end_ms": 11, "peak_memory_mb": 120},
        },
    ]
