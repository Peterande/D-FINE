from __future__ import annotations

import unittest

from benchmark.metrics.core import evaluate_records, protocol_fingerprint
from benchmark.harness.hit import BODY_PARTS, decide_hit
from benchmark.tests.fixture_adapter import run


class MetricsTest(unittest.TestCase):
    def test_golden_metrics(self):
        report = evaluate_records(run(dataset={}, model={}, settings={}))
        self.assertEqual(report["sample_count"], 2)
        self.assertEqual(report["product_hits"]["tp"], 1)
        self.assertEqual(report["product_hits"]["fn"], 1)
        self.assertEqual(report["product_hits"]["body_part_accuracy"], 1.0)
        self.assertEqual(report["runtime"]["inference_ms"]["p50"], 6.0)
        self.assertAlmostEqual(report["segmentation"]["pixel_accuracy"], 0.875)

    def test_protocol_is_order_independent(self):
        self.assertEqual(protocol_fingerprint({"a": 1, "b": 2}), protocol_fingerprint({"b": 2, "a": 1}))
        self.assertNotEqual(protocol_fingerprint({"a": 1}), protocol_fingerprint({"a": 2}))

    def test_authoritative_body_part_ids(self):
        self.assertEqual(
            BODY_PARTS,
            {0: "background", 1: "head", 2: "torso", 3: "arms", 4: "hands", 5: "legs", 6: "feet"},
        )

    def test_crosshair_uses_mask_coordinates(self):
        import numpy as np

        mask = np.zeros((640, 640), dtype=np.uint8)
        mask[319:322, 319:322] = 4
        verdict = decide_hit(mask, 640, 640, region_size=3)
        self.assertTrue(verdict["hit"])
        self.assertEqual(verdict["body_part"], "hands")


if __name__ == "__main__":
    unittest.main()
