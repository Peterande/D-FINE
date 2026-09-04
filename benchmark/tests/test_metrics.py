from __future__ import annotations

import unittest

from benchmark.metrics.core import evaluate_records, protocol_fingerprint
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


if __name__ == "__main__":
    unittest.main()
