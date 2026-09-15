import unittest

import numpy as np

from benchmark.harness.infer_crosshair import nms_indices


class AssociationTest(unittest.TestCase):
    def test_nms_removes_duplicate_person_boxes(self):
        boxes = np.array(
            [[0, 0, 100, 100], [2, 2, 98, 98], [150, 0, 250, 100]],
            np.float32,
        )
        scores = np.array([0.9, 0.8, 0.7], np.float32)
        self.assertEqual(nms_indices(boxes, scores, 0.6).tolist(), [0, 2])

    def test_nms_handles_empty_input(self):
        boxes = np.zeros((0, 4), dtype=np.float32)
        scores = np.zeros((0,), dtype=np.float32)
        self.assertEqual(nms_indices(boxes, scores, 0.6).tolist(), [])


if __name__ == "__main__":
    unittest.main()
