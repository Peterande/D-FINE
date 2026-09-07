from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from benchmark.pose_adapter.model import ResidualFeatureAdapters


class PoseAdapterTest(unittest.TestCase):
    def test_zero_initialization_is_exact_identity(self):
        adapters = ResidualFeatureAdapters([4, 8])
        features = [torch.randn(2, 4, 5, 7), torch.randn(2, 8, 3, 4)]
        outputs = adapters(features)
        for source, output in zip(features, outputs):
            self.assertTrue(torch.equal(source, output))

    def test_shape_and_gradient_are_local(self):
        adapters = ResidualFeatureAdapters([4])
        frozen = nn.Conv2d(3, 4, 1).requires_grad_(False)
        x = torch.randn(2, 3, 5, 5)
        loss = adapters([frozen(x)])[0].square().mean()
        loss.backward()
        self.assertIsNone(frozen.weight.grad)
        self.assertIsNotNone(adapters.projections[0].weight.grad)
        self.assertEqual(tuple(adapters([frozen(x)])[0].shape), (2, 4, 5, 5))

    def test_feature_level_count_is_checked(self):
        adapters = ResidualFeatureAdapters([4, 8])
        with self.assertRaises(ValueError):
            adapters([torch.randn(1, 4, 2, 2)])


if __name__ == "__main__":
    unittest.main()
