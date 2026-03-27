from __future__ import annotations

import unittest

import numpy as np

from snn_bench.scripts.train import _assert_split_alignment, _split_dataset


class TrainGuardrailsTest(unittest.TestCase):
    def test_walk_forward_split_validity(self):
        x = np.random.randn(20, 4).astype(np.float32)
        y = (x[:, 0] > 0).astype(np.int64)
        idx = np.arange(len(x), dtype=np.int64)

        x_train, x_test, y_train, y_test, idx_train, idx_test = _split_dataset(
            x,
            y,
            idx,
            seed=7,
            split_mode="walk_forward",
            walk_forward_ratio=0.7,
        )

        _assert_split_alignment(
            x_train,
            x_test,
            y_train,
            y_test,
            idx_train,
            idx_test,
            split_mode="walk_forward",
        )
        self.assertLess(np.max(idx_train), np.min(idx_test))

    def test_overlap_detection_raises(self):
        x_train = np.zeros((4, 2), dtype=np.float32)
        x_test = np.zeros((3, 2), dtype=np.float32)
        y_train = np.zeros((4,), dtype=np.int64)
        y_test = np.zeros((3,), dtype=np.int64)
        idx_train = np.array([0, 1, 2, 3], dtype=np.int64)
        idx_test = np.array([3, 4, 5], dtype=np.int64)

        with self.assertRaises(ValueError):
            _assert_split_alignment(
                x_train,
                x_test,
                y_train,
                y_test,
                idx_train,
                idx_test,
                split_mode="walk_forward",
            )


if __name__ == "__main__":
    unittest.main()
