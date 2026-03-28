from __future__ import annotations

import unittest

from snn_bench.multistream.train import _walk_forward_indices


class MultiStreamWalkForwardSplitTest(unittest.TestCase):
    def test_reduces_fold_count_for_small_datasets(self):
        splits = _walk_forward_indices(n=24, folds=4, train_ratio=0.7, val_ratio=0.1)
        self.assertEqual(len(splits), 2)
        for train_idx, val_idx, test_idx in splits:
            self.assertGreater(len(train_idx), 0)
            self.assertGreater(len(val_idx), 0)
            self.assertGreater(len(test_idx), 0)

    def test_empty_for_too_small_series(self):
        self.assertEqual(_walk_forward_indices(n=2, folds=4, train_ratio=0.7, val_ratio=0.1), [])


if __name__ == "__main__":
    unittest.main()
