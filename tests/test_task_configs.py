from __future__ import annotations

import unittest

from snn_bench.tasks.performance_realism import load_task_configs


class TaskConfigTest(unittest.TestCase):
    def test_task_configs_present(self):
        configs = load_task_configs("snn_bench/configs/tasks")
        names = {c.task_name for c in configs}
        self.assertIn("direction_5m_distribution", names)
        self.assertIn("direction_30m_distribution", names)
        self.assertIn("realized_vol_30m", names)
        self.assertIn("options_iv_skew_movement", names)


if __name__ == "__main__":
    unittest.main()
