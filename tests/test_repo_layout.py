from pathlib import Path
import unittest


class RepoLayoutTest(unittest.TestCase):
    def test_expected_dirs_exist(self):
        for rel in [
            "snn_bench/data_connectors",
            "snn_bench/feature_pipelines",
            "snn_bench/tasks",
            "snn_bench/models",
            "snn_bench/trainers",
            "snn_bench/eval",
            "snn_bench/configs",
            "snn_bench/scripts",
        ]:
            self.assertTrue(Path(rel).is_dir(), rel)


if __name__ == "__main__":
    unittest.main()
