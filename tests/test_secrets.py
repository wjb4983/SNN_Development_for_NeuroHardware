import os
import tempfile
import unittest
from pathlib import Path

from snn_bench.utils.secrets import load_massive_api_key


class SecretsTest(unittest.TestCase):
    def test_load_from_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "api_key.txt"
            path.write_text("abc123\n", encoding="utf-8")
            os.environ.pop("MASSIVE_API_KEY", None)
            os.environ["MASSIVE_API_KEY_FILE"] = str(path)
            try:
                self.assertEqual(load_massive_api_key(), "abc123")
            finally:
                os.environ.pop("MASSIVE_API_KEY_FILE", None)


if __name__ == "__main__":
    unittest.main()
