import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class TrainingEntrypointSourceTest(unittest.TestCase):
    def test_checkout_wins_over_conflicting_pythonpath_package(self):
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            fake_harl = Path(directory) / "harl"
            fake_harl.mkdir()
            (fake_harl / "__init__.py").write_text(
                "raise RuntimeError('wrong HARL checkout imported')\n",
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment["PYTHONPATH"] = directory
            result = subprocess.run(
                [sys.executable, str(repo_root / "examples" / "train.py"), "--help"],
                cwd=directory,
                env=environment,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(str(repo_root / "harl" / "__init__.py"), result.stdout)


if __name__ == "__main__":
    unittest.main()
