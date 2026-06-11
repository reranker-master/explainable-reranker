from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from explainable_reranker.config.env import load_project_dotenv


class EnvConfigTest(unittest.TestCase):
    def test_loads_env_and_local_without_overriding_shell(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text(
                "KEEP=shell\nBASE=base\nLOCAL=from_env\nQUOTED=\"hello world\"\n",
                encoding="utf-8",
            )
            (root / ".env.local").write_text("LOCAL=from_local\n", encoding="utf-8")

            old_env = dict(os.environ)
            try:
                os.environ.clear()
                os.environ.update({"KEEP": "already"})
                loaded = load_project_dotenv(root)
                self.assertEqual([path.name for path in loaded], [".env", ".env.local"])
                self.assertEqual(os.environ["KEEP"], "already")
                self.assertEqual(os.environ["BASE"], "base")
                self.assertEqual(os.environ["LOCAL"], "from_local")
                self.assertEqual(os.environ["QUOTED"], "hello world")
            finally:
                os.environ.clear()
                os.environ.update(old_env)


if __name__ == "__main__":
    unittest.main()
