"""Headless comparison entry point, Colab notebook and release version checks."""

import json
import sys
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from compare import PRESETS, main, run_case
from config import SimulationConfig


class CompareTests(unittest.TestCase):
    def test_run_case_frames_are_finite(self):
        frames = run_case(SimulationConfig(), steps=6, frame_interval=2)
        self.assertEqual(frames.shape, (3, 20, 101))
        self.assertTrue(np.isfinite(frames).all())
        self.assertGreater(frames.max(), 0)

    def test_presets_are_valid_configs(self):
        from dataclasses import replace

        for cases in PRESETS.values():
            for overrides in cases.values():
                replace(SimulationConfig(), **overrides).validate()

    def test_main_writes_video(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            out = main(["--steps", "6", "--out", str(Path(tmp) / "c")])
            self.assertGreater(out.stat().st_size, 0)

    def test_version_matches_pyproject(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
        self.assertEqual((ROOT / "VERSION").read_text().strip(), project["version"])
        self.assertTrue((ROOT / "docs" / "release-notes" / f"v{project['version']}.md").exists())

    def test_notebook_runs_existing_entry_point(self):
        nb = json.loads((ROOT / "notebooks" / "colab_compare.ipynb").read_text())
        code = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        compile(code.replace("#@param", "#"), "notebook", "exec")
        self.assertIn("run_compare.py", code)
        self.assertTrue((ROOT / "run_compare.py").exists())
