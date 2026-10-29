"""Headless comparison entry point, Colab notebook and release version checks."""

import json
import sys
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from compare import PRESETS, _can_encode_mp4, main, run_case
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

    def test_mp4_needs_ffmpeg_with_libx264(self):
        from types import SimpleNamespace
        from unittest import mock

        with mock.patch("shutil.which", return_value=None):
            self.assertFalse(_can_encode_mp4())
        ok = SimpleNamespace(returncode=0, stdout="V..... libx264")
        cases = (
            (ok, True),
            (SimpleNamespace(returncode=0, stdout="V..... mpeg4"), False),
            (SimpleNamespace(returncode=1, stdout="V..... libx264"), False),
        )
        for result, expected in cases:
            with (
                mock.patch("shutil.which", return_value="/usr/bin/ffmpeg"),
                mock.patch("subprocess.run", return_value=result),
            ):
                self.assertEqual(_can_encode_mp4(), expected)
        import subprocess

        for error in (OSError("bad exe"), subprocess.TimeoutExpired("ffmpeg", 10)):
            with (
                mock.patch("shutil.which", return_value="/usr/bin/ffmpeg"),
                mock.patch("subprocess.run", side_effect=error),
            ):
                self.assertFalse(_can_encode_mp4())

    def test_runs_when_notebook_sets_inline_backend(self):
        import os
        import subprocess
        import tempfile

        env = {**os.environ, "MPLBACKEND": "module://matplotlib_inline.backend_inline"}
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, str(ROOT / "run_compare.py"), "--steps", "2", "--out", str(Path(tmp) / "c")],
                env=env, capture_output=True, text=True, check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_non_positive_arguments(self):
        for flag in ("--steps", "--frame-interval", "--fps"):
            with self.subTest(flag=flag), self.assertRaises(SystemExit):
                main([flag, "0"])

    def test_version_matches_pyproject(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        self.assertEqual((ROOT / "VERSION").read_text(encoding="utf-8").strip(), project["version"])
        self.assertTrue((ROOT / "docs" / "release-notes" / f"v{project['version']}.md").exists())

    def test_notebook_runs_existing_entry_point(self):
        nb = json.loads((ROOT / "notebooks" / "colab_compare.ipynb").read_text(encoding="utf-8"))
        code = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        compile(code.replace("#@param", "#"), "notebook", "exec")
        self.assertIn("run_compare.py", code)
        self.assertTrue((ROOT / "run_compare.py").exists())
