"""Headless comparison entry point, Colab notebook and release version checks."""

import json
import sys
import tomllib
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from compare import PRESETS, _can_encode_mp4, main, parse_case, run_case
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

    def test_parse_case_types_values_like_config(self):
        import argparse

        base = SimulationConfig()
        name, overrides = parse_case("soft:poisson_ratio=0.4,boundary_type=fixed,boundary_width=3", base)
        self.assertEqual(name, "soft")
        self.assertEqual(overrides, {"poisson_ratio": 0.4, "boundary_type": "fixed", "boundary_width": 3})
        self.assertEqual(parse_case("x:gif_export=FALSE", base)[1], {"gif_export": False})
        self.assertEqual(parse_case("x:gif_export=true", base)[1], {"gif_export": True})
        for bad in ("x:nope=1", "x:poisson_ratio=abc", ":poisson_ratio=0.3", "x:poisson_ratio", "", ":", "x",
                    "x:", "x:,,", "x:poisson_ratio=0.3,", "x:gif_export=garbage"):
            with self.subTest(bad=bad), self.assertRaises(argparse.ArgumentTypeError):
                parse_case(bad, base)

    def test_bad_case_is_an_argument_error(self):
        with self.assertRaises(SystemExit):
            main(["--case", "x:nope=1"])

    def test_duplicate_case_names_are_rejected(self):
        with self.assertRaises(SystemExit):
            main(["--case", "x:poisson_ratio=0.3", "--case", "x:poisson_ratio=0.4"])

    def test_render_uses_each_cases_own_geometry(self):
        import tempfile
        from dataclasses import replace

        import matplotlib.pyplot as plt

        from compare import render

        configs = {"default": SimulationConfig(), "big": replace(SimulationConfig(), nx=21, dx=0.2)}
        cases = {n: run_case(c, steps=2, frame_interval=1) for n, c in configs.items()}
        seen = []
        original = plt.subplots

        def spy(*args, **kwargs):
            fig, axes = original(*args, **kwargs)
            seen.append(axes)
            return fig, axes

        with tempfile.TemporaryDirectory() as tmp, unittest.mock.patch("matplotlib.pyplot.subplots", spy):
            render(cases, configs, 2, 1, 5, Path(tmp) / "c")
        extents = [ax.images[0].get_extent() for ax in seen[0][:, 0]]
        self.assertAlmostEqual(extents[0][1], 100 * SimulationConfig().dx)
        self.assertAlmostEqual(extents[1][1], 20 * 0.2)

    def test_custom_cases_replace_preset(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            out = main(["--steps", "4", "--case", "a:poisson_ratio=0.3", "--case", "b:boundary_type=fixed",
                        "--out", str(Path(tmp) / "c")])
            self.assertGreater(out.stat().st_size, 0)

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
