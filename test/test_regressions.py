"""Numerical and output regressions; run with unittest discovery."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np
from PIL import Image

from config import SimulationConfig
from main import _initialize_simulation
from solver import FiniteDifferenceOperator
from gif_exporter import GifExporter
from visualizer import WaveVisualizer


class NumericalTests(unittest.TestCase):
    def test_nan_in_either_component_is_detected(self):
        config = SimulationConfig()
        state, solver = _initialize_simulation(config, "P")
        for component in ("u", "v"):
            state.reset()
            setattr(state, component, mx.full(state.u.shape, float("nan")))
            self.assertTrue(np.isnan(state.get_max_displacement()))
            self.assertTrue(solver.check_stability(state))

    def test_cfl_uses_fastest_speed_and_both_spacings(self):
        config = SimulationConfig(dx=0.02, dy=0.001, poisson_ratio=0.49)
        limit = np.sqrt(3) / (2 * config.p_wave_speed * np.hypot(1 / config.dx, 1 / config.dy))
        self.assertAlmostEqual(config.dt, config.cfl_safety_factor * limit)

    def test_invalid_configs_are_rejected(self):
        for options in ({"dx": np.nan}, {"density": np.inf}, {"nx": 4},
                        {"boundary_width": 0}, {"plot_interval": 0},
                        {"gif_frame_interval": 0}, {"steps": 1.5},
                        {"implicit_iterations": 0}, {"cfl_safety_factor": 1.1},
                        {"source_x_end": 200}, {"finite_diff_order": 2},
                        {"source_mode": "typo"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                SimulationConfig(**options).validate()

    def test_derivatives_and_operator_reuse(self):
        fd = FiniteDifferenceOperator(0.02, 0.03)
        for ny, nx in ((13, 17), (15, 21)):
            y, x = np.mgrid[:ny, :nx]
            field = ((x * fd.dx)**2 + (y * fd.dy)**2 + x * fd.dx * y * fd.dy).astype(np.float32)
            u = mx.array(field[None, :, :, None])
            derivatives = fd.compute_derivatives(u, u)
            for derivative, expected in zip(derivatives, (2, 2, 1, 2, 2, 1)):
                self.assertEqual(derivative.shape, u.shape)
                np.testing.assert_allclose(np.array(derivative)[0, 2:-2, 2:-2, 0], expected, atol=3e-4)

    def test_free_boundaries_preserve_rigid_motion(self):
        config = SimulationConfig(nx=21, ny=17, dx=0.03, dy=0.02, source_mode="interior")
        state, solver = _initialize_simulation(config, "P")
        y, x = np.mgrid[:config.ny, :config.nx]
        # A translation plus infinitesimal rotation has identically zero strain.
        u = mx.array((0.001 - 0.002 * y * config.dy).astype(np.float32)[None, :, :, None])
        v = mx.array((-0.001 + 0.002 * x * config.dx).astype(np.float32)[None, :, :, None])
        for derivative in solver.fd.compute_derivatives(u, v, solver.bc):
            np.testing.assert_allclose(np.array(derivative), 0, atol=8e-6)
        state.u, state.v, state.u_prev, state.v_prev = u, v, u, v
        solver.source.amplitude = 0
        solver.step(state, 0)
        np.testing.assert_allclose(np.array(state.u), np.array(u), atol=2e-9)
        np.testing.assert_allclose(np.array(state.v), np.array(v), atol=2e-9)

    def test_source_uses_next_state_time(self):
        config = SimulationConfig(time_integration="explicit")
        state, solver = _initialize_simulation(config, "P")
        solver.step(state, 0)
        np.testing.assert_allclose(np.array(state.u)[0, :, 0, 0], solver.source.get_amplitude(config.dt), rtol=1e-6)

    def test_interior_source_respects_fixed_walls(self):
        config = SimulationConfig(nx=25, ny=17, boundary_width=2, source_x_start=6, source_x_end=8,
                                  source_mode="interior", boundary_type="fixed", time_integration="explicit")
        state, solver = _initialize_simulation(config, "P")
        solver.step(state, 0)
        field = np.array(state.u)[0, :, :, 0]
        np.testing.assert_array_equal(field[:2], 0)
        np.testing.assert_array_equal(field[-2:], 0)
        self.assertGreater(np.max(np.abs(field[2:-2, 6:8])), 0)

    def test_traction_closure_on_each_free_side(self):
        config = SimulationConfig(nx=21, ny=17, source_mode="interior")
        state, solver = _initialize_simulation(config, "P")
        y, x = np.mgrid[:config.ny, :config.nx]
        u = mx.array((np.sin(x/8) * np.cos(y/9)).astype(np.float32)[None, :, :, None])
        v = mx.array((np.cos(x/7) * np.sin(y/8)).astype(np.float32)[None, :, :, None])
        pu, pv = (np.array(a)[0, :, :, 0] for a in solver.bc.pad_fields(u, v))
        alpha = config.lame_lambda / (config.lame_lambda + 2*config.lame_mu)
        for row in (2, config.ny+1):
            uy = (pu[row-2]-8*pu[row-1]+8*pu[row+1]-pu[row+2])/(12*config.dy)
            vy = (pv[row-2]-8*pv[row-1]+8*pv[row+1]-pv[row+2])/(12*config.dy)
            ux = (pu[row, 3:-1]-pu[row, 1:-3])/(2*config.dx)
            vx = (pv[row, 3:-1]-pv[row, 1:-3])/(2*config.dx)
            # Corner closures are sequential approximations, not simultaneous traction constraints.
            np.testing.assert_allclose(uy[3:-3]+vx[1:-1], 0, atol=2e-5)
            np.testing.assert_allclose(vy[3:-3]+alpha*ux[1:-1], 0, atol=2e-5)
        for col in (2, config.nx+1):
            ux = (pu[:, col-2]-8*pu[:, col-1]+8*pu[:, col+1]-pu[:, col+2])/(12*config.dx)
            vx = (pv[:, col-2]-8*pv[:, col-1]+8*pv[:, col+1]-pv[:, col+2])/(12*config.dx)
            uy = (pu[3:-1, col]-pu[1:-3, col])/(2*config.dy)
            vy = (pv[3:-1, col]-pv[1:-3, col])/(2*config.dy)
            np.testing.assert_allclose(ux[3:-3]+alpha*vy[1:-1], 0, atol=2e-5)
            np.testing.assert_allclose(vx[3:-3]+uy[1:-1], 0, atol=2e-5)

    def test_time_integrators_converge_quadratically(self):
        for method in ("explicit", "implicit"):
            errors = []
            for count in (20, 40):
                config = SimulationConfig(dx=0.2, dy=0.2, damping=1, time_integration=method)
                config.cfl_safety_factor *= (1/count)/config.dt
                state, solver = _initialize_simulation(config, "P")
                state.u = mx.ones_like(state.u)
                state.u_prev = state.u * np.cos(3*config.dt)
                solver._accelerations = lambda u, v: (-9*u, -9*v)
                solver._apply_constraints = lambda u, v, t: (u, v)
                for step in range(count):
                    solver.step(state, step*config.dt)
                errors.append(abs(state.u[0, 0, 0, 0].item()-np.cos(3)))
            with self.subTest(method=method):
                self.assertGreater(errors[0]/errors[1], 3)
                self.assertLess(errors[0]/errors[1], 5)

    def test_implicit_failure_is_reported_without_advancing(self):
        config = SimulationConfig(implicit_iterations=1, implicit_tolerance=1e-15)
        state, solver = _initialize_simulation(config, "P")
        initial = np.array(state.u)
        with self.assertRaises(RuntimeError):
            solver.step(state, 0)
        np.testing.assert_array_equal(np.array(state.u), initial)


class OutputTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_nonfinite_outputs_fail_and_active_source_does_not_stop_early(self):
        config = SimulationConfig(time_integration="explicit", source_amplitude=0)
        state, solver = _initialize_simulation(config, "P")
        view = WaveVisualizer(config, state, solver)
        view._min_steps_before_check = 0
        for frame in range(6):
            view._animate(frame)
        self.assertEqual(view.current_step, 12)
        state.v = mx.full(state.v.shape, float("nan"))
        with self.assertRaises(RuntimeError):
            view._animate(7)
        with tempfile.TemporaryDirectory() as directory:
            config.gif_output_dir = directory
            exporter = GifExporter(config, state, solver)
            with self.assertRaises(RuntimeError):
                exporter._collect_frame_data()

    def test_visualizer_runs_exact_step_count_and_contours_use_metres(self):
        config = SimulationConfig(steps=3, plot_interval=2, time_integration="explicit")
        state, solver = _initialize_simulation(config, "P")
        view = WaveVisualizer(config, state, solver)
        view._animate(0)
        view._animate(1)
        self.assertEqual(view.current_step, 3)
        self.assertAlmostEqual(view.im.get_extent()[1], (config.nx - 1) * config.dx)
        vertices = np.concatenate([path.vertices for path in view.ax.collections[-1].get_paths() if path.vertices.size])
        self.assertLessEqual(vertices[:, 0].max(), (config.nx - 1) * config.dx + 1e-6)

    def test_gif_scale_frame_count_and_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            config = SimulationConfig(gif_num_steps=3, gif_frame_interval=2, gif_frame_rate=10,
                                      gif_output_dir=directory, time_integration="explicit")
            state, solver = _initialize_simulation(config, "P")
            exporter = GifExporter(config, state, solver)
            self.assertEqual(exporter.total_frames, 2)
            exporter.run()
            self.assertLess(exporter.p_wave_max, 0.01)
            self.assertAlmostEqual(exporter.p_wave_min, -exporter.p_wave_max)
            for name in ("p_wave_animation.gif", "s_wave_animation.gif", "combined_magnitude.gif"):
                with Image.open(Path(directory) / name) as gif:
                    self.assertEqual(gif.n_frames, 2)
                    self.assertEqual(gif.info["duration"], 100)


if __name__ == "__main__":
    unittest.main()
