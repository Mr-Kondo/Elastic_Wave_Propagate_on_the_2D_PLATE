"""Numerical and output regressions; run with unittest discovery."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from config import SimulationConfig
from gif_exporter import GifExporter
from main import _initialize_simulation
from solver import ElasticOperator
from state import WaveFieldState
from visualizer import WaveVisualizer


class NumericalTests(unittest.TestCase):
    def test_nan_in_either_component_is_detected(self):
        config = SimulationConfig()
        state, solver = _initialize_simulation(config, "P")
        for component in ("u", "v"):
            state.reset()
            setattr(state, component, np.full(state.u.shape, float("nan")))
            self.assertTrue(np.isnan(state.get_max_displacement()))
            self.assertTrue(solver.check_stability(state))

    def test_cfl_uses_fastest_speed_and_both_spacings(self):
        config = SimulationConfig(dx=0.02, dy=0.001, poisson_ratio=0.49)
        limit = np.sqrt(config.density/(2*(config.lame_lambda+config.lame_mu))) / np.hypot(1/config.dx, 1/config.dy)
        self.assertAlmostEqual(config.dt, config.cfl_safety_factor * limit)

    def test_invalid_configs_are_rejected(self):
        for options in ({"dx": np.nan}, {"density": np.inf}, {"nx": 1},
                        {"boundary_width": 0}, {"plot_interval": 0},
                        {"gif_frame_interval": 0}, {"steps": 1.5},
                        {"cfl_safety_factor": 1.1},
                        {"source_x_end": 200}, {"source_mode": "typo"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                SimulationConfig(**options).validate()

    def test_affine_strain_energy(self):
        config = SimulationConfig(nx=9, ny=7, source_mode="interior")
        operator = ElasticOperator(config)
        y, x = np.mgrid[:config.ny, :config.nx]
        xx, yy = x*config.dx, y*config.dy
        exx, eyy, shear = 0.03, -0.02, 0.01
        u = (exx*xx+shear*yy)[None, :, :, None]
        v = (eyy*yy)[None, :, :, None]
        values = operator.pack(u, v)
        area = (config.nx-1)*config.dx*(config.ny-1)*config.dy
        density = config.lame_lambda/2*(exx+eyy)**2 + config.lame_mu*(exx**2+eyy**2+shear**2/2)
        self.assertAlmostEqual(values @ (operator.stiffness @ values)/2, area*density, places=12)
        self.assertAlmostEqual(operator.mass.sum(), 2*config.density*area, places=12)

    def test_free_boundaries_preserve_rigid_motion(self):
        config = SimulationConfig(nx=21, ny=17, dx=0.03, dy=0.02, source_mode="interior")
        state, solver = _initialize_simulation(config, "P")
        y, x = np.mgrid[:config.ny, :config.nx]
        # A translation plus infinitesimal rotation has identically zero strain.
        u = np.array((0.001 - 0.002 * y * config.dy).astype(np.float64)[None, :, :, None])
        v = np.array((-0.001 + 0.002 * x * config.dx).astype(np.float64)[None, :, :, None])
        acceleration = solver.operator.acceleration(solver.operator.pack(u, v))
        np.testing.assert_allclose(acceleration, 0, atol=1e-11)
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

    def test_driven_boundary_is_not_blocked_by_a_fixed_strip(self):
        for boundary in ("fixed", "free_top"):
            config = SimulationConfig(nx=25, ny=17, boundary_width=2, boundary_type=boundary)
            state, solver = _initialize_simulation(config, "P")
            for step in range(40):
                solver.step(state, step*config.dt)
            self.assertGreater(np.max(np.abs(state.u[0, 2:-2, 3:, 0])), 1e-8)

    def test_natural_boundary_stiffness_is_symmetric_and_nonnegative(self):
        from scipy.linalg import eigvalsh
        for nu, ratio in ((0, 1), (.25, 1), (.45, 1), (.49, 1), (.4999, 1), (.25, .5), (.25, 2)):
            config = SimulationConfig(nx=9, ny=9, dx=.02, dy=.02*ratio, poisson_ratio=nu)
            operator = ElasticOperator(config)
            normalized = operator.stiffness.toarray()/np.sqrt(operator.mass[:, None]*operator.mass[None, :])
            scale = np.max(np.abs(normalized))
            self.assertLess(np.max(np.abs(normalized-normalized.T))/scale, 100*np.finfo(float).eps)
            eigenvalues = eigvalsh(normalized)
            tolerance = 100*np.finfo(float).eps*eigenvalues[-1]
            self.assertGreater(eigenvalues[0], -tolerance)
            self.assertEqual(np.count_nonzero(np.abs(eigenvalues)<tolerance), 3)
            self.assertLess(config.dt*np.sqrt(eigenvalues[-1]), 2)

    def test_time_integrators_converge_quadratically(self):
        from scipy.linalg import eigh
        config = SimulationConfig(nx=5, ny=5, dx=.2, dy=.2, damping=1, source_mode="interior",
                                  source_amplitude=0, source_x_start=2, source_x_end=3)
        operator = ElasticOperator(config)
        eigenvalues, modes = eigh(operator.stiffness.toarray(), np.diag(operator.mass))
        omega, mode = np.sqrt(eigenvalues[3]), modes[:, 3]
        for method in ("explicit", "implicit"):
            errors = []
            for count in (20, 40):
                config.time_integration = method
                config.cfl_safety_factor *= (1/(omega*count))/config.dt
                state, solver = _initialize_simulation(config, "P")
                state.u, state.v = operator.fields(mode)
                state.u_prev, state.v_prev = operator.fields(mode*np.cos(omega*config.dt))
                for step in range(count):
                    solver.step(state, step*config.dt)
                amplitude = mode @ (operator.mass*operator.pack(state.u, state.v))
                errors.append(abs(amplitude-np.cos(1)))
            with self.subTest(method=method):
                self.assertGreater(errors[0]/errors[1], 3)
                self.assertLess(errors[0]/errors[1], 5)

    def test_failed_implicit_solve_does_not_advance(self):
        config = SimulationConfig()
        state, solver = _initialize_simulation(config, "P")
        initial = state.u.copy()
        solver._solve = lambda rhs: np.full_like(rhs, np.nan)
        with self.assertRaises(RuntimeError):
            solver.step(state, 0)
        np.testing.assert_array_equal(state.u, initial)


class OutputTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_magnitude_is_euclidean_and_does_not_square_extreme_values(self):
        with tempfile.TemporaryDirectory() as directory:
            config = SimulationConfig(nx=2, ny=2, gif_output_dir=directory)
            state = WaveFieldState(2, 2)
            exporter = GifExporter(config, state, None)
            state.set_display_mode("Both")
            for scale in (1., 1e200, 1e-200):
                with self.subTest(scale=scale):
                    state.u.fill(3*scale)
                    state.v.fill(4*scale)
                    expected = 5*scale
                    np.testing.assert_allclose(state.get_display_field(), expected, rtol=1e-14, atol=0)
                    np.testing.assert_allclose(exporter._collect_frame_data()[2], expected, rtol=1e-14, atol=0)
                    np.testing.assert_allclose(state.get_max_displacement(), expected, rtol=1e-14, atol=0)

    def test_nonfinite_outputs_fail_and_active_source_does_not_stop_early(self):
        config = SimulationConfig(time_integration="explicit", source_amplitude=0)
        state, solver = _initialize_simulation(config, "P")
        view = WaveVisualizer(config, state, solver)
        view._min_steps_before_check = 0
        for frame in range(6):
            view._animate(frame)
        self.assertEqual(view.current_step, 12)
        state.v = np.full(state.v.shape, float("nan"))
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
        left, right, bottom, top = view.im.get_extent()
        pixel_width = (right-left)/config.nx
        pixel_height = (top-bottom)/config.ny
        np.testing.assert_allclose(left+pixel_width*(np.arange(config.nx)+.5), np.arange(config.nx)*config.dx, atol=1e-14)
        np.testing.assert_allclose(bottom+pixel_height*(np.arange(config.ny)+.5), np.arange(config.ny)*config.dy, atol=1e-14)
        self.assertEqual(view.ax.get_aspect(), 1.)
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
