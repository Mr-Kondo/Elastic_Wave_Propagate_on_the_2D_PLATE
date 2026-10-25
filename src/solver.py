"""
Wave solver and finite difference operators for elastic wave simulation.
"""

import mlx.core as mx
import numpy as np
from config import SimulationConfig
from state import WaveFieldState
from source import WaveSourceGenerator


class FiniteDifferenceOperator:
    """4th-order finite difference operator for spatial derivatives."""

    def __init__(self, dx: float, dy: float, order: int = 4):
        """
        Initialize finite difference kernels.

        Args:
            dx: Grid spacing in x direction (meters)
            dy: Grid spacing in y direction (meters)
            order: Finite difference order (only 4 supported)
        """
        if order != 4:
            raise ValueError("Only 4th order finite differences are currently supported")

        self.dx = dx
        self.dy = dy

        # 4th order coefficients for second derivative
        coeffs = np.array([-1, 16, -30, 16, -1], dtype=np.float32) / 12.0

        # Convolution kernels for second derivatives
        self.k_xx = mx.array(coeffs.reshape(1, 1, 5, 1) / (dx**2))
        self.k_yy = mx.array(coeffs.reshape(1, 5, 1, 1) / (dy**2))

        # 4th order coefficients for first derivative
        d1_coeffs = np.array([1, -8, 0, 8, -1], dtype=np.float32) / 12.0

        # Kernel for mixed derivative
        k_xy_mat = np.outer(d1_coeffs, d1_coeffs) / (dx * dy)
        self.k_xy = mx.array(k_xy_mat.reshape(1, 5, 5, 1))

        # Store grid dimensions for cropping
        self.ny = None
        self.nx = None

    def compute_derivatives(self, u: mx.array, v: mx.array, bc=None):
        """
        Compute all spatial derivatives needed for wave equation.

        Args:
            u: X-displacement field
            v: Y-displacement field

        Returns:
            Tuple of (uxx, uyy, uxy, vxx, vyy, vxy)
        """
        # Store grid dimensions from input
        self.ny, self.nx = u.shape[1:3]

        # Pad for 4th-order stencils (2 points on each side)
        if bc is None:
            pad_u = mx.pad(u, ((0, 0), (2, 2), (2, 2), (0, 0)))
            pad_v = mx.pad(v, ((0, 0), (2, 2), (2, 2), (0, 0)))
        else:
            pad_u, pad_v = bc.pad_fields(u, v)

        # Compute derivatives via convolution
        uxx = self._center_crop(mx.conv2d(pad_u, self.k_xx))
        uyy = self._center_crop(mx.conv2d(pad_u, self.k_yy))
        uxy = self._center_crop(mx.conv2d(pad_u, self.k_xy))

        vxx = self._center_crop(mx.conv2d(pad_v, self.k_xx))
        vyy = self._center_crop(mx.conv2d(pad_v, self.k_yy))
        vxy = self._center_crop(mx.conv2d(pad_v, self.k_xy))

        return uxx, uyy, uxy, vxx, vyy, vxy

    def _center_crop(self, x: mx.array) -> mx.array:
        """Crop convolution output to original grid size."""
        h = x.shape[1]
        w = x.shape[2]
        start_y = (h - self.ny) // 2
        start_x = (w - self.nx) // 2
        return x[:, start_y : start_y + self.ny, start_x : start_x + self.nx, :]


class BoundaryConditions:
    """Manages boundary conditions and source injection."""

    def __init__(
        self,
        nx: int,
        ny: int,
        boundary_width: int = 5,
        source_x_start: int = 6,
        source_x_end: int = 8,
        boundary_type: str = "fixed",
        source_mode: str = "interior",
        config: "SimulationConfig" = None,
        fd_operator: "FiniteDifferenceOperator" = None,
    ):
        """
        Initialize boundary condition masks.

        Args:
            nx: Grid points in x direction
            ny: Grid points in y direction
            boundary_width: Width of fixed boundary region
            source_x_start: Left edge of source region
            source_x_end: Right edge of source region
            boundary_type: 'fixed' for all walls fixed, 'free_top' for free surface on top, 'stress_free_all' for all stress-free
            source_mode: 'boundary' for left edge injection, 'interior' for internal source
            config: Simulation configuration (needed for stress-free BC)
            fd_operator: Finite difference operator (needed for derivative calculations)
        """
        self.nx = nx
        self.ny = ny
        self.boundary_width = boundary_width
        self.boundary_type = boundary_type
        self.source_x_start = source_x_start
        self.source_x_end = source_x_end
        self.source_mode = source_mode
        self.config = config
        self.fd_operator = fd_operator
        if boundary_type != "fixed" and config is None:
            raise ValueError("Free boundaries require a SimulationConfig")

        # Fixed wall boundary mask (Dirichlet: u=0, v=0 at boundaries)
        mask_field = np.ones((1, ny, nx, 1), dtype=np.float32)
        if boundary_type != "stress_free_all":
            # For fixed and free_top types: apply masks on non-stress-free sides
            mask_field[:, :, 0:boundary_width, :] = 0  # Left
            mask_field[:, 0:boundary_width, :, :] = 0  # Bottom
            if boundary_type == "fixed":
                mask_field[:, -boundary_width:, :, :] = 0  # Top (only if fixed)
            mask_field[:, :, -boundary_width:, :] = 0  # Right
        self.mask_field = mx.array(mask_field)

        # Compute source center position
        source_x_center = (source_x_start + source_x_end - 1) / 2.0
        source_y_center = (ny - 1) / 2.0

        # Create coordinate grids centered at source
        x_coords = (np.arange(nx) - source_x_center) * (config.dx if config else 1)
        y_coords = (np.arange(ny) - source_y_center) * (config.dy if config else 1)
        X, Y = np.meshgrid(x_coords, y_coords)

        # Compute radial distance and angle from source
        R = np.sqrt(X**2 + Y**2)
        R = np.maximum(R, 1e-10)  # Avoid division by zero

        # Unit radial direction (for P wave): (cos θ, sin θ)
        cos_theta = X / R
        sin_theta = Y / R

        # Unit tangential direction (for S wave): (-sin θ, cos θ)
        # This is perpendicular to radial direction

        # Restrict the polarization field to the source columns
        source_mask = np.zeros((ny, nx), dtype=np.float32)
        source_mask[:, source_x_start:source_x_end] = 1.0

        # P-wave source: radial direction (u=cos θ, v=sin θ)
        self.source_p_u = mx.array((source_mask * cos_theta)[None, :, :, None])
        self.source_p_v = mx.array((source_mask * sin_theta)[None, :, :, None])

        # S-wave source: tangential direction (u=-sin θ, v=cos θ)
        self.source_s_u = mx.array((source_mask * (-sin_theta))[None, :, :, None])
        self.source_s_v = mx.array((source_mask * cos_theta)[None, :, :, None])

        # Uniform left-edge source masks for uniform wave injection
        # P-wave: uniform displacement in x-direction (u=1, v=0)
        edge_source_mask = np.zeros((ny, nx), dtype=np.float32)
        edge_source_mask[:, source_x_start:source_x_end] = 1.0
        self.source_edge_p_u = mx.array(edge_source_mask[None, :, :, None])
        self.source_edge_p_v = mx.array(np.zeros_like(edge_source_mask)[None, :, :, None])

        # S-wave: uniform displacement in y-direction (u=0, v=1)
        self.source_edge_s_u = mx.array(np.zeros_like(edge_source_mask)[None, :, :, None])
        self.source_edge_s_v = mx.array(edge_source_mask[None, :, :, None])

    def apply_fixed_boundaries(self, u: mx.array, v: mx.array):
        """
        Apply fixed (zero displacement) boundary conditions.

        Args:
            u: X-displacement field
            v: Y-displacement field

        Returns:
            Tuple of (u, v) with boundaries enforced
        """
        return u * self.mask_field, v * self.mask_field

    @staticmethod
    def _gradient(field, spacing, axis):
        field = mx.swapaxes(field, axis, 0)
        result = mx.concatenate((
            (field[1:2] - field[:1]) / spacing,
            (field[2:] - field[:-2]) / (2 * spacing),
            (field[-1:] - field[-2:-1]) / spacing,
        ), axis=0)
        return mx.swapaxes(result, 0, axis)

    @staticmethod
    def _extend(field, gradient, spacing, axis, lower, upper):
        field = mx.swapaxes(field, axis, 0)
        gradient = mx.swapaxes(gradient, axis, 0)
        low = mx.concatenate([field[k:k+1] - 2*k*spacing*gradient[:1] for k in (2, 1)])
        high = mx.concatenate([field[-k-1:-k] + 2*k*spacing*gradient[-1:] for k in (1, 2)])
        return mx.swapaxes(mx.concatenate((
            low if lower else mx.concatenate([2*field[:1] - field[k:k+1] for k in (2, 1)]), field,
            high if upper else mx.concatenate([2*field[-1:] - field[-k-1:-k] for k in (1, 2)]),
        )), 0, axis)

    def pad_fields(self, u, v):
        """Exterior ghost nodes enforce zero normal and shear traction.

        shortcut: second-order boundary closure and sequential corner extension;
        use an energy-stable closure for quantitative surface-wave studies.
        """
        u, v = u[0, :, :, 0], v[0, :, :, 0]
        if self.boundary_type == "fixed":
            for axis in (0, 1):
                u = self._extend(u, mx.zeros_like(u), 1, axis, False, False)
                v = self._extend(v, mx.zeros_like(v), 1, axis, False, False)
            return u[None, :, :, None], v[None, :, :, None]
        dx, dy = self.config.dx, self.config.dy
        alpha = self.config.lame_lambda / (self.config.lame_lambda + 2*self.config.lame_mu)
        all_free = self.boundary_type == "stress_free_all"
        u_x, v_x = self._gradient(u, dx, 1), self._gradient(v, dx, 1)
        u = self._extend(u, -v_x, dy, 0, all_free, True)
        v = self._extend(v, -alpha*u_x, dy, 0, all_free, True)
        u_y, v_y = self._gradient(u, dy, 0), self._gradient(v, dy, 0)
        u = self._extend(u, -alpha*v_y, dx, 1, all_free and self.source_mode != "boundary", all_free)
        v = self._extend(v, -u_y, dx, 1, all_free and self.source_mode != "boundary", all_free)
        return u[None, :, :, None], v[None, :, :, None]

    def apply_driven_left_boundary(self, u: mx.array, v: mx.array, amplitude: float, wave_type: str):
        """
        Apply driven (prescribed displacement) boundary condition at left edge.

        Prescribes the entire left boundary, including its endpoints.
        Polarization alone does not validate a propagated wave mode.

        Args:
            u: X-displacement field
            v: Y-displacement field
            amplitude: Source amplitude at current time (from WaveSourceGenerator)
            wave_type: 'P' for P-wave, 'S' for S-wave, 'Both' for both

        Returns:
            Tuple of (u, v) with driven boundary applied to full height
        """
        zero = mx.zeros_like(u[:, :, :1, :])
        drive = zero + amplitude
        u_edge = zero if wave_type == "S" else drive
        v_edge = zero if wave_type == "P" else drive
        return mx.concatenate((u_edge, u[:, :, 1:, :]), axis=2), mx.concatenate((v_edge, v[:, :, 1:, :]), axis=2)

    def apply_source(self, u: mx.array, v: mx.array, amplitude: float, wave_type: str, input_mode: str = "radial"):
        """
        Inject source into displacement field with physically correct polarization.

        For radial (point) sources:
        - P wave: radial displacement (direction from source center)
        - S wave: tangential displacement (perpendicular to radial)
        - Both: combination of P and S waves

        For uniform edge sources:
        - P wave: uniform x-direction displacement (u=1, v=0)
        - S wave: uniform y-direction displacement (u=0, v=1)
        - Both: combination of both

        Args:
            u: X-displacement field
            v: Y-displacement field
            amplitude: Source amplitude at current time
            wave_type: 'P' for P-wave, 'S' for S-wave, 'Both' for both
            input_mode: 'radial' for point source, 'uniform_edge' for left-edge uniform input

        Returns:
            Tuple of (u, v) with source applied
        """
        if input_mode == "uniform_edge":
            # Uniform left-edge input
            if wave_type == "P":
                # P wave: uniform x-direction
                u = u + (self.source_edge_p_u * amplitude)
                v = v + (self.source_edge_p_v * amplitude)
            elif wave_type == "S":
                # S wave: uniform y-direction
                u = u + (self.source_edge_s_u * amplitude)
                v = v + (self.source_edge_s_v * amplitude)
            else:  # Both
                # Superposition: x-direction + y-direction
                u = u + (self.source_edge_p_u * amplitude) + (self.source_edge_s_u * amplitude)
                v = v + (self.source_edge_p_v * amplitude) + (self.source_edge_s_v * amplitude)
        else:
            # Radial (point source) input (original behavior)
            if wave_type == "P":
                # P wave: radial direction (cos θ, sin θ)
                u = u + (self.source_p_u * amplitude)
                v = v + (self.source_p_v * amplitude)
            elif wave_type == "S":
                # S wave: tangential direction (-sin θ, cos θ), perpendicular to radial
                u = u + (self.source_s_u * amplitude)
                v = v + (self.source_s_v * amplitude)
            else:  # Both
                # Superposition of P and S waves
                u = u + (self.source_p_u * amplitude) + (self.source_s_u * amplitude)
                v = v + (self.source_p_v * amplitude) + (self.source_s_v * amplitude)
        return u, v


class WaveSolver:
    """Solves 2D elastic wave equation using finite differences."""

    def __init__(
        self,
        config: SimulationConfig,
        fd_operator: FiniteDifferenceOperator,
        bc: BoundaryConditions,
        source: WaveSourceGenerator,
    ):
        """
        Initialize wave solver.

        Args:
            config: Simulation configuration
            fd_operator: Finite difference operator
            bc: Boundary conditions
            source: Wave source generator
        """
        config.validate()
        self.config = config
        self.fd = fd_operator
        self.bc = bc
        self.source = source

    def step(self, state: WaveFieldState, t: float) -> None:
        """Advance from time t to t + dt; retain state on convergence failure."""
        if self.config.time_integration == "implicit":
            u_next, v_next = self._step_implicit(state, t)
        else:
            acc_u, acc_v = self._accelerations(state.u, state.v)
            u_next, v_next = self._time_integrate_explicit(
                state.u, state.v, state.u_prev, state.v_prev, acc_u, acc_v)
            u_next, v_next = self._apply_constraints(u_next, v_next, t)
        state.advance(u_next, v_next)

    def _accelerations(self, u, v):
        uxx, uyy, uxy, vxx, vyy, vxy = self.fd.compute_derivatives(u, v, self.bc)
        return self._compute_acceleration_u(uxx, uyy, vxy), self._compute_acceleration_v(vxx, vyy, uxy)

    def _apply_constraints(self, u, v, t):
        u, v = self.bc.apply_fixed_boundaries(u, v)
        amplitude = self.source.get_amplitude(t + self.config.dt)
        if self.config.source_mode == "boundary":
            return self.bc.apply_driven_left_boundary(u, v, amplitude, "P")
        u, v = self.bc.apply_source(u, v, amplitude, "P", self.config.input_mode)
        return self.bc.apply_fixed_boundaries(u, v)

    def _step_implicit(self, state: WaveFieldState, t: float):
        """Newmark average-acceleration recurrence, solved by fixed-point iteration."""
        acc_u, acc_v = self._accelerations(state.u, state.v)
        prev_u, prev_v = self._accelerations(state.u_prev, state.v_prev)
        u_next, v_next = self._time_integrate_explicit(
            state.u, state.v, state.u_prev, state.v_prev, acc_u, acc_v)
        u_next, v_next = self._apply_constraints(u_next, v_next, t)
        for _ in range(self.config.implicit_iterations):
            next_u, next_v = self._accelerations(u_next, v_next)
            candidate_u, candidate_v = self._time_integrate_explicit(
                state.u, state.v, state.u_prev, state.v_prev,
                (prev_u + 2*acc_u + next_u)/4, (prev_v + 2*acc_v + next_v)/4)
            candidate_u, candidate_v = self._apply_constraints(candidate_u, candidate_v, t)
            error = mx.maximum(mx.max(mx.abs(candidate_u-u_next)), mx.max(mx.abs(candidate_v-v_next)))
            scale = mx.maximum(mx.maximum(mx.max(mx.abs(candidate_u)), mx.max(mx.abs(candidate_v))), 1e-12)
            relative_change = (error / scale).item()
            u_next, v_next = candidate_u, candidate_v
            # MLX uses float32; iteration noise cannot resolve sub-ULP tolerances.
            if np.isfinite(relative_change) and relative_change <= max(self.config.implicit_tolerance, 4 * np.finfo(np.float32).eps):
                return u_next, v_next
        raise RuntimeError("Implicit solver did not converge; reduce cfl_safety_factor or increase implicit_iterations")

    def _compute_acceleration_u(self, uxx: mx.array, uyy: mx.array, vxy: mx.array) -> mx.array:
        """Compute X-acceleration from Navier-Cauchy equation."""
        return (
            (self.config.lame_lambda + 2 * self.config.lame_mu) * uxx
            + self.config.lame_mu * uyy
            + (self.config.lame_lambda + self.config.lame_mu) * vxy
        ) / self.config.density

    def _compute_acceleration_v(self, vxx: mx.array, vyy: mx.array, uxy: mx.array) -> mx.array:
        """Compute Y-acceleration from Navier-Cauchy equation."""
        return (
            self.config.lame_mu * vxx
            + (self.config.lame_lambda + 2 * self.config.lame_mu) * vyy
            + (self.config.lame_lambda + self.config.lame_mu) * uxy
        ) / self.config.density

    def _time_integrate_explicit(
        self, u: mx.array, v: mx.array, u_prev: mx.array, v_prev: mx.array, acc_u: mx.array, acc_v: mx.array
    ):
        """
        Integrate in time using central differences (leap-frog/explicit).

        Args:
            u, v: Current displacement fields
            u_prev, v_prev: Previous time step fields
            acc_u, acc_v: Accelerations

        Returns:
            Tuple of (u_next, v_next)
        """
        dt = self.config.dt

        retention = self.config.damping
        u_next = (1 + retention) * u - retention * u_prev + (dt**2) * (1 + retention)/2 * acc_u
        v_next = (1 + retention) * v - retention * v_prev + (dt**2) * (1 + retention)/2 * acc_v

        return u_next, v_next

    def check_stability(self, state: WaveFieldState) -> bool:
        """
        Check if simulation has become unstable.

        Args:
            state: Current wave field state

        Returns:
            True if unstable (exploded), False if stable
        """
        max_disp = state.get_max_displacement()
        return not np.isfinite(max_disp) or max_disp > max(1e-3, 100 * abs(self.config.source_amplitude))
