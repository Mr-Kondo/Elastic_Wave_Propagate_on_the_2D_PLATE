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

    def compute_derivatives(self, u: mx.array, v: mx.array):
        """
        Compute all spatial derivatives needed for wave equation.

        Args:
            u: X-displacement field
            v: Y-displacement field

        Returns:
            Tuple of (uxx, uyy, uxy, vxx, vyy, vxy)
        """
        # Store grid dimensions from input
        if self.ny is None or self.nx is None:
            self.ny = u.shape[1]
            self.nx = u.shape[2]

        # Pad for 4th-order stencils (2 points on each side)
        pad_u = mx.pad(u, ((0, 0), (2, 2), (2, 2), (0, 0)))
        pad_v = mx.pad(v, ((0, 0), (2, 2), (2, 2), (0, 0)))

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
        source_x_center = (source_x_start + source_x_end) / 2.0
        source_y_center = ny / 2.0

        # Create coordinate grids centered at source
        x_coords = np.arange(nx) - source_x_center
        y_coords = np.arange(ny) - source_y_center
        X, Y = np.meshgrid(x_coords, y_coords)

        # Compute radial distance and angle from source
        R = np.sqrt(X**2 + Y**2)
        R = np.maximum(R, 1e-10)  # Avoid division by zero

        # Unit radial direction (for P wave): (cos θ, sin θ)
        cos_theta = X / R
        sin_theta = Y / R

        # Unit tangential direction (for S wave): (-sin θ, cos θ)
        # This is perpendicular to radial direction

        # Create source masks (Gaussian profile around source)
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

    def apply_free_surface_top(self, u: mx.array, v: mx.array) -> tuple[mx.array, mx.array]:
        """
        Apply stress-free boundary condition at top surface (Rayleigh wave support).
        Uses mirror boundary: stress must be zero at surface.
        This approximates σ_zz = 0 and σ_zx = 0 at the top boundary.

        Args:
            u: X-displacement field
            v: Y-displacement field

        Returns:
            Tuple of (u, v) with free surface condition applied
        """
        if self.boundary_type != "free_top":
            return u, v

        # Apply symmetric extension for stress-free boundary
        # σ_yy = 0 requires ∂v/∂y = 0 (symmetric for v)
        # σ_xy = 0 requires ∂u/∂y = 0 (symmetric for u)
        # Mirror the displacement symmetrically across the boundary
        bw = self.boundary_width
        u_np = np.array(u[0, :, :, 0])
        v_np = np.array(v[0, :, :, 0])

        for i in range(bw):
            # Top boundary: symmetric mirror for both components
            # This creates ∂u/∂y = 0 and ∂v/∂y = 0 at the boundary
            # Mirror index: ghost point at -(i+1) mirrors interior point at -(2*bw-i)
            mirror_idx = -(2 * bw - i)
            if mirror_idx < 0 and mirror_idx >= -v_np.shape[0]:
                v_np[-(i + 1), :] = v_np[mirror_idx, :]
                u_np[-(i + 1), :] = u_np[mirror_idx, :]
            else:
                # Fallback for edge cases: use nearest valid point
                v_np[-(i + 1), :] = v_np[-(bw + 1), :]
                u_np[-(i + 1), :] = u_np[-(bw + 1), :]

        u = mx.array(u_np[None, :, :, None])
        v = mx.array(v_np[None, :, :, None])

        return u, v

    def apply_stress_free_all(self, u: mx.array, v: mx.array) -> tuple[mx.array, mx.array]:
        """
        Apply stress-free (Neumann) boundary conditions on all 4 sides.
        Enforces σ = 0 (zero stress) at boundaries using explicit ghost point calculation.

        Stress tensor in 2D:
            σ_xx = λ(u_x + v_y) + 2μ u_x
            σ_yy = λ(u_x + v_y) + 2μ v_y
            σ_xy = μ(u_y + v_x)

        For stress-free boundaries (σ = 0):
            σ_yy = 0: v_y = -λ(u_x+v_y)/(λ+2μ) = -λu_x/(λ+2μ) when v_y is eliminated
            σ_xy = 0: u_y = -v_x

        Ghost point calculation uses Taylor expansion:
            φ_ghost = φ_interior - 2*Δ*∂φ/∂n

        Args:
            u: X-displacement field [1, ny, nx, 1]
            v: Y-displacement field [1, ny, nx, 1]

        Returns:
            Tuple of (u, v) with stress-free conditions enforced
        """
        if self.boundary_type != "stress_free_all":
            return u, v

        bw = self.boundary_width
        u_np = np.array(u[0, :, :, 0])
        v_np = np.array(v[0, :, :, 0])

        ny, nx = u_np.shape
        dx = self.config.dx
        dy = self.config.dy

        # Lamé parameters
        lam = self.config.lame_lambda
        mu = self.config.lame_mu

        # === TOP BOUNDARY (rows ny-bw to ny-1): σ_yy = 0, σ_xy = 0 ===
        interior_row = ny - bw - 1
        for i in range(1, bw + 1):
            ghost_row = ny - i

            if interior_row >= 0 and ghost_row < ny and ghost_row > interior_row:
                # Compute u_x at interior row using central differences
                u_x = np.zeros(nx)
                u_x[1:-1] = (u_np[interior_row, 2:] - u_np[interior_row, :-2]) / (2 * dx)
                u_x[0] = (u_np[interior_row, 1] - u_np[interior_row, 0]) / dx
                u_x[-1] = (u_np[interior_row, -1] - u_np[interior_row, -2]) / dx

                # Compute v_x at interior row
                v_x = np.zeros(nx)
                v_x[1:-1] = (v_np[interior_row, 2:] - v_np[interior_row, :-2]) / (2 * dx)
                v_x[0] = (v_np[interior_row, 1] - v_np[interior_row, 0]) / dx
                v_x[-1] = (v_np[interior_row, -1] - v_np[interior_row, -2]) / dx

                # From σ_yy = 0: v_y = -λ u_x / (λ + 2μ)
                # From σ_xy = 0: u_y = -v_x
                alpha = lam / (lam + 2 * mu)

                v_np[ghost_row, :] = v_np[interior_row, :] - 2 * dy * (-alpha * u_x)
                u_np[ghost_row, :] = u_np[interior_row, :] - 2 * dy * (-v_x)

        # === BOTTOM BOUNDARY (rows 0 to bw-1): σ_yy = 0, σ_xy = 0 ===
        interior_row = bw
        for i in range(1, bw + 1):
            ghost_row = bw - i

            if interior_row < ny and ghost_row >= 0 and ghost_row < interior_row:
                # Compute u_x at interior row
                u_x = np.zeros(nx)
                u_x[1:-1] = (u_np[interior_row, 2:] - u_np[interior_row, :-2]) / (2 * dx)
                u_x[0] = (u_np[interior_row, 1] - u_np[interior_row, 0]) / dx
                u_x[-1] = (u_np[interior_row, -1] - u_np[interior_row, -2]) / dx

                # Compute v_x at interior row
                v_x = np.zeros(nx)
                v_x[1:-1] = (v_np[interior_row, 2:] - v_np[interior_row, :-2]) / (2 * dx)
                v_x[0] = (v_np[interior_row, 1] - v_np[interior_row, 0]) / dx
                v_x[-1] = (v_np[interior_row, -1] - v_np[interior_row, -2]) / dx

                # Bottom boundary: normal points downward, so flip sign
                alpha = lam / (lam + 2 * mu)

                v_np[ghost_row, :] = v_np[interior_row, :] + 2 * dy * (-alpha * u_x)
                u_np[ghost_row, :] = u_np[interior_row, :] + 2 * dy * (-v_x)

        # === RIGHT BOUNDARY (cols nx-bw to nx-1): σ_xx = 0, σ_xy = 0 ===
        interior_col = nx - bw - 1
        for i in range(1, bw + 1):
            ghost_col = nx - i

            if interior_col >= 0 and ghost_col < nx and ghost_col > interior_col:
                # Compute v_y at interior column
                v_y = np.zeros(ny)
                v_y[1:-1] = (v_np[2:, interior_col] - v_np[:-2, interior_col]) / (2 * dy)
                v_y[0] = (v_np[1, interior_col] - v_np[0, interior_col]) / dy
                v_y[-1] = (v_np[-1, interior_col] - v_np[-2, interior_col]) / dy

                # Compute u_y at interior column
                u_y = np.zeros(ny)
                u_y[1:-1] = (u_np[2:, interior_col] - u_np[:-2, interior_col]) / (2 * dy)
                u_y[0] = (u_np[1, interior_col] - u_np[0, interior_col]) / dy
                u_y[-1] = (u_np[-1, interior_col] - u_np[-2, interior_col]) / dy

                # From σ_xx = 0: u_x = -λ v_y / (λ + 2μ)
                # From σ_xy = 0: v_x = -u_y
                alpha = lam / (lam + 2 * mu)

                u_np[:, ghost_col] = u_np[:, interior_col] - 2 * dx * (-alpha * v_y)
                v_np[:, ghost_col] = v_np[:, interior_col] - 2 * dx * (-u_y)

        # === LEFT BOUNDARY (cols 0 to bw-1): σ_xx = 0, σ_xy = 0 ===
        # Skip if using boundary source mode
        if self.source_mode != "boundary":
            interior_col = bw
            for i in range(1, bw + 1):
                ghost_col = bw - i

                if interior_col < nx and ghost_col >= 0 and ghost_col < interior_col:
                    # Compute v_y at interior column
                    v_y = np.zeros(ny)
                    v_y[1:-1] = (v_np[2:, interior_col] - v_np[:-2, interior_col]) / (2 * dy)
                    v_y[0] = (v_np[1, interior_col] - v_np[0, interior_col]) / dy
                    v_y[-1] = (v_np[-1, interior_col] - v_np[-2, interior_col]) / dy

                    # Compute u_y at interior column
                    u_y = np.zeros(ny)
                    u_y[1:-1] = (u_np[2:, interior_col] - u_np[:-2, interior_col]) / (2 * dy)
                    u_y[0] = (u_np[1, interior_col] - u_np[0, interior_col]) / dy
                    u_y[-1] = (u_np[-1, interior_col] - u_np[-2, interior_col]) / dy

                    # Left boundary: normal points leftward, so flip sign
                    alpha = lam / (lam + 2 * mu)

                    u_np[:, ghost_col] = u_np[:, interior_col] + 2 * dx * (-alpha * v_y)
                    v_np[:, ghost_col] = v_np[:, interior_col] + 2 * dx * (-u_y)

        u = mx.array(u_np[None, :, :, None])
        v = mx.array(v_np[None, :, :, None])

        return u, v

    def apply_stress_free_explicit(self, u: mx.array, v: mx.array) -> tuple[mx.array, mx.array]:
        """
        Apply explicit stress-free boundary conditions (alias for apply_stress_free_all).
        This method delegates to apply_stress_free_all which enforces σ = 0 at all boundaries.

        Args:
            u: X-displacement field
            v: Y-displacement field

        Returns:
            Tuple of (u, v) with stress-free conditions enforced
        """
        return self.apply_stress_free_all(u, v)

    def apply_driven_left_boundary(self, u: mx.array, v: mx.array, amplitude: float, wave_type: str):
        """
        Apply driven (prescribed displacement) boundary condition at left edge.

        Drives the entire left boundary (all y heights) to generate complete
        P-wave input and enable Head Wave generation at stress-free boundaries.

        Args:
            u: X-displacement field
            v: Y-displacement field
            amplitude: Source amplitude at current time (from WaveSourceGenerator)
            wave_type: 'P' for P-wave, 'S' for S-wave, 'Both' for both

        Returns:
            Tuple of (u, v) with driven boundary applied to full height
        """
        u_np = np.array(u[0, :, :, 0])
        v_np = np.array(v[0, :, :, 0])

        # Drive entire left boundary (all heights) to generate Head Waves
        # Full height input at left boundary enables complete stress-free
        # boundary interactions at top/bottom, producing von Schmidt Head Waves

        if wave_type == "P":
            # P wave: x-direction displacement (compressional)
            u_np[:, 0] = amplitude  # All heights
            v_np[:, 0] = 0.0
        elif wave_type == "S":
            # S wave: y-direction displacement (shear)
            u_np[:, 0] = 0.0
            v_np[:, 0] = amplitude  # All heights
        else:  # Both
            # Combined P and S waves
            u_np[:, 0] = amplitude  # P-wave component
            v_np[:, 0] = amplitude  # S-wave component

        u = mx.array(u_np[None, :, :, None])
        v = mx.array(v_np[None, :, :, None])

        return u, v

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
        self.config = config
        self.fd = fd_operator
        self.bc = bc
        self.source = source

    def step(self, state: WaveFieldState, t: float) -> None:
        """
        Advance simulation by one time step.

        Args:
            state: Current wave field state
            t: Current simulation time (seconds)
        """
        # Choose integration method
        if self.config.time_integration == "implicit":
            self._step_implicit(state, t)
        else:
            self._step_explicit(state, t)

    def _step_explicit(self, state: WaveFieldState, t: float) -> None:
        """Explicit time integration (leap-frog)."""
        # Compute spatial derivatives
        uxx, uyy, uxy, vxx, vyy, vxy = self.fd.compute_derivatives(state.u, state.v)

        # Compute accelerations from Navier-Cauchy equations
        acc_u = self._compute_acceleration_u(uxx, uyy, vxy)
        acc_v = self._compute_acceleration_v(vxx, vyy, uxy)

        # Time integration (leap-frog/Verlet scheme)
        u_next, v_next = self._time_integrate_explicit(state.u, state.v, state.u_prev, state.v_prev, acc_u, acc_v)

        # Apply damping
        u_next = u_next * self.config.damping
        v_next = v_next * self.config.damping

        # Apply boundary conditions
        if self.config.boundary_type == "stress_free_all":
            u_next, v_next = self.bc.apply_stress_free_explicit(u_next, v_next)
        else:
            u_next, v_next = self.bc.apply_fixed_boundaries(u_next, v_next)
            u_next, v_next = self.bc.apply_free_surface_top(u_next, v_next)

        # Apply wave source (always P-wave input)
        source_amplitude = self.source.get_amplitude(t)
        if self.config.source_mode == "boundary":
            # Driven boundary: prescribe displacement at left edge
            u_next, v_next = self.bc.apply_driven_left_boundary(u_next, v_next, source_amplitude, "P")
        else:
            # Interior source: inject at internal grid points
            u_next, v_next = self.bc.apply_source(u_next, v_next, source_amplitude, "P", self.config.input_mode)

        # Update state
        state.advance(u_next, v_next)

    def _step_implicit(self, state: WaveFieldState, t: float) -> None:
        """
        Semi-implicit time integration with iterative solver.

        Uses fixed-point iteration with convergence check to solve:
        u^{n+1} = 2u^n - u^{n-1} + dt^2 * acc(u^{n+1}, v^{n+1})
        """
        dt = self.config.dt

        # Initial guess: use explicit step as predictor
        uxx, uyy, uxy, vxx, vyy, vxy = self.fd.compute_derivatives(state.u, state.v)
        acc_u = self._compute_acceleration_u(uxx, uyy, vxy)
        acc_v = self._compute_acceleration_v(vxx, vyy, uxy)

        u_next = 2 * state.u - state.u_prev + (dt**2) * acc_u
        v_next = 2 * state.v - state.v_prev + (dt**2) * acc_v

        # Fixed-point iterations (corrector) with convergence check
        for iteration in range(self.config.implicit_iterations):
            u_prev_iter = u_next
            v_prev_iter = v_next

            # Compute acceleration at predicted next state
            uxx, uyy, uxy, vxx, vyy, vxy = self.fd.compute_derivatives(u_next, v_next)
            acc_u_next = self._compute_acceleration_u(uxx, uyy, vxy)
            acc_v_next = self._compute_acceleration_v(vxx, vyy, uxy)

            # Average accelerations (Crank-Nicolson style)
            acc_u_avg = 0.5 * (acc_u + acc_u_next)
            acc_v_avg = 0.5 * (acc_v + acc_v_next)

            # Update prediction
            u_next = 2 * state.u - state.u_prev + (dt**2) * acc_u_avg
            v_next = 2 * state.v - state.v_prev + (dt**2) * acc_v_avg

            # Check convergence
            u_diff = mx.abs(u_next - u_prev_iter)
            v_diff = mx.abs(v_next - v_prev_iter)
            u_scale = mx.abs(u_next) + 1e-12  # Avoid division by zero
            v_scale = mx.abs(v_next) + 1e-12

            relative_change_u = mx.max(u_diff / u_scale).item()
            relative_change_v = mx.max(v_diff / v_scale).item()
            max_relative_change = max(relative_change_u, relative_change_v)

            if max_relative_change < self.config.implicit_tolerance:
                # Converged - print diagnostic info occasionally
                if t < 0.001 or iteration > 3:  # Log first step or slow convergence
                    print(f"Implicit solver converged in {iteration + 1} iterations (tol={max_relative_change:.2e})")
                break

        # Apply damping
        u_next = u_next * self.config.damping
        v_next = v_next * self.config.damping

        # Apply boundary conditions
        if self.config.boundary_type == "stress_free_all":
            u_next, v_next = self.bc.apply_stress_free_explicit(u_next, v_next)
        else:
            u_next, v_next = self.bc.apply_fixed_boundaries(u_next, v_next)
            u_next, v_next = self.bc.apply_free_surface_top(u_next, v_next)

        # Apply wave source (always P-wave input)
        source_amplitude = self.source.get_amplitude(t)
        if self.config.source_mode == "boundary":
            # Driven boundary: prescribe displacement at left edge
            u_next, v_next = self.bc.apply_driven_left_boundary(u_next, v_next, source_amplitude, "P")
        else:
            # Interior source: inject at internal grid points
            u_next, v_next = self.bc.apply_source(u_next, v_next, source_amplitude, "P", self.config.input_mode)

        # Update state
        state.advance(u_next, v_next)

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

        u_next = 2 * u - u_prev + (dt**2) * acc_u
        v_next = 2 * v - v_prev + (dt**2) * acc_v

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
        return max_disp > 1e-3
