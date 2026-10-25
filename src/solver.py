"""
Energy-form finite elements, boundary constraints and elastic wave integration.
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import factorized
from skfem import Basis, BilinearForm, ElementQuad1, ElementVector, MeshQuad, asm
from skfem.helpers import dot
from skfem.models.elasticity import linear_elasticity

from config import SimulationConfig
from source import WaveSourceGenerator
from state import WaveFieldState


class ElasticOperator:
    """B-bar Q1 plane-strain stiffness and positive row-sum lumped mass."""

    def __init__(self, config: SimulationConfig):
        config.validate()
        mesh = MeshQuad.init_tensor(np.arange(config.nx)*config.dx, np.arange(config.ny)*config.dy)
        basis = Basis(mesh, ElementVector(ElementQuad1()))
        self.dofs = basis.nodal_dofs.reshape(2, config.nx, config.ny).transpose(0, 2, 1)
        # B-bar: fully integrate deviatoric strain and project volume strain onto cell constants.
        reduced = Basis(mesh, ElementVector(ElementQuad1()), quadrature=(np.full((2, 1), .5), np.ones(1)))
        self.stiffness = (
            asm(linear_elasticity(-2*config.lame_mu/3, config.lame_mu), basis)
            + asm(linear_elasticity(config.lame_lambda+2*config.lame_mu/3, 0), reduced)
        ).tocsr()
        mass = asm(BilinearForm(lambda u, v, w: dot(u, v)), basis)
        self.mass = config.density*np.asarray(mass.sum(axis=1)).ravel()

    def pack(self, u, v):
        values = np.empty(self.mass.size)
        values[self.dofs[0]] = np.asarray(u)[0, :, :, 0]
        values[self.dofs[1]] = np.asarray(v)[0, :, :, 0]
        return values

    def fields(self, values):
        u, v = values[self.dofs]
        return u[None, :, :, None], v[None, :, :, None]

    def acceleration(self, values):
        return -(self.stiffness @ values)/self.mass


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
        config: SimulationConfig | None = None,
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
            config: Simulation configuration supplying physical source-grid spacing
        """
        self.nx = nx
        self.ny = ny
        self.boundary_width = boundary_width
        self.boundary_type = boundary_type
        self.source_x_start = source_x_start
        self.source_x_end = source_x_end
        self.source_mode = source_mode
        self.config = config

        # Fixed wall boundary mask (Dirichlet: u=0, v=0 at boundaries)
        mask_field = np.ones((1, ny, nx, 1), dtype=np.float32)
        if boundary_type != "stress_free_all":
            # For fixed and free_top types: apply masks on non-stress-free sides
            if source_mode != "boundary":
                mask_field[:, :, 0:boundary_width, :] = 0  # Left
            mask_field[:, 0:boundary_width, :, :] = 0  # Bottom
            if boundary_type == "fixed":
                mask_field[:, -boundary_width:, :, :] = 0  # Top (only if fixed)
            mask_field[:, :, -boundary_width:, :] = 0  # Right
        self.mask_field = np.array(mask_field)

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
        self.source_p_u = np.array((source_mask * cos_theta)[None, :, :, None])
        self.source_p_v = np.array((source_mask * sin_theta)[None, :, :, None])

        # S-wave source: tangential direction (u=-sin θ, v=cos θ)
        self.source_s_u = np.array((source_mask * (-sin_theta))[None, :, :, None])
        self.source_s_v = np.array((source_mask * cos_theta)[None, :, :, None])

        # Uniform left-edge source masks for uniform wave injection
        # P-wave: uniform displacement in x-direction (u=1, v=0)
        edge_source_mask = np.zeros((ny, nx), dtype=np.float32)
        edge_source_mask[:, source_x_start:source_x_end] = 1.0
        self.source_edge_p_u = np.array(edge_source_mask[None, :, :, None])
        self.source_edge_p_v = np.array(np.zeros_like(edge_source_mask)[None, :, :, None])

        # S-wave: uniform displacement in y-direction (u=0, v=1)
        self.source_edge_s_u = np.array(np.zeros_like(edge_source_mask)[None, :, :, None])
        self.source_edge_s_v = np.array(edge_source_mask[None, :, :, None])

    def apply_fixed_boundaries(self, u: np.array, v: np.array):
        """
        Apply fixed (zero displacement) boundary conditions.

        Args:
            u: X-displacement field
            v: Y-displacement field

        Returns:
            Tuple of (u, v) with boundaries enforced
        """
        return u * self.mask_field, v * self.mask_field

    def apply_driven_left_boundary(self, u: np.array, v: np.array, amplitude: float, wave_type: str):
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
        zero = np.zeros_like(u[:, :, :1, :])
        drive = zero + amplitude
        u_edge = zero if wave_type == "S" else drive
        v_edge = zero if wave_type == "P" else drive
        return np.concatenate((u_edge, u[:, :, 1:, :]), axis=2), np.concatenate((v_edge, v[:, :, 1:, :]), axis=2)

    def apply_source(self, u: np.array, v: np.array, amplitude: float, wave_type: str, input_mode: str = "radial"):
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
    """Central differences or a directly solved average-acceleration recurrence."""

    def __init__(self, config: SimulationConfig, operator: ElasticOperator,
                 bc: BoundaryConditions, source: WaveSourceGenerator):
        config.validate()
        self.config, self.operator, self.bc, self.source = config, operator, bc, source
        self.dt, self.damping = config.dt, config.damping
        free_mask = np.array(bc.mask_field, copy=True)
        if config.source_mode == "boundary":
            free_mask[:, :, 0, :] = 0
        self.free = np.flatnonzero(operator.pack(free_mask, free_mask))
        self.fixed = np.flatnonzero(1-operator.pack(free_mask, free_mask))
        self._coefficient = self.dt**2*(1+self.damping)/8
        if config.time_integration == "implicit":
            self._matrix = diags(operator.mass) + self._coefficient*operator.stiffness
            self._solve = factorized(self._matrix[self.free][:, self.free].tocsc())

    def _apply_constraints(self, u, v, t):
        u, v = self.bc.apply_fixed_boundaries(u, v)
        amplitude = self.source.get_amplitude(t+self.dt)
        if self.config.source_mode == "boundary":
            return self.bc.apply_driven_left_boundary(u, v, amplitude, "P")
        u, v = self.bc.apply_source(u, v, amplitude, "P", self.config.input_mode)
        return self.bc.apply_fixed_boundaries(u, v)

    def step(self, state: WaveFieldState, t: float) -> None:
        """Advance to t+dt; retain the state if input or computed fields are nonfinite."""
        current = self.operator.pack(state.u, state.v)
        previous = self.operator.pack(state.u_prev, state.v_prev)
        if not np.isfinite(current).all() or not np.isfinite(previous).all():
            raise RuntimeError("Non-finite displacement before integration")
        zero = np.zeros_like(state.u)
        source = self.operator.pack(*self._apply_constraints(zero, zero, t))
        predictor = (1+self.damping)*current-self.damping*previous
        if self.config.time_integration == "implicit":
            rhs = self.operator.mass*(predictor+source)
            rhs -= self._coefficient*(self.operator.stiffness @ (previous+2*current))
            result = np.zeros_like(current)
            result[self.fixed] = source[self.fixed]
            result[self.free] = self._solve((rhs-self._matrix @ result)[self.free])
        else:
            result = predictor + self.dt**2*(1+self.damping)/2*self.operator.acceleration(current) + source
            result[self.fixed] = source[self.fixed]
        if not np.isfinite(result).all():
            raise RuntimeError("Non-finite displacement after integration")
        state.advance(*self.operator.fields(result))

    def check_stability(self, state: WaveFieldState) -> bool:
        max_disp = state.get_max_displacement()
        return not np.isfinite(max_disp) or max_disp > max(1e-3, 100*abs(self.config.source_amplitude))
