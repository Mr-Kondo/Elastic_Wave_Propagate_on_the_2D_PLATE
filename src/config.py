"""
Configuration management for elastic wave simulation.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class SimulationConfig:
    """Configuration for elastic wave propagation simulation."""

    # Grid settings
    nx: int = 101
    ny: int = 20
    dx: float = 0.009901  # Grid spacing in x (meters) - 1m domain / 101 points
    dy: float = 0.0095  # Grid spacing in y (meters) - 0.19m domain / 20 points

    # Material properties
    youngs_modulus: float = 1000.0  # Young's modulus (Pa)
    poisson_ratio: float = 0.25  # Poisson's ratio (0.25 provides stable Lamé parameters)
    density: float = 1000.0  # Density (kg/m^3)

    # Simulation settings
    steps: int = 2000  # Total simulation steps (increased to observe longer propagation)
    plot_interval: int = 2  # Update visualization every N steps
    playback_seconds: float = 20.0  # Animation playback duration
    damping: float = 0.9995  # Energy damping factor (0.05% loss per step - preserves 49% amplitude at 2000 steps)
    cfl_safety_factor: float = 0.15  # CFL condition safety factor (slightly conservative, better wave resolution)
    time_integration: str = "implicit"  # Time integration method: 'explicit' or 'implicit'
    implicit_iterations: int = 30  # Maximum iterations for implicit solver
    implicit_tolerance: float = 1e-7  # Convergence tolerance for implicit solver
    boundary_type: str = "stress_free_all"  # Boundary condition type: 'fixed', 'free_top', or 'stress_free_all'

    # Source settings
    target_wavelength: float = 0.64  # Target wavelength (m) - primary design condition
    source_amplitude: float = 1e-3  # Source amplitude (meters) - increased 10x for better S-wave visibility
    source_duration_cycles: float = 10.0  # Source duration in cycles (increased to 10.0 for continuous wave observation)
    source_x_start: int = 1  # Source left boundary (driven BC applies to columns 0:source_x_end)
    source_x_end: int = 2  # Source right boundary (only first 2 columns driven, allows wave propagation)
    input_mode: str = "uniform_edge"  # Input mode: 'radial' or 'uniform_edge'
    wave_type: str = "p_wave"  # Wave type: 'p_wave', 's_wave', 'both_separate'
    source_mode: str = "boundary"  # Source mode: 'boundary' (left edge injection) or 'interior' (internal source)

    # Visualization settings
    colorbar_mode: str = "dynamic"  # Colorbar scale: 'source_amplitude' (fixed at ±1.15×A) or 'dynamic' (per-frame max)

    # Boundary settings
    boundary_width: int = 5  # Width of fixed boundary (grid points)

    # Numerical settings
    finite_diff_order: int = 4  # Finite difference order

    # GIF export settings
    gif_export: bool = False  # Enable GIF export mode
    gif_num_steps: int = 560  # Number of simulation steps for GIF
    gif_frame_rate: int = 10  # GIF frame rate (fps)
    gif_output_dir: str = "./animation"  # Output directory for GIF files
    gif_frame_interval: int = 2  # Visualization updates per solver step

    @property
    def lame_mu(self) -> float:
        """Lamé's first parameter (shear modulus)."""
        return self.youngs_modulus / (2 * (1 + self.poisson_ratio))

    @property
    def lame_lambda(self) -> float:
        """Lamé's second parameter."""
        return (self.youngs_modulus * self.poisson_ratio) / ((1 + self.poisson_ratio) * (1 - 2 * self.poisson_ratio))

    @property
    def p_wave_speed(self) -> float:
        """P-wave (compressional) speed (m/s).

        Calculated from: vp = sqrt((λ + 2μ) / ρ)
        With E=1000 Pa, ν=0.25, ρ=1000 kg/m³: vp ≈ 1.0954 m/s
        """
        return np.sqrt((self.lame_lambda + 2 * self.lame_mu) / self.density)

    @property
    def s_wave_speed(self) -> float:
        """S-wave (shear) speed (m/s)."""
        return np.sqrt(self.lame_mu / self.density)

    @property
    def input_freq(self) -> float:
        """Source frequency (Hz) calculated from wavelength condition.

        Computed as: f = vp / wavelength
        Ensures that the injected wavelength matches target_wavelength.
        """
        return self.p_wave_speed / self.target_wavelength

    @property
    def dt(self) -> float:
        """Time step size based on CFL condition for both P and S waves."""
        # Use minimum wave speed to ensure stability for both wave types
        min_wave_speed = min(self.p_wave_speed, self.s_wave_speed)
        return self.cfl_safety_factor * self.dx / (min_wave_speed * np.sqrt(2))

    @property
    def source_duration(self) -> float:
        """Source signal duration (seconds)."""
        return self.source_duration_cycles / self.input_freq

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.youngs_modulus <= 0:
            raise ValueError("Young's modulus must be positive")
        if not (0 <= self.poisson_ratio < 0.5):
            raise ValueError("Poisson's ratio must be in [0, 0.5)")
        if self.poisson_ratio > 0.48:
            import warnings

            warnings.warn(
                f"Poisson ratio {self.poisson_ratio} is very close to 0.5. "
                "This can cause numerical instability (ill-conditioned Lamé parameters). "
                "Consider using ν ≤ 0.45 for better stability, or use implicit time integration."
            )
        if self.density <= 0:
            raise ValueError("Density must be positive")
        if self.dx <= 0 or self.dy <= 0:
            raise ValueError("Grid spacing must be positive")
        if self.nx <= 0 or self.ny <= 0:
            raise ValueError("Grid dimensions must be positive")
        if self.boundary_width * 2 >= min(self.nx, self.ny):
            raise ValueError("Boundary width too large for grid size")
        if self.time_integration not in ["explicit", "implicit"]:
            raise ValueError("time_integration must be 'explicit' or 'implicit'")
        if self.boundary_type not in ["fixed", "free_top", "stress_free_all"]:
            raise ValueError("boundary_type must be 'fixed', 'free_top', or 'stress_free_all'")
        if self.target_wavelength <= 0:
            raise ValueError("Target wavelength must be positive")

    def print_info(self) -> None:
        """Print simulation configuration information."""
        print("=" * 60)
        print("Elastic Wave Simulation Configuration")
        print("=" * 60)
        print(f"Grid: {self.nx} x {self.ny} ({self.dx * 1e3:.2f} mm spacing)")
        print(f"Material: E={self.youngs_modulus:.2e} Pa, ν={self.poisson_ratio:.3f}, ρ={self.density:.1f} kg/m³")
        print(
            f"Lamé parameters: λ={self.lame_lambda:.2e} Pa, μ={self.lame_mu:.2e} Pa (λ/μ={self.lame_lambda / self.lame_mu:.2f})"
        )
        print(
            f"Wave speeds: vp={self.p_wave_speed:.2f} m/s, vs={self.s_wave_speed:.2f} m/s (vp/vs={self.p_wave_speed / self.s_wave_speed:.2f})"
        )
        print(f"Time step: {self.dt * 1e6:.4f} μs (CFL={self.cfl_safety_factor}, method={self.time_integration})")
        print(
            f"Boundary: {self.boundary_type} (Rayleigh waves: {'enabled' if self.boundary_type == 'free_top' else 'disabled'})"
        )
        print(
            f"Source: {self.input_freq:.4f} Hz (λ={self.target_wavelength} m), {self.source_amplitude * 1e6:.2f} μm amplitude"
        )
        print(f"Simulation: {self.steps} steps, {self.steps * self.dt * 1e3:.2f} ms total time")
        print("=" * 60)
