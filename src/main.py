"""
2D Elastic Wave Propagation Simulation using MLX

Simulates P-wave and S-wave propagation in a 2D elastic medium
using the Navier-Cauchy equation with 4th-order finite differences.
"""

from config import SimulationConfig
from state import WaveFieldState
from source import WaveSourceGenerator
from solver import FiniteDifferenceOperator, BoundaryConditions, WaveSolver
from visualizer import WaveVisualizer


def run_simulation(config: SimulationConfig, wave_type: str) -> None:
    """
    Run a single simulation with specified wave type.

    Args:
        config: Simulation configuration
        wave_type: 'P' for P-wave, 'S' for S-wave, 'Both' for both
    """
    # Set the wave type in config
    config.wave_type = wave_type

    # Initialize wave field state
    state = WaveFieldState(config.nx, config.ny)
    state.set_wave_type(wave_type)

    # Initialize finite difference operator
    fd_operator = FiniteDifferenceOperator(config.dx, config.dy, order=4)

    # Initialize boundary conditions
    bc = BoundaryConditions(
        config.nx,
        config.ny,
        boundary_width=config.boundary_width,
        source_x_start=config.source_x_start,
        source_x_end=config.source_x_end,
        boundary_type=config.boundary_type,
        source_mode=config.source_mode,
        config=config,
        fd_operator=fd_operator,
    )

    # Initialize wave source generator
    source = WaveSourceGenerator(
        frequency=config.input_freq, amplitude=config.source_amplitude, duration=config.source_duration
    )

    # Initialize wave solver
    solver = WaveSolver(config, fd_operator, bc, source)

    # Initialize visualizer and run simulation
    visualizer = WaveVisualizer(config, state, solver, wave_type=wave_type)
    visualizer.run()


def main():
    """Main entry point for elastic wave simulation."""

    # Initialize configuration with default parameters
    config = SimulationConfig()

    # Validate and display configuration
    config.validate()
    config.print_info()

    # Always run P-wave input (display can be switched with p/s/b keys)
    # S-wave and combined modes arise from mode conversion
    print("\n" + "=" * 60)
    print("Running Elastic Wave Simulation (P-wave Input)")
    print("Use keyboard shortcuts to switch display mode:")
    print("  p: Show P-wave (X-displacement, compressional)")
    print("  s: Show S-wave (Y-displacement, mode conversion)")
    print("  b: Show both (combined magnitude)")
    print("=" * 60)
    run_simulation(config, "P")


if __name__ == "__main__":
    main()
