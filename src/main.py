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
from gif_exporter import GifExporter


def _initialize_simulation(config: SimulationConfig, wave_type: str) -> tuple:
    """
    Initialize simulation infrastructure.

    Args:
        config: Simulation configuration
        wave_type: 'P' for P-wave, 'S' for S-wave, 'Both' for both

    Returns:
        Tuple of (state, solver)
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

    return state, solver


def run_simulation(config: SimulationConfig, wave_type: str, export_gif: bool = False) -> None:
    """
    Run a single simulation with specified wave type.

    Args:
        config: Simulation configuration
        wave_type: 'P' for P-wave, 'S' for S-wave, 'Both' for both
        export_gif: If True, export to GIF; otherwise run visualization
    """
    state, solver = _initialize_simulation(config, wave_type)

    if export_gif:
        # Use GIF exporter
        config.steps = config.gif_num_steps
        exporter = GifExporter(config, state, solver)
        exporter.run()
    else:
        # Initialize visualizer and run simulation
        visualizer = WaveVisualizer(config, state, solver, wave_type=wave_type)
        visualizer.run()


def main(export_gif: bool = False):
    """
    Main entry point for elastic wave simulation.

    Args:
        export_gif: If True, export animations as GIF files instead of displaying real-time visualization
    """

    # Initialize configuration with default parameters
    config = SimulationConfig()

    # Validate and display configuration
    config.validate()
    config.print_info()

    if export_gif:
        print("\n" + "=" * 60)
        print("GIF Export Mode")
        print(f"Steps: {config.gif_num_steps}")
        print(f"Frame rate: {config.gif_frame_rate} fps")
        print(f"Output directory: {config.gif_output_dir}")
        print("Exporting animations: P-wave, S-wave, Combined magnitude")
        print("=" * 60)
    else:
        # Always run P-wave input (display can be switched with p/s/b keys)
        # S-wave and combined modes arise from mode conversion
        print("\n" + "=" * 60)
        print("Running Elastic Wave Simulation (P-wave Input)")
        print("Use keyboard shortcuts to switch display mode:")
        print("  p: Show P-wave (X-displacement, compressional)")
        print("  s: Show S-wave (Y-displacement, mode conversion)")
        print("  b: Show both (combined magnitude)")
        print("=" * 60)

    run_simulation(config, "P", export_gif=export_gif)


if __name__ == "__main__":
    main()
