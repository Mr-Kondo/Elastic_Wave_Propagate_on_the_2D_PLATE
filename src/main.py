"""
2D Elastic Wave Propagation Simulation

Simulates P-wave and S-wave propagation in a 2D elastic medium
using a Q1 energy-form finite-element discretization.
"""

from config import SimulationConfig
from gif_exporter import GifExporter
from solver import BoundaryConditions, ElasticOperator, WaveSolver
from source import WaveSourceGenerator
from state import WaveFieldState
from visualizer import WaveVisualizer


def _initialize_simulation(config: SimulationConfig, wave_type: str) -> tuple:
    """
    Initialize simulation infrastructure.

    Args:
        config: Simulation configuration
        wave_type: display component: 'P' for x, 'S' for y, 'Both' for magnitude; input stays P-polarized

    Returns:
        Tuple of (state, solver)
    """
    config.validate()

    # Initialize wave field state
    state = WaveFieldState(config.nx, config.ny)
    state.set_display_mode(wave_type)

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
    )

    # Initialize wave source generator
    source = WaveSourceGenerator(
        frequency=config.input_freq, amplitude=config.source_amplitude, duration=config.source_duration
    )

    # Initialize wave solver
    operator = ElasticOperator(config)
    solver = WaveSolver(config, operator, bc, source)

    return state, solver


def run_simulation(config: SimulationConfig, wave_type: str, export_gif: bool = False) -> None:
    """
    Run a single simulation with specified wave type.

    Args:
        config: Simulation configuration
        wave_type: display component: 'P' for x, 'S' for y, 'Both' for magnitude; input stays P-polarized
        export_gif: If True, export to GIF; otherwise run visualization
    """
    state, solver = _initialize_simulation(config, wave_type)

    if export_gif:
        # Use GIF exporter
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
        print("  p: Show X-displacement")
        print("  s: Show Y-displacement")
        print("  b: Show displacement magnitude")
        print("=" * 60)

    run_simulation(config, "P", export_gif=export_gif)


if __name__ == "__main__":
    main()
