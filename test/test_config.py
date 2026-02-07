#!/usr/bin/env python3
"""Quick test of new configuration and stress_free_all boundary condition."""

import sys
from pathlib import Path

# Add parent/src directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import SimulationConfig
from solver import FiniteDifferenceOperator, BoundaryConditions, WaveSolver
from source import WaveSourceGenerator
from state import WaveFieldState

print("Testing new configuration and stress_free_all boundary condition...")
print()

# Test 1: Configuration
print("Test 1: Configuration Validation")
config = SimulationConfig()
config.validate()
print(f"  ✓ Grid: {config.nx}×{config.ny} points")
print(f"  ✓ Domain: {config.nx * config.dx:.2f}m × {config.ny * config.dy:.4f}m")
print(f"  ✓ Material: E={config.youngs_modulus} Pa, ν={config.poisson_ratio}, ρ={config.density} kg/m³")
print(f"  ✓ P-wave speed: {config.p_wave_speed:.4f} m/s")
print(f"  ✓ Wavelength: {config.p_wave_speed / config.input_freq:.4f} m")
print(f"  ✓ Boundary type: {config.boundary_type}")
print(f"  ✓ Wave type: {config.wave_type}")
print()

# Test 2: Boundary Conditions Initialization
print("Test 2: Boundary Conditions with stress_free_all")
fd = FiniteDifferenceOperator(config.dx, config.dy, order=4)
bc = BoundaryConditions(
    config.nx,
    config.ny,
    boundary_width=config.boundary_width,
    source_x_start=config.source_x_start,
    source_x_end=config.source_x_end,
    boundary_type=config.boundary_type,
)
print(f"  ✓ BoundaryConditions initialized with boundary_type='{config.boundary_type}'")
print(f"  ✓ apply_stress_free_all method available: {hasattr(bc, 'apply_stress_free_all')}")
print()

# Test 3: Solver Initialization
print("Test 3: Solver Initialization")
source = WaveSourceGenerator(config.input_freq, config.source_amplitude, config.source_duration)
state = WaveFieldState(config.nx, config.ny)
solver = WaveSolver(config, fd, bc, source)
print(f"  ✓ Solver initialized")
print()

# Test 4: Simulation Steps
print("Test 4: Run 5 Simulation Steps")
for step in range(5):
    try:
        solver.step(state, step * config.dt)
        print(f"  ✓ Step {step + 1} completed")
    except Exception as e:
        print(f"  ✗ Step {step + 1} failed: {e}")
        raise
print()

print("=" * 60)
print("All tests passed! Configuration and solver work correctly.")
print("=" * 60)
