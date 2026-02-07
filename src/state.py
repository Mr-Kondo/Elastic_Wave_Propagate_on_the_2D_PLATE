"""
Wave field state management for elastic wave simulation.
"""

import mlx.core as mx
from typing import Literal


WaveType = Literal["P", "S", "Both"]


class WaveFieldState:
    """Manages displacement fields and simulation state."""

    def __init__(self, nx: int, ny: int):
        """
        Initialize wave field state.

        Args:
            nx: Grid points in x direction
            ny: Grid points in y direction
        """
        self.nx = nx
        self.ny = ny

        # Displacement fields (current and previous time steps)
        self.u = mx.zeros((1, ny, nx, 1))  # X-displacement
        self.v = mx.zeros((1, ny, nx, 1))  # Y-displacement
        self.u_prev = mx.zeros((1, ny, nx, 1))
        self.v_prev = mx.zeros((1, ny, nx, 1))

        # Wave type selection - ALWAYS P-wave input (for source injection)
        # S-wave and combined modes arise from mode conversion
        self.wave_type: WaveType = "P"

        # Display mode - controls what field is visualized (independent of input)
        self.display_mode: WaveType = "P"

    def advance(self, u_next: mx.array, v_next: mx.array) -> None:
        """
        Advance simulation to next time step.

        Args:
            u_next: New X-displacement field
            v_next: New Y-displacement field
        """
        self.u_prev = self.u
        self.v_prev = self.v
        self.u = u_next
        self.v = v_next

    def get_display_field(self) -> mx.array:
        """
        Get the appropriate displacement field for visualization.
        Uses display_mode, not wave_type (which controls source injection).

        Returns:
            Current displacement field based on display mode
        """
        if self.display_mode == "P":
            return self.u
        elif self.display_mode == "S":
            return self.v
        else:  # Both
            # Show combined magnitude: sqrt(u^2 + v^2)
            return mx.sqrt(self.u**2 + self.v**2)

    def set_wave_type(self, wave_type: WaveType) -> None:
        """
        Deprecated: source is always P-wave. Use set_display_mode() instead.
        Kept for backward compatibility.

        Args:
            wave_type: (ignored - always P-wave input)
        """
        self.wave_type = "P"

    def set_display_mode(self, display_mode: WaveType) -> None:
        """
        Set the display mode for visualization (P, S, or combined).
        Does NOT affect source injection (always P-wave).

        Args:
            display_mode: 'P' for X-displacement, 'S' for Y-displacement, 'Both' for magnitude
        """
        self.display_mode = display_mode

    def get_max_displacement(self) -> float:
        """
        Get maximum absolute displacement in current fields.

        Returns:
            Maximum displacement magnitude
        """
        max_u = mx.max(mx.abs(self.u)).item()
        max_v = mx.max(mx.abs(self.v)).item()
        return max(max_u, max_v)

    def reset(self) -> None:
        """Reset all fields to zero."""
        self.u = mx.zeros((1, self.ny, self.nx, 1))
        self.v = mx.zeros((1, self.ny, self.nx, 1))
        self.u_prev = mx.zeros((1, self.ny, self.nx, 1))
        self.v_prev = mx.zeros((1, self.ny, self.nx, 1))
