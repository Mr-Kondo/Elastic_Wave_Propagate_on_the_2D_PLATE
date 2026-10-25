"""
Wave field state management for elastic wave simulation.
"""

from typing import Literal

import numpy as np

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
        self.u = np.zeros((1, ny, nx, 1))  # X-displacement
        self.v = np.zeros((1, ny, nx, 1))  # Y-displacement
        self.u_prev = np.zeros((1, ny, nx, 1))
        self.v_prev = np.zeros((1, ny, nx, 1))

        # Wave type selection - ALWAYS P-wave input (for source injection)
        # Display selection does not decompose the field into wave modes.
        self.wave_type: WaveType = "P"

        # Display mode - controls what field is visualized (independent of input)
        self.display_mode: WaveType = "P"

    def advance(self, u_next: np.ndarray, v_next: np.ndarray) -> None:
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

    def get_display_field(self) -> np.ndarray:
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
            return np.sqrt(self.u**2 + self.v**2)

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
        max_u = np.max(np.abs(self.u))
        max_v = np.max(np.abs(self.v))
        return np.maximum(max_u, max_v).item()

    def reset(self) -> None:
        """Reset all fields to zero."""
        self.u = np.zeros((1, self.ny, self.nx, 1))
        self.v = np.zeros((1, self.ny, self.nx, 1))
        self.u_prev = np.zeros((1, self.ny, self.nx, 1))
        self.v_prev = np.zeros((1, self.ny, self.nx, 1))
