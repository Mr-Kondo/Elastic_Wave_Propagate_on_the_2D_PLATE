"""
Wave source generation for elastic wave simulation.
"""

import numpy as np


class WaveSourceGenerator:
    """Generates time-dependent wave source signals."""

    def __init__(self, frequency: float, amplitude: float, duration: float):
        """
        Initialize wave source generator.

        Args:
            frequency: Source frequency (Hz)
            amplitude: Source amplitude (meters)
            duration: Source duration (seconds)
        """
        self.frequency = frequency
        self.amplitude = amplitude
        self.duration = duration

    def get_amplitude(self, t: float) -> float:
        """
        Get source amplitude at given time.

        Args:
            t: Current simulation time (seconds)

        Returns:
            Source amplitude at time t
        """
        if t > self.duration:
            return 0.0
        return self.amplitude * np.sin(2 * np.pi * self.frequency * t)
