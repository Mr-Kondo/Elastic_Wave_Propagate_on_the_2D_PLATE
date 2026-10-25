"""Configuration and initialization smoke check."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from config import SimulationConfig
from main import _initialize_simulation


class ConfigurationTests(unittest.TestCase):
    def test_default_simulation(self):
        config = SimulationConfig()
        config.validate()
        state, solver = _initialize_simulation(config, "P")
        for step in range(5):
            solver.step(state, step * config.dt)
        self.assertTrue(np.isfinite(state.get_max_displacement()))
        self.assertGreater(state.get_max_displacement(), 0)


if __name__ == "__main__":
    unittest.main()
