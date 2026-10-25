"""Independent analytical and conservation checks for the energy-form operator."""

import sys
import unittest
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root), str(root / "src")]

import numpy as np
from scipy.linalg import eigvalsh

from config import SimulationConfig
from solver import ElasticOperator
from validation.validate_physics import energy_check, rayleigh_speed


class EnergyFormTests(unittest.TestCase):
    def test_selective_integration_has_no_hourglass_modes(self):
        config = SimulationConfig(nx=2, ny=2, poisson_ratio=.4999)
        operator = ElasticOperator(config)
        values = eigvalsh(operator.stiffness.toarray())
        tolerance = 100*np.finfo(float).eps*values[-1]
        self.assertEqual(np.count_nonzero(np.abs(values)<tolerance), 3)
        self.assertTrue(np.all(values[3:] > tolerance))

    def test_energy_conservation_without_damping(self):
        for method in ("explicit", "implicit"):
            energy_check(.4999, 2, method, 300)

    def test_nearly_incompressible_rayleigh_mode(self):
        result = rayleigh_speed(32, nu=.4999)
        self.assertLess(result["relative_error"], .002)
        self.assertGreater(min(result["surface_mass_fraction"]), .65)


if __name__ == "__main__":
    unittest.main()
