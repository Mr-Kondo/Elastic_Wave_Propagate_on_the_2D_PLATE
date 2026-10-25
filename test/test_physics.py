"""Source polarization geometry; this does not validate propagated wave modes."""

import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from config import SimulationConfig
from solver import BoundaryConditions


class PolarizationTests(unittest.TestCase):
    def test_active_source_directions_in_physical_coordinates(self):
        config = SimulationConfig(nx=25, ny=17, dx=0.03, dy=0.01, source_x_start=6, source_x_end=8)
        bc = BoundaryConditions(config.nx, config.ny, source_x_start=6, source_x_end=8, config=config)
        for y, x in ((8, 6), (8, 7), (10, 7), (6, 7)):
            p = np.array([bc.source_p_u[0, y, x, 0].item(), bc.source_p_v[0, y, x, 0].item()])
            s = np.array([bc.source_s_u[0, y, x, 0].item(), bc.source_s_v[0, y, x, 0].item()])
            radial = np.array([(x-6.5)*config.dx, (y-8)*config.dy])
            np.testing.assert_allclose(p, radial/np.linalg.norm(radial), atol=1e-7)
            self.assertAlmostEqual(float(np.dot(p, s)), 0, places=6)
            self.assertAlmostEqual(float(np.linalg.norm(s)), 1, places=6)


if __name__ == "__main__":
    unittest.main()
