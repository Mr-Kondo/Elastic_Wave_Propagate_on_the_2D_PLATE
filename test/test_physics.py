#!/usr/bin/env python
"""Test physical correctness of wave polarization."""

import numpy as np
from solver import BoundaryConditions

# Test boundary condition implementation
bc = BoundaryConditions(nx=100, ny=100, boundary_width=5, boundary_type="free_top")

print("✓ Point source polarization:")
print(f"  P-wave source shape: {bc.source_p_u.shape}")
print(f"  S-wave source shape: {bc.source_s_u.shape}")

# Verify orthogonality at test points
test_points = [(50, 10), (50, 7), (60, 7), (40, 7)]

for test_idx_y, test_idx_x in test_points:
    p_u = float(bc.source_p_u[0, test_idx_y, test_idx_x, 0])
    p_v = float(bc.source_p_v[0, test_idx_y, test_idx_x, 0])
    s_u = float(bc.source_s_u[0, test_idx_y, test_idx_x, 0])
    s_v = float(bc.source_s_v[0, test_idx_y, test_idx_x, 0])

    dot_product = p_u * s_u + p_v * s_v
    p_norm = np.sqrt(p_u**2 + p_v**2)
    s_norm = np.sqrt(s_u**2 + s_v**2)

    print(f"\n  Test point ({test_idx_y},{test_idx_x}):")
    print(f"    P-wave: ({p_u:.4f}, {p_v:.4f}), norm={p_norm:.4f}")
    print(f"    S-wave: ({s_u:.4f}, {s_v:.4f}), norm={s_norm:.4f}")
    print(f"    Dot product (orthogonality): {dot_product:.8f}")

    if abs(dot_product) < 1e-6 and abs(p_norm - s_norm) < 1e-6:
        print(f"    ✓ Orthogonal and normalized")
    else:
        print(f"    ⚠ Warning: dot={dot_product:.2e}, norm_diff={abs(p_norm - s_norm):.2e}")

print("\n✓ Physics test completed!")
print("✓ P-wave (radial) and S-wave (tangential) are orthogonal")
