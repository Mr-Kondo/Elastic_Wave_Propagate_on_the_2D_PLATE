# 2D Elastic Wave Simulation (MLX)

Educational simulation of two-dimensional, homogeneous isotropic linear elasticity on Apple Silicon. The constitutive law is **plane strain**, using the 3D Lamé parameters; this is not a thin-plate bending solver.

Interior spatial derivatives use fourth-order central differences. Free edges use a second-order traction closure, with sequential corner extension. The full method has **not** been demonstrated to have fourth-order global accuracy or quantitatively correct Rayleigh/Lamb/Head waves. See [the numerical limitations](STRESS_FREE_BC_IMPLEMENTATION.md) and [open validation work](https://github.com/Mr-Kondo/Elastic_Wave_Propagate_on_the_2D_PLATE/issues/9).

## Requirements and installation

- Apple Silicon macOS, Python **3.14** (tested with 3.14.9).
- MLX, NumPy, Matplotlib and Pillow; exact versions are recorded in `uv.lock`.

```bash
uv sync --locked
uv run python run.py
uv run python run.py --gif
```

The project selects Python 3.14 through `.python-version`. This does not change your system Python. Alternatively, install in a Python 3.14 virtual environment with `pip install .`.

## Configuration

Edit `SimulationConfig` in [src/config.py](src/config.py). Defaults:

| Parameter | Value |
|---|---|
| Grid | 101 × 20 nodes |
| Spacing | dx = 0.009901 m, dy = 0.0095 m |
| Node-to-node extent | 0.9901 m × 0.1805 m |
| Young's modulus / Poisson ratio / density | 1000 Pa / 0.25 / 1000 kg/m³ |
| Lamé parameters | λ = 400 Pa, μ = 400 Pa |
| Bulk P / S speeds | approximately 1.0954 / 0.6325 m/s |
| Input wavelength / frequency | 0.64 m / approximately 1.7116 Hz |
| Source | prescribed x-displacement on the entire left edge, amplitude 1 mm |
| Source duration | 10 cycles, approximately 5.8424 s |
| Boundaries | free top, bottom and right; prescribed left edge |
| Time integration | implicit average acceleration, fixed-point iteration |
| CFL safety factor / timestep | 0.15 / approximately 0.812891 ms |
| Simulation duration | 2000 steps, approximately 1.62578 s |
| Damping | 0.9995 per-step velocity retention |

The source remains active throughout the default simulation. `source_mode="interior"` instead adds a displacement source in `source_x_start:source_x_end`; `input_mode` selects radial or uniform polarization there. A radial polarization mask alone is not a proof of a pure P/S source.

`boundary_type` supports `fixed`, `free_top` and `stress_free_all`. A prescribed left-edge source takes precedence over a fixed or free left edge, including its endpoints. `boundary_width` controls fixed strips only; exterior ghost padding always has two layers.

The legacy `wave_type` configuration field does not change the source. `run_simulation(config, wave_type)` uses `"P"`, `"S"`, `"Both"` to select the initial displayed component; input remains x-polarized.

## Display and GIF output

- **p**: x-displacement; **s**: y-displacement; **b**: displacement magnitude.
- Components of displacement are **not** a general P/S decomposition.
- Axes and contour coordinates use metres. Dynamic scaling is per displayed frame.
- GIF export collects globally scaled component fields and renders one component at a time with Pillow. All-zero fields use a small nonzero plotting range.
- Existing filenames are preserved: `p_wave_animation.gif`, `s_wave_animation.gif`, `combined_magnitude.gif`. The first two contain x/y displacement, respectively.
- `gif_num_steps`, `gif_frame_interval`, `gif_frame_rate` and `gif_output_dir` control export. A final partial interval produces a final frame.
- GIF timing is quantized to the format's 10 ms units. Default 10 fps is 100 ms per frame.

## Numerical method and checks

The solver advances the Navier equation:

$$
\rho\,\ddot{\mathbf{u}}
=\mu\nabla^2\mathbf{u}+(\lambda+\mu)\nabla(\nabla\cdot\mathbf{u}),
\quad
\mu=\frac{E}{2(1+\nu)},\quad
\lambda=\frac{E\nu}{(1+\nu)(1-2\nu)}.
$$

The timestep uses the fastest bulk wave and both spacings:

$$
\Delta t=s\,\frac{\sqrt{3}}{2c_p\sqrt{1/\Delta x^2+1/\Delta y^2}}.
$$

This is an interior-stencil bound, not a proof of stability for every boundary/material setting. Both explicit central differences and implicit average acceleration use this conservative timestep. The implicit solve applies constraints during iteration, has a float32 tolerance floor of four machine epsilons, and raises `RuntimeError` without advancing the state if it fails to converge. Nonfinite output is also an error. Reduce `cfl_safety_factor` or increase `implicit_iterations` when needed; “implicit” does not make this fixed-point solve unconditionally convergent.

`damping=1` disables velocity damping. A smaller value damps velocity increments, preserving static rigid translation; it is not an energy-retention factor.

```bash
uv run --locked python -m unittest discover -s test -v
uv lock --check
uv pip check
```

The checks cover configuration rejection, polynomial derivatives, operator reuse, rigid motion, edge traction away from corners, source geometry and timing, second-order time convergence on an oscillator, nonconvergence, NaNs, exact step counts, GIF scaling and file timing. Default 2000-step runs and the boundary-mode checks are recorded in [the audit](docs/code-audit-2026-10-25.md). These checks do not establish full continuum-model accuracy.

## Development and references

- [Boundary equations and limitations](STRESS_FREE_BC_IMPLEMENTATION.md)
- [Audit, library decisions and validation](docs/code-audit-2026-10-25.md)
- Raw execution records belong in `feedback/`, which is excluded from Git. Bytecode, local environments, `.DS_Store` and local agent metadata are also excluded. Reference papers and example animations remain tracked.
- [J. Bleyer: Linear elasticity](https://bleyerj.github.io/comet-fenicsx/intro/linear_elasticity/linear_elasticity.html) — stress/strain and traction definitions.
- [MOOSE: NewmarkBeta](https://raw.githubusercontent.com/idaholab/moose/master/framework/doc/content/source/timeintegrators/NewmarkBeta.md) — average-acceleration time integration; stability claims for its method do not prove stability of this solver's boundary closure.
- [Pillow GIF format](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif).
- Contextual paper retained in `article/81001946.pdf`: Uenishi (2009), [DOI 10.24546/81001946](https://doi.org/10.24546/81001946). It discusses MPS; this repository implements finite differences, not MPS, and the citation is not numerical validation of this implementation.
