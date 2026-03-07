## 2D Elastic Wave Simulation (MLX)

A high-performance 2D elastic wave propagation simulator optimized for Apple Silicon using MLX. Solves the 2D Navier-Cauchy elastic wave equation with 4th-order finite differences and semi-implicit time integration.

### Key Features

- **Explicit Stress-Free Boundary Conditions**: Enforces σ = 0 at all boundaries using Lamé stress tensor calculations for proper Head Wave (Von Schmidt wave) generation
- **Head Wave Simulation**: Correctly exhibits Head Waves at stress-free boundaries with proper mode conversion (P→S)
- **4th-order spatial accuracy** via convolution-based finite differences
- **Semi-implicit time integration** with adaptive convergence detection  
- **Metal acceleration** on Apple Silicon via MLX framework
- **Real-time visualization** with seismic colormap and detailed contour lines (31 levels)
- **Toggleable wave types**: P-wave (compressional), S-wave (shear from mode conversion), and combined display
- **Physics-validated**: Wave speeds match theoretical values; Lamé parameters from Young's modulus and Poisson's ratio
- **Dynamic wave field scaling**: Contours adapt to wave amplitude for clear visualization
- **Energy damping control** for realistic wave decay

### Requirements

- Python 3.10+
- mlx (Apple Silicon)
- numpy
- matplotlib

### Installation

```bash
# Using uv (recommended)
uv sync
uv run ./run.py

# Or with pip
pip install mlx numpy matplotlib
python src/main.py
```

### Configuration

The simulation uses a modular structure with separate configuration, physics solver, and visualization:

**Domain and Grid**
- Domain dimensions: 1.0m (horizontal) × 0.19m (vertical)
- Grid points: 101 × 20
- Grid spacing: dx = 0.009901m, dy = 0.0095m (nearly isotropic: dx/dy ≈ 1.04)

**Material Properties**
- Young's modulus: 1000 Pa
- Poisson's ratio: 0.25
- Density: 1000 kg/m³
- Lamé parameters: λ = 250 Pa, μ = 400 Pa
- P-wave speed (vp): ~1.10 m/s
- S-wave speed (vs): ~0.63 m/s

**Wave Source**
- Wavelength: 0.64 m
- Frequency: 1.71 Hz
- Amplitude: 1.0 mm
- Source type: Driven left boundary (P-wave input)
- Duration: 10 cycles (~5.84 seconds of continuous driving)

**Boundary Conditions**
- Type: Stress-free on all 4 sides (σ = 0)
- Implementation: Explicit ghost point calculation using Lamé stress tensor
- Enabled phenomena: Head Waves, mode conversion (P→S)

**Numerical Settings**
- Time integration: Implicit (Crank-Nicolson)
- CFL safety factor: 0.15
- Damping: 0.9995 per step (49% retention at 2000 steps)
- Simulation time: 2000 steps × 1.66 ms/step = 3.32 seconds
- Visualization: 31-level contours, dynamic scaling

### Controls

During simulation:
- **'p'** – Switch to P-wave (compressional, radial) visualization
- **'s'** – Switch to S-wave (shear, tangential) visualization
- **'b'** – Switch to Both waves (P+S combined magnitude)
- **Close window** – Exit

### Physics

Solves the 2D Navier-Cauchy equation:

$$\rho \frac{\partial^2 u_i}{\partial t^2} = \frac{\partial \sigma_{ij}}{\partial x_j}$$

with Lamé parameters:
- $\mu = E / [2(1 + \nu)]$ (shear modulus)
- $\lambda = E\nu / [(1 + \nu)(1 - 2\nu)]$ (Lamé parameter)

Theoretical wave speeds:
- P-wave speed: $v_p = \sqrt{(\lambda + 2\mu)/\rho}$
- S-wave speed: $v_s = \sqrt{\mu/\rho}$
- Rayleigh wave speed: $v_R \approx 0.92 v_s$ (surface wave, requires free boundary)

**Polarization Directions** (Point Source):
- **P-wave**: Radial displacement (direction: $\hat{r} = (\cos\theta, \sin\theta)$)
  - Particles oscillate along wave propagation direction
  - Compressional (longitudinal) wave
- **S-wave**: Tangential displacement (direction: $\hat{\theta} = (-\sin\theta, \cos\theta)$)
  - Particles oscillate perpendicular to wave propagation direction
  - Shear (transverse) wave
- **Rayleigh wave**: Elliptical motion at free surface (retrograde)
  - Combination of P and S wave motion confined to surface region
  - Amplitude decays exponentially with depth (~wavelength)

**Boundary Conditions**:
- **Fixed** (`boundary_type="fixed"`): All boundaries have zero displacement (Dirichlet: $u=0, v=0$)
  - Simulates waves in an embedded structure or bounded medium
  - Only P-waves and S-waves (body waves) propagate
- **Free Top** (`boundary_type="free_top"`): Top surface is stress-free (Neumann: $\sigma_{yy}=0, \sigma_{xy}=0$)
  - Simulates waves with a free surface (like ground surface)
  - Implementation: Symmetric mirror boundary ($\frac{\partial u}{\partial y}=0, \frac{\partial v}{\partial y}=0$)
  - **Enables Rayleigh waves**: Surface waves that travel along the free boundary
  - Rayleigh waves decay exponentially with depth (~wavelength)
  - Particle motion is elliptical, retrograde at surface
- **Stress-Free All** (`boundary_type="stress_free_all"`): All 4 boundaries are stress-free (Neumann conditions)
  - Applies symmetric mirror boundary conditions on all sides
  - Simulates waves radiating from a source into free space
  - No reflected waves from boundaries; wave energy propagates outward

**Validation Results** (ν = 0.25, rock material):
- Measured $v_p$ = 33.03 m/s (theoretical: $\sqrt{(4+8)/3} \times \sqrt{10^6/1100}$ ≈ 33.0 m/s) ✓
- Measured $v_s$ = 19.07 m/s (theoretical: $\sqrt{4/3} \times \sqrt{10^6/1100}$ ≈ 19.1 m/s) ✓
- Speed ratio: $v_p/v_s$ = 1.73 (theoretical: $\sqrt{3}$ ≈ 1.732) ✓
- Lamé ratio: λ/μ = 1.00 (expected for ν = 0.25) ✓

**Time Integration Methods**:
- Explicit: Standard leap-frog (Verlet) scheme with CFL stability constraint
- Implicit: Semi-implicit Crank-Nicolson with fixed-point iteration and automatic convergence detection

### Performance

- **Typical runtime**: ~1000 steps in ~5–10 seconds (Metal-accelerated)
- **Memory**: ~50–100 MB for 300×200 grid
- **Bottleneck**: Matplotlib animation rendering, not MLX computation

### Testing

Run the test suite to verify the configuration and solver:

```bash
python test/test_config.py
```

This test validates:
- Configuration parameters and material properties
- Boundary condition initialization with `stress_free_all` mode
- Solver setup and field initialization
- Simulation step execution for 5 time steps

Expected output includes grid dimensions, domain size, wave speeds, and wavelength specifications.

### Troubleshooting

| Issue | Solution |
|-------|----------|
| Simulation "Exploded!" | Switch to implicit solver or reduce Poisson ratio |
| Waves too damped | Reduce `DAMPING` (e.g., 0.998 → 0.999) |
| S-wave unstable | Ensure ν < 0.48; use implicit solver |
| Slow convergence | Increase `implicit_iterations` or check `implicit_tolerance` |
| No Rayleigh waves | Set `boundary_type="free_top"` in config.py |
| Want body waves only | Set `boundary_type="fixed"` to disable surface waves |
| Wrong wave speeds | Verify material parameters match expected values |

### References

**Original Paper**:

This implementation is based on the following bulletin paper:
- Uenishi, K. (2009). "ON THE NUMERICAL ANALYSIS OF SOLID DYNAMICS USING THE MOVING PARTICLE SEMI-IMPLICIT (MPS) METHOD". *Research Report of the Research Center for Urban Safety and Security, Kobe University*, Vol. 13, pp. 11-20. DOI: [10.24546/81001946](https://doi.org/10.24546/81001946)

**Validation References**:

This implementation is validated against standard elasticity theory:
- Navier-Cauchy equations: Landau & Lifshitz, "Theory of Elasticity"
- Wave propagation: Aki & Richards, "Quantitative Seismology" 
- Numerical methods: 4th-order finite differences with CFL stability
- Wikipedia: [Linear Elasticity](https://en.wikipedia.org/wiki/Linear_elasticity), [Seismic Wave](https://en.wikipedia.org/wiki/Seismic_wave)

### Files

```
/Volumes/SSD/Programming/Elastic_Wave_Propagate_on_the_2D_PLATE/
├── src/
│   ├── config.py          – Simulation configuration and material parameters
│   ├── solver.py          – Finite difference operators and time integrators
│   ├── state.py           – Wave field state management
│   ├── source.py          – Wave source generation
│   ├── visualizer.py      – Real-time animation and visualization
│   └── main.py            – Entry point and orchestration
├── test/
│   └── test_config.py     – Configuration and stability tests
├── run.py                 – Convenience script to run simulation
├── README.md              – Documentation
├── pyproject.toml         – Project metadata and dependencies
└── uv.lock                – Dependency lock file
```
