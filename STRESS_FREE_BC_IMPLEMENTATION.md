# Free-boundary implementation and limitations

For the plane-strain displacement field (u, v),

$$
\sigma_{xx}=(\lambda+2\mu)u_x+\lambda v_y,\qquad
\sigma_{yy}=\lambda u_x+(\lambda+2\mu)v_y,\qquad
\sigma_{xy}=\mu(u_y+v_x).
$$

A free edge requires **traction** \(\boldsymbol\sigma\mathbf n=0\), not every component of stress to vanish. A horizontal free edge therefore imposes

$$
v_y=-\alpha u_x,\qquad u_y=-v_x,\qquad
\alpha=\frac{\lambda}{\lambda+2\mu}.
$$

A vertical free edge imposes

$$
u_x=-\alpha v_y,\qquad v_x=-u_y.
$$

## Discretization

`BoundaryConditions.pad_fields` constructs two **exterior** ghost layers for each spatial derivative evaluation. It keeps all physical grid points. Tangential derivatives use central differences on edge interiors and one-sided differences at their endpoints.

For a normal-coordinate derivative g at a boundary, distance kh from it, and k=1,2, the reflected interior node is extended as

$$
\phi_{-k}=\phi_k-2khg_0,\qquad
\phi_{N-1+k}=\phi_{N-1-k}+2khg_{N-1}.
$$

Opposite sides use the sign of the actual coordinate distance. The two ghost layers use different reflected nodes and distances. This replaces the earlier constant-value overwriting of physical boundary strips. The same coupled traction equations now apply to the top surface of `free_top`.

Dirichlet sides use odd reflection around their prescribed boundary value, \(\phi_{-k}=2\phi_0-\phi_k\). Fixed strips are masked in the physical grid. A driven left edge takes precedence at its endpoints. The width of a fixed strip is independent of the two ghost layers.

## Time integration

Let L be the acceleration operator. With damping disabled, the explicit recurrence is

$$
u^{n+1}=2u^n-u^{n-1}+\Delta t^2L(u^n).
$$

The implicit average-acceleration recurrence uses

$$
u^{n+1}=2u^n-u^{n-1}
+\frac{\Delta t^2}{4}\left[L(u^{n-1})+2L(u^n)+L(u^{n+1})\right].
$$

Both components are coupled through L. Constraints are applied within the fixed-point iteration. Failure to converge raises an error before `state.advance`. The relative maximum-norm convergence test has a floor of four float32 epsilons.

For configured velocity retention d, the central velocity-damping discretization replaces the first two terms with \((1+d)u^n-du^{n-1}\), and multiplies the acceleration contribution by \((1+d)/2\). It leaves a static rigid translation unchanged. Prescribed boundary displacement is evaluated at the new time t+dt.

## Verified scope and open work

- Polynomial tests check the fourth-order **interior stencil**, including reuse on different grid sizes.
- Translation and infinitesimal rigid rotation produce zero strain and approximately zero acceleration.
- Discrete normal/shear traction residuals on each straight edge are checked **away from corners**.
- An oscillator verifies second-order time convergence in both integrators.
- Existing default-material 2000-step runs check finite, bounded displacement; all three boundary modes are also exercised.
- Actual GIFs are reopened to verify frame count and duration.

**Limitations:** the edge closure is second order. Corners extend y first and x second; simultaneous corner traction, global convergence order, energy stability for arbitrary materials/grids, reflected-wave amplitudes and Rayleigh/Lamb/Head-wave dispersion are unverified. Corner residuals are observable and are not hidden by the edge test. These limitations remain in [Issue #9](https://github.com/Mr-Kondo/Elastic_Wave_Propagate_on_the_2D_PLATE/issues/9). Passing a smoke test does not establish any of these stronger properties.

References: [Bleyer, Linear elasticity](https://bleyerj.github.io/comet-fenicsx/intro/linear_elasticity/linear_elasticity.html), [MOOSE NewmarkBeta](https://raw.githubusercontent.com/idaholab/moose/master/framework/doc/content/source/timeintegrators/NewmarkBeta.md). Neither source validates this particular ghost/corner closure.
