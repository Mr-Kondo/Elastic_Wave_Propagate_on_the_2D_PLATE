# Energy-form boundary and time discretization

Version 0.3 replaces the v0.2 strong-form displacement ghosts with a weak plane-strain formulation on connected, axis-aligned rectangular Q1 elements.

## Elastic energy and free boundaries

Let $\varepsilon(u)=(\nabla u+\nabla u^T)/2$, with out-of-plane strain zero. Define the 3D deviatoric part $\varepsilon^D=\varepsilon-\operatorname{tr}(\varepsilon)I/3$ and bulk modulus $B=\lambda+2\mu/3$. The discrete energy is

$$
W_h(u)=\sum_e\int_e\mu\,\varepsilon^D(u):\varepsilon^D(u)\,dA
+\sum_e\frac{B|e|}{2}\left(\overline{\nabla\cdot u}_e\right)^2.
$$

This B-bar projection fully integrates the deviatoric term and uses the cell-average volumetric strain. For the configured affine rectangular Q1 mesh, the average divergence equals its value at the cell centre. Assembly therefore uses scikit-fem's fully integrated `linear_elasticity(-2*mu/3, mu)` and a centre quadrature for `linear_elasticity(lambda+2*mu/3, 0)`.

The negative first Lamé argument in the deviatoric form is an algebraic split, not a material with negative total bulk modulus. For plane strain the deviatoric energy is

$$
\mu\left[\frac23(\varepsilon_{xx}^2+\varepsilon_{yy}^2-\varepsilon_{xx}\varepsilon_{yy})
+2\varepsilon_{xy}^2\right]\ge0.
$$

The volumetric term is also nonnegative. Thus the assembled stiffness $K$ is symmetric positive semidefinite in exact arithmetic. Its kernel on the connected free grid consists of two translations and one infinitesimal rotation. The one-element test checks that selective volumetric integration adds no hourglass zero modes.

Free traction is the natural boundary condition of this energy: no external traction integral is added on a free edge. The weak condition covers all edges meeting at a corner; it does not assign an artificial single normal to the corner point or force nodal pointwise stress to zero. No y-then-x ghost extension remains.

Fixed nodal strips are constrained. Boundary-input mode omits the left fixed strip and prescribes column 0, with precedence at its endpoints. For the implicit step, fixed columns are eliminated from the sparse linear system; prescribed values are retained in the right-hand side.

## Positive mass and timestep

The row-sum mass is $M_i=\rho\int N_i\,dA>0$ for these Q1 elements. In the absence of source work and damping,

$$
M\ddot u+Ku=0,\qquad
E=\tfrac12\dot u^TM\dot u+\tfrac12u^TKu.
$$

Hence $M^{-1}K$ is similar to a symmetric positive semidefinite matrix. This resolves the complex spatial eigenvalues of the old ghost operator.

For the supported $\lambda\ge0,\mu>0$ and uniform rectangular mesh, the full-integration energy and Q1 inverse estimate give

$$
\omega_{\max}^2\le
\frac{8(\lambda+\mu)}{\rho}\left(h_x^{-2}+h_y^{-2}\right).
$$

The volumetric projection can only reduce the full-integration energy, so this bound also applies to B-bar. The configured step is

$$
\Delta t=s\sqrt{\frac{\rho}{2(\lambda+\mu)}}\,
\left(h_x^{-2}+h_y^{-2}\right)^{-1/2},\qquad 0<s\le1.
$$

This is a stability bound, not a guarantee of adequate temporal or spatial resolution.

## Time integration

With damping disabled, explicit central differences use

$$
u^{n+1}=2u^n-u^{n-1}-\Delta t^2M^{-1}Ku^n.
$$

The average-acceleration recurrence uses

$$
(M+cK)u^{n+1}
=M(2u^n-u^{n-1})-cK(u^{n-1}+2u^n),\qquad c=\Delta t^2/4.
$$

SciPy factorizes the constrained matrix once. It is positive definite because the mass is positive. There is no fixed-point convergence tolerance or iteration limit. Nonfinite inputs/results are rejected before advancing the state.

For configured velocity retention $d$, the predictor is $(1+d)u^n-du^{n-1}$, the explicit acceleration coefficient is $\Delta t^2(1+d)/2$, and $c=\Delta t^2(1+d)/8$. This corresponds to a centred mass-proportional damping coefficient $2(1-d)/[\Delta t(1+d)]$. Source displacement is evaluated at $t+\Delta t$.

For $d=1$, let $v_{n-1/2}=(u^n-u^{n-1})/\Delta t$ and $\bar u=(u^n+u^{n-1})/2$. The conserved discrete energy is

$$
E_{\mathrm{implicit}}=\tfrac12v^TMv+\tfrac12\bar u^TK\bar u,\qquad
E_{\mathrm{explicit}}=E_{\mathrm{implicit}}-\tfrac{\Delta t^2}{8}v^TKv.
$$

The explicit energy is positive under a strict stable step. This is the energy measured by the validation script; it is not a claim that an arbitrary displacement/velocity estimator is exactly conserved.

## Verified scope

See [the v0.3 validation report](docs/energy-form-validation-2026-10-25.md) for spectral tests, long-time energy, P/S dispersion, Rayleigh convergence including $\nu=0.4999$, and normal reflection. Smooth-wave tests demonstrate second-order spatial and temporal convergence. Default finite-plate waves differ from the five-wavelength-deep periodic Rayleigh reference problem.

The model remains linear, homogeneous and plane strain. These checks do not calibrate a real material, validate a Kirchhoff plate approximation, or provide an error bound independent of mesh, source and geometry.

References: [scikit-fem](https://scikit-fem.readthedocs.io/), [Bower §8.6](https://solidmechanics.org/Text/Chapter8_6/Chapter8_6.php), [MOOSE NewmarkBeta](https://raw.githubusercontent.com/idaholab/moose/master/framework/doc/content/source/timeintegrators/NewmarkBeta.md).
