"""Reproducible physics checks. Print JSON; store execution records under feedback/."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq
from scipy.sparse import csr_matrix

from config import SimulationConfig
from main import _initialize_simulation
from solver import ElasticOperator


def discrete_energy(state, solver):
    operator = solver.operator
    current = operator.pack(state.u, state.v)
    previous = operator.pack(state.u_prev, state.v_prev)
    velocity = (current-previous)/solver.dt
    midpoint = (current+previous)/2
    energy = (velocity @ (operator.mass*velocity) + midpoint @ (operator.stiffness @ midpoint))/2
    if solver.config.time_integration == "explicit":
        energy -= solver.dt**2/8*(velocity @ (operator.stiffness @ velocity))
    return float(energy)


def energy_check(nu=.25, ratio=1., method="implicit", steps=1000):
    config = SimulationConfig(nx=9, ny=9, dx=.02, dy=.02*ratio, poisson_ratio=nu,
                              damping=1, source_amplitude=0, source_mode="interior",
                              cfl_safety_factor=.8, time_integration=method)
    state, solver = _initialize_simulation(config, "P")
    values = np.random.default_rng(17).normal(size=solver.operator.mass.size)*1e-4
    state.u, state.v = solver.operator.fields(values)
    state.u_prev, state.v_prev = state.u.copy(), state.v.copy()
    initial = discrete_energy(state, solver)
    largest_error = 0.
    for step in range(steps):
        solver.step(state, step*config.dt)
        largest_error = max(largest_error, abs(discrete_energy(state, solver)/initial-1))
    assert largest_error < 1e-8, largest_error
    return {"nu": nu, "dy_dx": ratio, "method": method, "steps": steps, "relative_energy_drift": largest_error}


def bulk_speeds(points):
    config = SimulationConfig(nx=points+1, ny=2, dx=1/points, dy=1)
    operator = ElasticOperator(config)
    sine = np.sin(2*np.pi*np.arange(points+1)/points)[None, None, :, None]
    field = np.broadcast_to(sine, (1, 2, points+1, 1))
    speeds = []
    for u, v in ((field, np.zeros_like(field)), (np.zeros_like(field), field)):
        values = operator.pack(u, v)
        speeds.append(float(np.sqrt(values @ (operator.stiffness @ values)/(values @ (operator.mass*values)))/(2*np.pi)))
    errors = [abs(speed/exact-1) for speed, exact in zip(speeds, (config.p_wave_speed, config.s_wave_speed))]
    return {"points": points, "p_speed": speeds[0], "s_speed": speeds[1], "relative_errors": errors}


def rayleigh_speed(points, depth=5., nu=.25):
    """Lowest two Fourier-reduced free-strip modes, compared with a half-space root."""
    config = SimulationConfig(nx=points+1, ny=round(depth*points)+1, dx=1/points, dy=1/points, poisson_ratio=nu)
    operator = ElasticOperator(config)
    beta = (config.s_wave_speed/config.p_wave_speed)**2
    root = brentq(lambda r: (2-r*r)**2-4*np.sqrt(1-r*r)*np.sqrt(1-beta*r*r), .1, .999999)
    exact = root*config.s_wave_speed
    # One horizontal Fourier mode imposes periodic x, not an extra production boundary mode.
    shape = operator.dofs.shape
    columns = np.broadcast_to(np.arange(2*config.ny).reshape(2, config.ny, 1), shape).ravel()
    phase = np.broadcast_to(np.exp(2j*np.pi*np.arange(config.nx)/points), shape).ravel()
    projection = csr_matrix((phase, (operator.dofs.ravel(), columns)), shape=(operator.mass.size, 2*config.ny))
    stiffness = (projection.conjugate().T @ operator.stiffness @ projection).toarray()
    mass = np.asarray(projection.conjugate().multiply(projection).real.T @ operator.mass).ravel()
    normalized = stiffness/np.sqrt(mass[:, None]*mass[None, :])
    eigenvalues, vectors = eigh(normalized, subset_by_index=(0, 1))
    speeds = np.sqrt(eigenvalues)/(2*np.pi)
    y = np.arange(config.ny)/points
    surface = np.tile((y < .5) | (y > depth-.5), 2)
    localization = np.sum(np.abs(vectors[surface])**2, axis=0)
    assert np.all(localization > .5), localization
    assert np.all((speeds > 0) & (speeds < config.s_wave_speed)), speeds
    return {"points": points, "depth_wavelengths": depth, "exact": exact, "speeds": speeds.tolist(),
                "relative_error": float(np.max(np.abs(speeds/exact-1))), "surface_mass_fraction": localization.tolist()}


def reflection_check(points, method="implicit"):
    """Normal P reflection at nu=0 has displacement coefficient +1 without lateral coupling."""
    config = SimulationConfig(nx=points+1, ny=3, dx=1/points, dy=.1, poisson_ratio=0,
                              damping=1, source_amplitude=0, source_mode="interior", time_integration=method)
    state, solver = _initialize_simulation(config, "P")
    x = np.arange(points+1)/points
    def packet(position):
        return np.exp(-((position-.3)/.04)**2)
    state.u = np.broadcast_to(packet(x)[None, None, :, None], state.u.shape).copy()
    state.u_prev = np.broadcast_to(packet(x+config.p_wave_speed*config.dt)[None, None, :, None], state.u.shape).copy()
    steps = int(np.ceil(1.3/config.p_wave_speed/config.dt))
    for step in range(steps):
        solver.step(state, step*config.dt)
    time = steps*config.dt
    exact = packet(x-config.p_wave_speed*time)+packet(2-x-config.p_wave_speed*time)
    observed = state.u[0, 1, :, 0]
    error = np.linalg.norm(observed-exact)/np.linalg.norm(exact)
    return {"points": points, "method": method, "steps": steps, "displacement_reflection_peak": float(observed.max()),
                "relative_profile_error": float(error), "transverse_amplitude": float(np.abs(state.v).max())}


def run():
    energy = [energy_check(nu, ratio, method, 10000 if nu == .25 and ratio == 1 else 1000)
              for nu, ratio in ((0., 1), (.25, 1), (.45, 1), (.49, 1), (.4999, 1), (.25, .5), (.25, 2))
              for method in ("explicit", "implicit")]
    bulk = [bulk_speeds(n) for n in (16, 32, 64)]
    surface = [rayleigh_speed(n) for n in (16, 32, 64)]
    incompressible = [rayleigh_speed(n, nu=.4999) for n in (16, 32, 64)]
    reflection = [reflection_check(n, method) for method in ("explicit", "implicit") for n in (200, 400, 800)]
    for i in (0, 1):
        assert bulk[i]["relative_errors"][0]/bulk[i+1]["relative_errors"][0] > 3.8
        assert surface[i]["relative_error"]/surface[i+1]["relative_error"] > 3
        assert incompressible[i]["relative_error"]/incompressible[i+1]["relative_error"] > 3
    for group in (reflection[:3], reflection[3:]):
        assert group[-1]["relative_profile_error"] < .01, group
        assert abs(group[-1]["displacement_reflection_peak"]-1) < .01, group
        assert group[0]["relative_profile_error"]/group[1]["relative_profile_error"] > 3, group
        assert group[1]["relative_profile_error"]/group[2]["relative_profile_error"] > 3, group
    return {"energy": energy, "bulk": bulk, "rayleigh": surface,
            "nearly_incompressible_rayleigh": incompressible, "normal_reflection": reflection}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
