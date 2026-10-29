"""Headless side-by-side comparison of several simulation cases.

Runs each case without a display and writes one video (mp4 via ffmpeg, else GIF)
that stacks the displacement-magnitude fields on a shared colour scale.
"""

import argparse
import os
import shutil
import subprocess
import time
from dataclasses import fields, replace
from pathlib import Path

# Force Agg: notebook kernels (Colab) export MPLBACKEND=module://matplotlib_inline..., which this venv lacks.
os.environ["MPLBACKEND"] = "Agg"
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from config import SimulationConfig
from main import _initialize_simulation

# Case name -> SimulationConfig overrides.
PRESETS = {
    "boundaries": {
        "stress_free_all": {"boundary_type": "stress_free_all"},
        "free_top": {"boundary_type": "free_top"},
        "fixed": {"boundary_type": "fixed"},
    },
    "poisson": {
        "nu=0.10": {"poisson_ratio": 0.10},
        "nu=0.25": {"poisson_ratio": 0.25},
        "nu=0.40": {"poisson_ratio": 0.40},
    },
    "integrators": {
        "implicit": {"time_integration": "implicit"},
        "explicit": {"time_integration": "explicit"},
    },
}


def run_case(config: SimulationConfig, steps: int, frame_interval: int) -> np.ndarray:
    """Step the solver and return displacement magnitude frames, shape (frames, ny, nx)."""
    state, solver = _initialize_simulation(config, "Both")
    frames = []
    for step in range(steps):
        solver.step(state, step * config.dt)
        if (step + 1) % frame_interval == 0 or step + 1 == steps:
            u = state.u[0, :, :, 0]
            v = state.v[0, :, :, 0]
            if not (np.isfinite(u).all() and np.isfinite(v).all()):
                raise RuntimeError(f"Non-finite displacement at step {step + 1}")
            frames.append(np.hypot(u, v))
    return np.stack(frames)


def _can_encode_mp4() -> bool:
    """True only if ffmpeg exists and was built with libx264 (else fall back to GIF)."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    try:
        listing = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True,
                                 check=False, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return listing.returncode == 0 and "libx264" in listing.stdout


def render(cases: dict, config: SimulationConfig, steps: int, frame_interval: int, fps: int, out: Path) -> Path:
    """Write the stacked comparison; returns the file written (.mp4, or .gif without ffmpeg)."""
    names = list(cases)
    vmax = max(float(f.max()) for f in cases.values()) or 1e-12  # shared scale makes cases comparable
    extent = (0, (config.nx - 1) * config.dx, 0, (config.ny - 1) * config.dy)
    fig, axes = plt.subplots(len(names), 1, figsize=(8, 2.4 * len(names)), dpi=100, squeeze=False,
                             constrained_layout=True)
    images = []
    for ax, name in zip(axes[:, 0], names):
        images.append(ax.imshow(cases[name][0], origin="lower", extent=extent, vmin=0, vmax=vmax, cmap="viridis"))
        ax.set_title(name, fontsize=10)
        ax.set_ylabel("y (m)")
    axes[-1, 0].set_xlabel("x (m)")
    fig.colorbar(images[0], ax=axes[:, 0].tolist(), label="|u| (m)")
    label = fig.suptitle("")
    fig.canvas.draw()  # solve the layout once, then freeze it: re-solving every frame took half the render time
    fig.set_layout_engine("none")

    def update(i):
        for im, name in zip(images, names):
            im.set_data(cases[name][i])
        label.set_text(f"step {min((i + 1) * frame_interval, steps)}")
        return images

    nframes = len(next(iter(cases.values())))
    anim = animation.FuncAnimation(fig, update, frames=nframes)
    if _can_encode_mp4():
        out = out.with_suffix(".mp4")
        writer = animation.FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    else:
        out = out.with_suffix(".gif")
        writer = animation.PillowWriter(fps=fps)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=writer)
    plt.close(fig)
    return out


def parse_case(text: str, base: SimulationConfig) -> tuple[str, dict]:
    """Parse 'name:key=value,key=value' into (name, overrides) with values typed like the base config."""
    name, _, pairs = text.partition(":")
    known = {f.name for f in fields(base)}
    overrides = {}
    for pair in filter(None, pairs.split(",")):
        key, eq, raw = pair.partition("=")
        if not name or not eq or key not in known:
            raise argparse.ArgumentTypeError(f"bad --case {text!r}: expected name:key=value with a SimulationConfig field")
        current = getattr(base, key)
        try:
            overrides[key] = current if isinstance(current, bool) and raw == str(current) else type(current)(raw)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"bad value for {key} in {text!r}: {exc}") from None
    return name, overrides


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {text}")
    return value


def main(argv=None) -> Path:
    parser = argparse.ArgumentParser(description="Compare simulation cases side by side (headless).")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="boundaries")
    parser.add_argument("--case", action="append", default=[], metavar="NAME:KEY=VALUE,...",
                        help="custom case (repeatable); replaces --preset, e.g. soft:poisson_ratio=0.4,boundary_type=fixed")
    parser.add_argument("--steps", type=_positive_int, default=560)
    parser.add_argument("--frame-interval", type=_positive_int, default=2)
    parser.add_argument("--fps", type=_positive_int, default=20)
    parser.add_argument("--out", type=Path, default=Path("feedback/compare"), help="output path; suffix is chosen automatically")
    args = parser.parse_args(argv)

    base = SimulationConfig()
    try:
        custom = dict(parse_case(text, base) for text in args.case)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    cases = {}
    for name, overrides in (custom or PRESETS[args.preset]).items():
        start = time.perf_counter()
        print(f"[{name}] running {args.steps} steps ...", flush=True)
        cases[name] = run_case(replace(base, **overrides), args.steps, args.frame_interval)
        print(f"[{name}] done in {time.perf_counter() - start:.1f} s", flush=True)
    print("rendering video ...", flush=True)
    out = render(cases, base, args.steps, args.frame_interval, args.fps, args.out)
    print(f"wrote {out}", flush=True)
    return out
