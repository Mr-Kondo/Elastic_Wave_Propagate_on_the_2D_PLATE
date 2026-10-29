"""Headless side-by-side comparison of several simulation cases.

Runs each case without a display and writes one video (mp4 via ffmpeg, else GIF)
that stacks the displacement-magnitude fields on a shared colour scale.
"""

import argparse
import shutil
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
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
    listing = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True, check=False)
    return "libx264" in listing.stdout


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


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {text}")
    return value


def main(argv=None) -> Path:
    parser = argparse.ArgumentParser(description="Compare simulation cases side by side (headless).")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="boundaries")
    parser.add_argument("--steps", type=_positive_int, default=560)
    parser.add_argument("--frame-interval", type=_positive_int, default=2)
    parser.add_argument("--fps", type=_positive_int, default=20)
    parser.add_argument("--out", type=Path, default=Path("feedback/compare"), help="output path; suffix is chosen automatically")
    args = parser.parse_args(argv)

    base = SimulationConfig()
    cases = {}
    for name, overrides in PRESETS[args.preset].items():
        start = time.perf_counter()
        print(f"[{name}] running {args.steps} steps ...", flush=True)
        cases[name] = run_case(replace(base, **overrides), args.steps, args.frame_interval)
        print(f"[{name}] done in {time.perf_counter() - start:.1f} s", flush=True)
    print("rendering video ...", flush=True)
    out = render(cases, base, args.steps, args.frame_interval, args.fps, args.out)
    print(f"wrote {out}", flush=True)
    return out
