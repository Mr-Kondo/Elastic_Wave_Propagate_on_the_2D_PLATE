"""
GIF export functionality for elastic wave simulation animations.

Exports x-displacement, y-displacement and magnitude; legacy filenames are retained.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from config import SimulationConfig
from solver import WaveSolver
from state import WaveFieldState


class GifExporter:
    """Exports wave field simulations as animated GIF files."""

    def __init__(self, config: SimulationConfig, state: WaveFieldState, solver: WaveSolver):
        """
        Initialize GIF exporter.

        Args:
            config: Simulation configuration
            state: Wave field state
            solver: Wave solver
        """
        self.config = config
        self.state = state
        self.solver = solver
        self.current_step = 0

        # Create output directory if it doesn't exist
        self.output_dir = Path(config.gif_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Calculate frame parameters
        self.total_frames = (config.gif_num_steps + config.gif_frame_interval - 1) // config.gif_frame_interval
        self.frame_interval_ms = max(1, int(1000 / config.gif_frame_rate))

        # Frame storage
        self.p_wave_frames = []  # X-displacement frames
        self.s_wave_frames = []  # Y-displacement frames
        self.combined_frames = []  # Combined magnitude frames

        # Track value ranges for consistent scaling
        self.p_wave_min = 0.0
        self.p_wave_max = 0.0
        self.s_wave_min = 0.0
        self.s_wave_max = 0.0
        self.combined_min = 0.0
        self.combined_max = 0.0

    def _update_field_ranges(self, p_field: np.ndarray, s_field: np.ndarray, combined_field: np.ndarray) -> None:
        """
        Update min/max values for consistent scaling across all frames.

        Args:
            p_field: P-wave field (X-displacement)
            s_field: S-wave field (Y-displacement)
            combined_field: Combined magnitude field
        """
        p_abs_max = np.abs(p_field).max()
        s_abs_max = np.abs(s_field).max()
        combined_abs_max = np.abs(combined_field).max()

        self.p_wave_max = max(self.p_wave_max, p_abs_max)
        self.p_wave_min = -self.p_wave_max
        self.s_wave_max = max(self.s_wave_max, s_abs_max)
        self.s_wave_min = -self.s_wave_max
        self.combined_max = max(self.combined_max, combined_abs_max)

    def _render_frame_with_scaling(
        self, field_data: np.ndarray, vmin: float, vmax: float, title: str
    ) -> np.ndarray:
        """
        Render a frame with contour lines only (no colormap background).

        Args:
            field_data: 2D field data (ny x nx)
            vmin: Minimum value for scaling
            vmax: Maximum value for scaling
            title: Frame title

        Returns:
            RGB image array
        """
        # Create a figure for rendering
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

        # Calculate physical extent
        x_extent = (self.config.nx - 1) * self.config.dx
        y_extent = (self.config.ny - 1) * self.config.dy

        # Create X, Y coordinate grids for contour plot
        x = np.linspace(0, x_extent, self.config.nx)
        y = np.linspace(0, y_extent, self.config.ny)
        X, Y = np.meshgrid(x, y)

        # Plot contour lines only (no background colormap)
        if vmax <= vmin:
            vmax = vmin + 1e-12
        levels = np.linspace(vmin, vmax, 15)  # 15 contour levels
        contours = ax.contour(X, Y, field_data, levels=levels, colors="black", linewidths=1.0, alpha=0.7)

        # Add contour labels
        ax.clabel(contours, inline=True, fontsize=8, fmt="%.2e")

        ax.set_xlabel("X direction (m)")
        ax.set_ylabel("Y direction (m)")
        ax.set_title(title)
        ax.set_xlim(0, x_extent)
        ax.set_ylim(0, y_extent)
        ax.set_aspect("equal")

        fig.canvas.draw()
        rgb_array = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)

        return rgb_array

    def _collect_frame_data(self) -> tuple:
        """
        Collect frame data for all three wave types.

        Returns:
            Tuple of (p_wave_field, s_wave_field, combined_field) as numpy arrays
        """
        # Get fields as numpy arrays
        u_np = np.array(self.state.u[0, :, :, 0])  # X-displacement
        v_np = np.array(self.state.v[0, :, :, 0])  # Y-displacement

        if not np.isfinite(u_np).all() or not np.isfinite(v_np).all():
            raise RuntimeError(f"Non-finite displacement at step {self.current_step}")

        # Combined magnitude
        combined_np = np.hypot(u_np, v_np)

        return u_np, v_np, combined_np

    def run(self) -> None:
        """Run simulation and collect frames for all three wave types."""
        print("\nStarting GIF export simulation...")
        print(f"Output directory: {self.output_dir}")
        print(f"Total steps: {self.config.gif_num_steps}")
        print(f"Frame rate: {self.config.gif_frame_rate} fps")
        print(f"Expected frames: {self.total_frames}")
        print("Collecting frames for: X-displacement, Y-displacement, magnitude\n")

        # First pass: collect frames and determine scaling
        print("Pass 1: Collecting frames and determining scaling...")
        self.current_step = 0
        frame_count = 0

        while self.current_step < self.config.gif_num_steps:
            # Run multiple solver steps per frame
            for _ in range(self.config.gif_frame_interval):
                t = self.current_step * self.config.dt
                self.solver.step(self.state, t)
                self.current_step += 1

                if self.current_step >= self.config.gif_num_steps:
                    break

            # Collect frame data
            p_field, s_field, combined_field = self._collect_frame_data()

            # Update field ranges
            self._update_field_ranges(p_field, s_field, combined_field)

            # Store frame data
            self.p_wave_frames.append(p_field)
            self.s_wave_frames.append(s_field)
            self.combined_frames.append(combined_field)

            frame_count += 1

            # Progress indicator
            if frame_count % max(1, self.total_frames // 10) == 0:
                print(f"  Collected {frame_count}/{self.total_frames} frames...")

        print(f"Collected {frame_count} total frames")
        print(f"Field ranges - P: [{self.p_wave_min:.3e}, {self.p_wave_max:.3e}]")
        print(f"             - S: [{self.s_wave_min:.3e}, {self.s_wave_max:.3e}]")
        print(f"             - Combined: [{self.combined_min:.3e}, {self.combined_max:.3e}]\n")

        # Second pass: render and save GIFs
        print("Pass 2: Rendering and saving GIF files...")
        self._render_and_save_gifs()
        print("GIF export complete!")

    def _render_and_save_gifs(self) -> None:
        """Render one component at a time; Pillow's GIF duration is in milliseconds."""
        for fields, vmin, vmax, filename, label in (
            (self.p_wave_frames, self.p_wave_min, self.p_wave_max, "p_wave_animation.gif", "X-displacement"),
            (self.s_wave_frames, self.s_wave_min, self.s_wave_max, "s_wave_animation.gif", "Y-displacement"),
            (self.combined_frames, self.combined_min, self.combined_max, "combined_magnitude.gif", "Displacement magnitude"),
        ):
            images = []
            for i, field in enumerate(fields):
                step = min((i + 1) * self.config.gif_frame_interval, self.config.gif_num_steps)
                frame = self._render_frame_with_scaling(field, vmin, vmax, f"{label} | Step: {step:4d}")
                images.append(Image.fromarray(frame))
            images[0].save(self.output_dir / filename, save_all=True, append_images=images[1:],
                           duration=self.frame_interval_ms, loop=0)
            for frame in images:
                frame.close()
            del images
            fields.clear()
