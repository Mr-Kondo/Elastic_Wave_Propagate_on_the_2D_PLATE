"""
GIF export functionality for elastic wave simulation animations.

Exports wave field animations as GIF files for P-wave, S-wave, and combined magnitude.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import imageio
import io
from PIL import Image
import mlx.core as mx

from config import SimulationConfig
from state import WaveFieldState
from solver import WaveSolver


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
        self.total_frames = config.gif_num_steps // config.gif_frame_interval
        self.frame_interval_ms = max(1, int(1000 / config.gif_frame_rate))

        # Frame storage
        self.p_wave_frames = []  # X-displacement frames
        self.s_wave_frames = []  # Y-displacement frames
        self.combined_frames = []  # Combined magnitude frames

        # Track value ranges for consistent scaling
        self.p_wave_min = 0.0
        self.p_wave_max = 1.0
        self.s_wave_min = 0.0
        self.s_wave_max = 1.0
        self.combined_min = 0.0
        self.combined_max = 1.0

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

        if p_abs_max > 0:
            if self.p_wave_max < p_abs_max:
                self.p_wave_max = p_abs_max
                self.p_wave_min = -p_abs_max

        if s_abs_max > 0:
            if self.s_wave_max < s_abs_max:
                self.s_wave_max = s_abs_max
                self.s_wave_min = -s_abs_max

        if combined_abs_max > 0:
            if self.combined_max < combined_abs_max:
                self.combined_max = combined_abs_max

    def _render_frame_with_scaling(
        self, field_data: np.ndarray, vmin: float, vmax: float, title: str, wave_type: str = "P"
    ) -> np.ndarray:
        """
        Render a frame with contour lines only (no colormap background).

        Args:
            field_data: 2D field data (ny x nx)
            vmin: Minimum value for scaling
            vmax: Maximum value for scaling
            title: Frame title
            wave_type: Wave type for title ('P', 'S', or 'Combined')

        Returns:
            RGB image array
        """
        import io
        from PIL import Image

        # Create a figure for rendering
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

        # Calculate physical extent
        x_extent = self.config.nx * self.config.dx
        y_extent = self.config.ny * self.config.dy

        # Create X, Y coordinate grids for contour plot
        x = np.linspace(0, x_extent, self.config.nx)
        y = np.linspace(0, y_extent, self.config.ny)
        X, Y = np.meshgrid(x, y)

        # Plot contour lines only (no background colormap)
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

        # Render to RGB array using PIL buffer
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100, facecolor="white")
        buf.seek(0)
        img = Image.open(buf)
        rgb_array = np.array(img.convert("RGB"))

        plt.close(fig)

        return rgb_array

    def _collect_frame_data(self) -> tuple:
        """
        Collect frame data for all three wave types.

        Returns:
            Tuple of (p_wave_field, s_wave_field, combined_field) as numpy arrays
        """
        # Evaluate MLX arrays
        mx.eval(self.state.u, self.state.v)

        # Get fields as numpy arrays
        u_np = np.array(self.state.u[0, :, :, 0])  # X-displacement
        v_np = np.array(self.state.v[0, :, :, 0])  # Y-displacement

        # Combined magnitude
        combined_np = np.sqrt(u_np**2 + v_np**2)

        return u_np, v_np, combined_np

    def run(self) -> None:
        """Run simulation and collect frames for all three wave types."""
        print(f"\nStarting GIF export simulation...")
        print(f"Output directory: {self.output_dir}")
        print(f"Total steps: {self.config.gif_num_steps}")
        print(f"Frame rate: {self.config.gif_frame_rate} fps")
        print(f"Expected frames: {self.total_frames}")
        print(f"Collecting frames for: P-wave, S-wave, Combined magnitude\n")

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
            self.p_wave_frames.append(p_field.copy())
            self.s_wave_frames.append(s_field.copy())
            self.combined_frames.append(combined_field.copy())

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
        """Render collected frames and save as GIF files."""
        p_wave_images = []
        s_wave_images = []
        combined_images = []

        print(f"  Rendering P-wave animation...")
        for i, p_field in enumerate(self.p_wave_frames):
            title = f"P-wave Displacement | Step: {(i + 1) * self.config.gif_frame_interval:4d}"
            img = self._render_frame_with_scaling(p_field, self.p_wave_min, self.p_wave_max, title, wave_type="P")
            p_wave_images.append(img)

            if (i + 1) % max(1, len(self.p_wave_frames) // 10) == 0:
                print(f"    Rendered {i + 1}/{len(self.p_wave_frames)} frames...")

        print(f"  Rendering S-wave animation...")
        for i, s_field in enumerate(self.s_wave_frames):
            title = f"S-wave Displacement | Step: {(i + 1) * self.config.gif_frame_interval:4d}"
            img = self._render_frame_with_scaling(s_field, self.s_wave_min, self.s_wave_max, title, wave_type="S")
            s_wave_images.append(img)

            if (i + 1) % max(1, len(self.s_wave_frames) // 10) == 0:
                print(f"    Rendered {i + 1}/{len(self.s_wave_frames)} frames...")

        print(f"  Rendering combined magnitude animation...")
        for i, combined_field in enumerate(self.combined_frames):
            title = f"Combined Magnitude | Step: {(i + 1) * self.config.gif_frame_interval:4d}"
            img = self._render_frame_with_scaling(
                combined_field, self.combined_min, self.combined_max, title, wave_type="Combined"
            )
            combined_images.append(img)

            if (i + 1) % max(1, len(self.combined_frames) // 10) == 0:
                print(f"    Rendered {i + 1}/{len(self.combined_frames)} frames...")

        # Save GIFs
        print(f"\n  Saving GIF files...")

        p_wave_path = self.output_dir / "p_wave_animation.gif"
        imageio.mimsave(p_wave_path, p_wave_images, duration=self.frame_interval_ms / 1000.0)
        print(f"    Saved: {p_wave_path}")

        s_wave_path = self.output_dir / "s_wave_animation.gif"
        imageio.mimsave(s_wave_path, s_wave_images, duration=self.frame_interval_ms / 1000.0)
        print(f"    Saved: {s_wave_path}")

        combined_path = self.output_dir / "combined_magnitude.gif"
        imageio.mimsave(combined_path, combined_images, duration=self.frame_interval_ms / 1000.0)
        print(f"    Saved: {combined_path}")

        print(f"\nAll GIF files saved to: {self.output_dir}")
