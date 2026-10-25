"""
Visualization and animation for elastic wave simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mlx.core as mx

from config import SimulationConfig
from state import WaveFieldState
from solver import WaveSolver


class WaveVisualizer:
    """Handles visualization and animation of wave fields."""

    def __init__(self, config: SimulationConfig, state: WaveFieldState, solver: WaveSolver, wave_type: str = None):
        """
        Initialize visualizer.

        Args:
            config: Simulation configuration
            state: Wave field state
            solver: Wave solver
            wave_type: Optional wave type label for display
        """
        self.config = config
        self.state = state
        self.solver = solver
        self.wave_type_label = wave_type

        # Animation reference (prevent garbage collection)
        self._anim = None

        # Time step counter
        self.current_step = 0

        # Early stopping variables
        self._small_disp_count = 0  # Count of consecutive frames with negligible displacement
        self._min_steps_before_check = 100  # Minimum steps before checking for early stop
        self._small_disp_threshold = 1e-9  # Threshold for negligible displacement
        self._consecutive_frames_needed = 5  # Consecutive frames needed to trigger stop

        # Setup plot
        self._setup_plot()

    def _setup_plot(self):
        """Initialize matplotlib figure and axes."""
        self.fig, self.ax = plt.subplots(figsize=(12, 8))

        # Initialize with zeros
        initial_data = np.zeros((self.config.ny, self.config.nx))

        # Calculate physical extent for correct aspect ratio
        # extent = [left, right, bottom, top] in data coordinates
        x_extent = (self.config.nx - 1) * self.config.dx
        y_extent = (self.config.ny - 1) * self.config.dy
        extent = [0, x_extent, 0, y_extent]

        # Initialize with dynamic scaling (will be updated each frame)
        self.im = self.ax.imshow(
            initial_data,
            cmap="seismic",
            vmin=-1e-6,  # Small initial range, will scale dynamically
            vmax=1e-6,
            origin="lower",
            extent=extent,
            aspect="auto",
        )

        # Colorbar removed - display area maximized for wave visualization
        self.title = self.ax.set_title("Elastic Wave Simulation (MLX) - Press 'p' for X, 's' for Y displacement")

        # Set axis labels with physical units
        self.ax.set_xlabel("X direction (m)")
        self.ax.set_ylabel("Y direction (m)")

        # Connect keyboard events
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _animate(self, frame: int):
        """
        Animation callback for each frame.

        Args:
            frame: Frame number

        Returns:
            List of artists to redraw, or empty list to terminate animation
        """
        # Check if simulation is complete
        if self.current_step >= self.config.steps:
            print("Simulation complete - animation stopped")
            if self._anim is not None:
                self._anim.event_source.stop()
            return [self.im]  # Return last frame to keep display

        max_disp_check = self.state.get_max_displacement()
        if not np.isfinite(max_disp_check):
            raise RuntimeError(f"Non-finite displacement at step {self.current_step}")
        if (self.current_step >= self._min_steps_before_check
                and self.current_step * self.config.dt >= self.config.source_duration):
            self._small_disp_count = self._small_disp_count + 1 if max_disp_check < self._small_disp_threshold else 0
            if self._small_disp_count >= self._consecutive_frames_needed:
                print(f"Wave negligible at step {self.current_step} - animation stopped")
                if self._anim is not None:
                    self._anim.event_source.stop()
                return [self.im]

        # Run multiple time steps per frame
        for _ in range(min(self.config.plot_interval, self.config.steps - self.current_step)):
            t = self.current_step * self.config.dt

            # Advance simulation
            self.solver.step(self.state, t)

            if not np.isfinite(self.state.get_max_displacement()):
                raise RuntimeError(f"Non-finite displacement at step {self.current_step + 1}")

            # Finite large amplitudes are a diagnostic, not proof of instability.
            if self.solver.check_stability(self.state):
                print(f"Warning: Large displacement detected at step {self.current_step} (continuing simulation)")

            self.current_step += 1

        # Evaluate MLX arrays
        mx.eval(self.state.u, self.state.v)

        # Get display field based on wave type
        disp_field = self.state.get_display_field()
        disp_np = np.array(disp_field[0, :, :, 0])

        # Use colorbar limits based on configuration
        if self.config.colorbar_mode == "dynamic":
            # Dynamic scaling: use actual max displacement each frame
            # Use percentile to avoid saturation from noise
            max_val = np.abs(disp_np).max()
            if max_val < 1e-8:
                margin = 1e-7
            else:
                # Increase margin from 0.15 to 0.4 for better input wave visualization
                # This emphasizes wave arrivals and gradients
                margin = max_val * 0.4
            fixed_vmax = max_val + margin
            fixed_vmin = -fixed_vmax
        else:
            # Fixed scaling: based on source amplitude (default)
            fixed_vmax = max(abs(self.config.source_amplitude) * 1.15, 1e-12)
            fixed_vmin = -fixed_vmax
        self.im.set_clim(vmin=fixed_vmin, vmax=fixed_vmax)

        # Add contour lines to visualize wave structure (wave crests/troughs)
        # Clear previous contours
        for coll in self.ax.collections:
            coll.remove()

        # Plot contour lines at key levels to visualize wave crests/troughs
        levels = np.linspace(fixed_vmin, fixed_vmax, 31)  # 31 contour levels (detailed wave structure)
        self.ax.contour(
            np.arange(self.config.nx) * self.config.dx,
            np.arange(self.config.ny) * self.config.dy,
            disp_np, levels=levels, colors="gray", alpha=0.3, linewidths=0.5,
        )

        # Update image data
        self.im.set_data(disp_np)

        # Update title with time and wave status
        max_disp = self.state.get_max_displacement()

        # Always use display_mode for current view (supports both fixed and interactive modes)
        source_label = "boundary input" if self.config.source_mode == "boundary" else "internal source"
        if self.state.display_mode == "P":
            wave_label = f"X-displacement ({source_label})"
        elif self.state.display_mode == "S":
            wave_label = "Y-displacement"
        else:
            wave_label = "Displacement magnitude"

        t_current = self.current_step * self.config.dt
        stability_status = " [CHECK AMPLITUDE/FINITE VALUES]" if self.solver.check_stability(self.state) else ""
        self.title.set_text(
            f"{wave_label} | Step: {self.current_step:4d} | Time: {t_current * 1e3:6.2f} ms | Max displacement: {max_disp:.3e} m{stability_status}"
        )

        return [self.im]

    def _on_key(self, event):
        """
        Keyboard event handler - switches display mode only (not input wave).
        Input is always P-wave; S-wave and combined modes are from mode conversion.

        Args:
            event: Keyboard event
        """
        if event.key == "p":
            if self.state.display_mode != "P":
                self.state.set_display_mode("P")
                print("Display switched to X-displacement")
        elif event.key == "s":
            if self.state.display_mode != "S":
                self.state.set_display_mode("S")
                print("Display switched to Y-displacement")
        elif event.key == "b":
            if self.state.display_mode != "Both":
                self.state.set_display_mode("Both")
                print("Display switched to combined (magnitude of P+S)")

    def run(self):
        """Run the animation."""
        print("Starting simulation...")

        total_frames = (self.config.steps + self.config.plot_interval - 1) // self.config.plot_interval
        interval_ms = max(1, int(1000 * self.config.playback_seconds / max(1, total_frames)))

        self._anim = FuncAnimation(
            self.fig,
            self._animate,
            frames=total_frames,
            interval=interval_ms,
            cache_frame_data=False,
            blit=False,
            repeat=False,  # Stop after all frames are complete
        )

        plt.show()
