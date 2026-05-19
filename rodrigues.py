"""
Rodrigues Rotation Formula — Interactive 3D Visualizer
======================================================

Demonstrates the Rodrigues rotation formula:
  v_rot = v*cos(θ) + (k × v)*sin(θ) + k*(k·v)*(1 - cos(θ))

Where:
  v     — original vector
  k     — unit rotation axis
  θ     — rotation angle
  v_rot — rotated vector

Interactive controls:
  - Slider: adjust rotation angle θ from -π to +π
  - Mouse drag on the 3D axes: rotate the viewing angle
  - Text boxes: set rotation axis (kx, ky, kz) and vector (vx, vy, vz)

Visualization:
  - Blue arrow: rotation axis k (unit vector, extended for visibility)
  - Green arrow: original vector v₀
  - Red arrow: rotated vector v_rot
  - Purple translucent disk: rotation plane (perpendicular to k, at the
    component of v perpendicular to k)
  - Dashed arc: shows the rotation path from v₀ to v_rot on the plane
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
from mpl_toolkits.mplot3d import art3d
import matplotlib.patches as mpatches


def rodrigues_rotate(v, k, theta):
    """Apply Rodrigues rotation: rotate v around unit axis k by angle theta."""
    k = k / np.linalg.norm(k)
    v_rot = (v * np.cos(theta)
             + np.cross(k, v) * np.sin(theta)
             + k * np.dot(k, v) * (1 - np.cos(theta)))
    return v_rot


def create_rotation_arc(v, k, theta, n_points=60):
    """Generate points along the rotation arc from v to v_rot."""
    k = k / np.linalg.norm(k)
    angles = np.linspace(0, theta, n_points)
    arc = np.array([rodrigues_rotate(v, k, a) for a in angles])
    return arc


def create_rotation_disk(v, k, center, radius, n_points=50):
    """Create a disk on the rotation plane for visualization."""
    k = k / np.linalg.norm(k)
    # Find two orthogonal vectors in the plane perpendicular to k
    if abs(k[0]) < 0.9:
        u = np.cross(k, np.array([1, 0, 0]))
    else:
        u = np.cross(k, np.array([0, 1, 0]))
    u = u / np.linalg.norm(u)
    w = np.cross(k, u)

    theta_range = np.linspace(0, 2 * np.pi, n_points)
    r_range = np.linspace(0, radius, 10)
    points = []
    for r in r_range:
        for t in theta_range:
            p = center + r * (np.cos(t) * u + np.sin(t) * w)
            points.append(p)
    return np.array(points)


class RodriguesVisualizer:
    def __init__(self):
        self.v = np.array([1.0, 0.0, 0.0])
        self.k = np.array([0.0, 0.0, 1.0])
        self.theta = np.pi / 3

        self.fig = plt.figure(figsize=(12, 9))
        self.fig.patch.set_facecolor('#1a1a2e')

        # 3D axes (leave room for sliders at bottom)
        self.ax = self.fig.add_axes([0.05, 0.25, 0.9, 0.72], projection='3d')
        self.ax.set_facecolor('#16213e')

        self._setup_widgets()
        self._draw()

    def _setup_widgets(self):
        # Angle slider
        ax_theta = self.fig.add_axes([0.15, 0.12, 0.55, 0.03])
        self.slider_theta = Slider(
            ax_theta, 'θ (rad)', -np.pi, np.pi,
            valinit=self.theta, valstep=0.01,
            color='#e94560'
        )
        self.slider_theta.on_changed(self._on_theta_change)

        # Axis input boxes
        ax_kx = self.fig.add_axes([0.15, 0.05, 0.08, 0.04])
        ax_ky = self.fig.add_axes([0.30, 0.05, 0.08, 0.04])
        ax_kz = self.fig.add_axes([0.45, 0.05, 0.08, 0.04])
        self.tb_kx = TextBox(ax_kx, 'kx ', initial=f'{self.k[0]:.2f}')
        self.tb_ky = TextBox(ax_ky, 'ky ', initial=f'{self.k[1]:.2f}')
        self.tb_kz = TextBox(ax_kz, 'kz ', initial=f'{self.k[2]:.2f}')
        self.tb_kx.on_submit(self._on_axis_change)
        self.tb_ky.on_submit(self._on_axis_change)
        self.tb_kz.on_submit(self._on_axis_change)

        # Vector input boxes
        ax_vx = self.fig.add_axes([0.63, 0.05, 0.08, 0.04])
        ax_vy = self.fig.add_axes([0.78, 0.05, 0.08, 0.04])
        ax_vz = self.fig.add_axes([0.90, 0.05, 0.08, 0.04])
        self.tb_vx = TextBox(ax_vx, 'vx ', initial=f'{self.v[0]:.2f}')
        self.tb_vy = TextBox(ax_vy, 'vy ', initial=f'{self.v[1]:.2f}')
        self.tb_vz = TextBox(ax_vz, 'vz ', initial=f'{self.v[2]:.2f}')
        self.tb_vx.on_submit(self._on_vector_change)
        self.tb_vy.on_submit(self._on_vector_change)
        self.tb_vz.on_submit(self._on_vector_change)

    def _on_theta_change(self, val):
        self.theta = val
        self._draw()

    def _on_axis_change(self, _text):
        try:
            kx = float(self.tb_kx.text)
            ky = float(self.tb_ky.text)
            kz = float(self.tb_kz.text)
            k_new = np.array([kx, ky, kz])
            if np.linalg.norm(k_new) > 1e-8:
                self.k = k_new
                self._draw()
        except ValueError:
            pass

    def _on_vector_change(self, _text):
        try:
            vx = float(self.tb_vx.text)
            vy = float(self.tb_vy.text)
            vz = float(self.tb_vz.text)
            self.v = np.array([vx, vy, vz])
            self._draw()
        except ValueError:
            pass

    def _draw(self):
        ax = self.ax
        # Preserve current view angle
        elev, azim = ax.elev, ax.azim

        ax.cla()
        ax.set_facecolor('#16213e')

        k_unit = self.k / np.linalg.norm(self.k)
        v_rot = rodrigues_rotate(self.v, self.k, self.theta)

        # Determine plot limits
        all_pts = np.array([self.v, v_rot, k_unit * 1.5, -k_unit * 0.5])
        max_range = max(np.abs(all_pts).max(), 1.2)
        lim = max_range * 1.1
        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])
        ax.set_zlim([-lim, lim])

        # Draw coordinate axes (thin gray)
        axis_len = lim * 0.9
        for i, (color, label) in enumerate(zip(
                ['#555555', '#555555', '#555555'], ['X', 'Y', 'Z'])):
            direction = np.zeros(3)
            direction[i] = axis_len
            ax.plot([0, direction[0]], [0, direction[1]], [0, direction[2]],
                    color=color, linewidth=0.8, linestyle='--', alpha=0.5)
            ax.text(direction[0]*1.05, direction[1]*1.05, direction[2]*1.05,
                    label, color='#888888', fontsize=10)

        # Rotation axis k (blue, extended line)
        k_ext = 1.5
        ax.quiver(0, 0, 0, k_unit[0]*k_ext, k_unit[1]*k_ext, k_unit[2]*k_ext,
                  color='#4fc3f7', arrow_length_ratio=0.08, linewidth=2.5,
                  label=f'k = [{k_unit[0]:.2f}, {k_unit[1]:.2f}, {k_unit[2]:.2f}]')
        # Extend axis in both directions
        ax.plot([-k_unit[0]*0.5, k_unit[0]*k_ext],
                [-k_unit[1]*0.5, k_unit[1]*k_ext],
                [-k_unit[2]*0.5, k_unit[2]*k_ext],
                color='#4fc3f7', linewidth=1.0, alpha=0.4, linestyle=':')

        # Original vector v₀ (green)
        ax.quiver(0, 0, 0, self.v[0], self.v[1], self.v[2],
                  color='#66bb6a', arrow_length_ratio=0.1, linewidth=2.5,
                  label=f'v₀ = [{self.v[0]:.2f}, {self.v[1]:.2f}, {self.v[2]:.2f}]')

        # Rotated vector v_rot (red)
        ax.quiver(0, 0, 0, v_rot[0], v_rot[1], v_rot[2],
                  color='#ef5350', arrow_length_ratio=0.1, linewidth=2.5,
                  label=f'v_rot = [{v_rot[0]:.2f}, {v_rot[1]:.2f}, {v_rot[2]:.2f}]')

        # Rotation plane / disk visualization
        # The rotation happens in the plane perpendicular to k
        # Project v onto the plane perpendicular to k
        v_parallel = np.dot(self.v, k_unit) * k_unit
        v_perp = self.v - v_parallel
        v_perp_mag = np.linalg.norm(v_perp)

        if v_perp_mag > 1e-8:
            # Draw rotation arc
            arc = create_rotation_arc(self.v, self.k, self.theta)
            ax.plot(arc[:, 0], arc[:, 1], arc[:, 2],
                    color='#ffab40', linewidth=2.0, linestyle='-', alpha=0.8)

            # Draw the rotation plane (translucent disk)
            disk_center = v_parallel
            n_circ = 80
            angles = np.linspace(0, 2 * np.pi, n_circ)

            # Construct orthonormal basis on the plane
            e1 = v_perp / v_perp_mag
            e2 = np.cross(k_unit, e1)
            radius = v_perp_mag

            # Create disk as a polygon collection
            circle_pts = np.array([
                disk_center + radius * (np.cos(a) * e1 + np.sin(a) * e2)
                for a in angles
            ])
            # Draw filled disk
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            verts = [circle_pts.tolist()]
            disk_poly = Poly3DCollection(verts, alpha=0.12, facecolor='#ab47bc',
                                         edgecolor='#ab47bc', linewidth=0.5)
            ax.add_collection3d(disk_poly)

            # Draw circle outline
            ax.plot(circle_pts[:, 0], circle_pts[:, 1], circle_pts[:, 2],
                    color='#ab47bc', linewidth=1.0, alpha=0.5)

            # Mark the parallel component (projection onto axis)
            if np.linalg.norm(v_parallel) > 1e-8:
                ax.plot([0, v_parallel[0]], [0, v_parallel[1]], [0, v_parallel[2]],
                        color='#4fc3f7', linewidth=1.0, alpha=0.4, linestyle='--')
                ax.plot([v_parallel[0], self.v[0]],
                        [v_parallel[1], self.v[1]],
                        [v_parallel[2], self.v[2]],
                        color='#66bb6a', linewidth=1.0, alpha=0.4, linestyle='--')

        # Title with formula
        ax.set_title(
            r"$\mathbf{v}_{rot} = \mathbf{v}\cos\theta + "
            r"(\mathbf{k}\times\mathbf{v})\sin\theta + "
            r"\mathbf{k}(\mathbf{k}\cdot\mathbf{v})(1-\cos\theta)$"
            f"\n θ = {self.theta:.3f} rad = {np.degrees(self.theta):.1f}°",
            color='white', fontsize=12, pad=10
        )

        ax.set_xlabel('X', color='#888888')
        ax.set_ylabel('Y', color='#888888')
        ax.set_zlabel('Z', color='#888888')
        ax.tick_params(colors='#666666')

        # Legend
        ax.legend(loc='upper left', fontsize=9, framealpha=0.7,
                  facecolor='#1a1a2e', edgecolor='#333333', labelcolor='white')

        # Restore view angle
        ax.view_init(elev=elev, azim=azim)

        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


if __name__ == "__main__":
    viz = RodriguesVisualizer()
    viz.show()
