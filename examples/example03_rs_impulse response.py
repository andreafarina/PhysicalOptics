"""
Example 03 - Rayleigh-Sommerfeld impulse response.
Plot of the amplitude of the RS impulse response as a function of the distance
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from physical_optics.common.grid import Grid

from physical_optics.diffraction import rayleigh_sommerfeld

from physical_optics.visualization.plot import (
    show_surface,
)

# -------------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------------

wavelength = 633e-9   # [m]
dz = wavelength#1e-6            # [m]

Nx = 512 * 4
Ny = 512 * 4

dx = 2e-6 / 2           # [m]
dy = 2e-6 / 2          # [m]

radius = 5e-6          # [m]


# -------------------------------------------------------------------------
# Input grid
# -------------------------------------------------------------------------

grid = Grid(Nx, Ny, dx, dy)

fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(1, 1, 1, projection="3d")
# ax2 = fig.add_subplot(1, 2, 2, projection="3d")
txt = fig.text(
        0.5,
        0.95,
        "",
        ha="center",
        va="top",
        fontsize=14,
    )
plt.show(block=False)
def update(i):
    ax1.clear()
    # ax2.clear()
    z = dz * (i+1)
    # -------------------------------------------------------------------------
    # Display propagator impulse response
    # -------------------------------------------------------------------------
    h = rayleigh_sommerfeld.impulse_response(grid, wavelength, z)
    show_surface(np.abs(h),grid,ax=ax1,domain='space',title='Amplitude',zoom = 16)
    # ax1.set_zlim(0, 2e11)
    # phase = np.unwrap(np.unwrap(np.angle(h), axis=0), axis=1)
    # show_surface(phase, grid, ax=ax2, domain='space', title='Phase', cmap = "Blues",zoom = 16)
    txt.set_text(rf"$z = {z / wavelength:.2f}\,\lambda$")
    plt.tight_layout()

ani = FuncAnimation(fig, update, frames=100, interval=5, repeat=True)
plt.show()