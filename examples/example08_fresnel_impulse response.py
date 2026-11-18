"""
Example 08 - Fresnel impulse response.
The phase for short distance is unstable, aliasing is present because of the denominator of z.
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from physical_optics.common.grid import Grid

from physical_optics.diffraction import fresnel

from physical_optics.visualization.plot import (
    show_surface,
)

# -------------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------------

wavelength = 633e-9   # [m]
dz = 1e-5            # [m]

Nx = 512 * 4
Ny = 512 * 4

dx = 1e-6/10           # [m]
dy = 1e-6/10          # [m]

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
    z = dz * (i+10)
    # -------------------------------------------------------------------------
    # Display propagator impulse response
    # -------------------------------------------------------------------------
    h = fresnel.impulse_response(grid, wavelength, z)
    # show_surface(np.abs(h),grid,ax=ax1,domain='space',title='Amplitude',zoom = 16)

    phase = np.unwrap(np.unwrap(np.angle(h), axis=0), axis=1)
    show_surface(phase, grid, ax=ax1, domain='space', title='Phase', cmap = "Blues",zoom = 1)
    ax1.set_zlim(-1000, 1)
    txt.set_text(rf"$z = {z / wavelength:.2f}\,\lambda$")
    plt.tight_layout()

ani = FuncAnimation(fig, update, frames=100, interval=5, repeat=True)
plt.show()
