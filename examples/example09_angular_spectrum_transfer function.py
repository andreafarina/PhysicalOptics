"""
Example 09 - Angular spectrum transfer function.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from physical_optics.common.grid import Grid
from physical_optics.common.field import Field

from physical_optics.objects.apertures import (
    circular_aperture,
    rectangular_aperture
)


from physical_optics.diffraction import angular_spectrum

from physical_optics.visualization.plot import (
    show_intensity,
    show_phase,
    show_amplitude,
    show_image,
    show_surface,
)

# -------------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------------

wavelength = 633e-9   # [m]
dz = 1e-7         # [m]

Nx = 512 * 4
Ny = 512 * 4

dx = 2e-6 / 8           # [m]
dy = 2e-6 / 8          # [m]

radius = 50e-6          # [m]


# -------------------------------------------------------------------------
# Input grid
# -------------------------------------------------------------------------

grid = Grid(Nx, Ny, dx, dy)

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax2 = fig.add_subplot(1, 2, 2, projection="3d")
txt = fig.text(
        0.5,
        0.95,
        "",
        ha="center",
        va="top",
        fontsize=14,
    )
plt.ion()
for i in range(20):
    ax1.clear()
    ax2.clear()
    z = dz * i
# -------------------------------------------------------------------------
# Display propagator transfer function
# -------------------------------------------------------------------------
    H = angular_spectrum.transfer_function(grid, wavelength, z)
    show_surface(np.abs(H),grid,ax=ax1,domain='frequency',title='Amplitude')
    phase = np.unwrap(np.unwrap(np.angle(H), axis=0), axis=1)
    show_surface(phase, grid, ax=ax2, domain='frequency', title='Phase')
    ax2.set_zlim(0,5)

    txt.set_text(rf"$z = {z / wavelength:.2f}\,\lambda$")

    plt.tight_layout()
    plt.pause(0.5)