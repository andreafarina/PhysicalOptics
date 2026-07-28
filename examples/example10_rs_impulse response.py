"""
Example 10 - RS impulse response.
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


from physical_optics.diffraction import rayleigh_sommerfeld

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
dz = 5e-6            # [m]

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
plt.show(block=False)
for i in range(20):
    ax1.clear()
    ax2.clear()
    z = dz * i
# -------------------------------------------------------------------------
# Display propagator impuse response
# -------------------------------------------------------------------------
    h = rayleigh_sommerfeld.impulse_response_goodman(grid, wavelength, z)
    show_surface(np.abs(h),grid,ax=ax1,domain='space',title='Amplitude')
    phase = np.unwrap(np.unwrap(np.angle(h), axis=0), axis=1)
    show_surface(phase, grid, ax=ax2, domain='space', title='Phase')

    plt.tight_layout()
    plt.pause(0.5)