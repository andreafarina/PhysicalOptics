"""
Example 03 - Fraunhofer diffraction from a circular aperture.
"""

import matplotlib.pyplot as plt

from physical_optics.common.grid import Grid
from physical_optics.common.field import Field

from physical_optics.objects.apertures import (
    circular_aperture,
    rectangular_aperture,
    slit)

from physical_optics.diffraction import fraunhofer

from physical_optics.visualization.plot import (
    show_intensity, show_spectrum, show_phase, show_amplitude,
)


# -------------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------------

wavelength = 633e-9         # [m]
z = 1.0                  # [m]

Nx = 512 * 8
Ny = 512 * 8

dx = 5e-6                   # [m]
dy = 5e-6                    # [m]

radius = 500e-6             # [m]


# -------------------------------------------------------------------------
# Input field
# -------------------------------------------------------------------------

grid = Grid(Nx, Ny, dx, dy)

field = Field(grid, wavelength)

field.U *= circular_aperture(grid, radius)
# field.U*= rectangular_aperture(
#     grid,
#     width=400e-6,
#     height=800e-6*3,
#     x0 = 0,
#     y0 = 0)
# -------------------------------------------------------------------------
# Fraunhofer propagation
# -------------------------------------------------------------------------

field_far = fraunhofer.propagate(field, z)


# -------------------------------------------------------------------------
# Display
# -------------------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

show_amplitude(field,ax = axs[0],title = 'Object amplitude')
show_phase(field_far,ax = axs[1])
show_intensity(field_far,ax = axs[2],log=True)
plt.tight_layout()
plt.show()