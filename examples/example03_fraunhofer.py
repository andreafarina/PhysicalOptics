"""
Example 03 - Fraunhofer diffraction from a circular aperture.
"""

import matplotlib.pyplot as plt
import numpy as np

from physical_optics.common.grid import Grid
from physical_optics.common.field import Field

from physical_optics.objects.apertures import (
    circular_aperture,
    rectangular_aperture
)

from physical_optics.diffraction import fraunhofer

from physical_optics.visualization.plot import (
    show_intensity, show_spectrum, show_phase, show_amplitude,
)


# -------------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------------

wavelength = 633e-9         # [m]
z = 2.0                  # [m]

Nx = 512
Ny = 512

dx = 2e-6                   # [m]
dy = 2e-6                    # [m]

radius = 50e-6             # [m]


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
#field.U *= field.grid.dx * field.grid.dy
print(np.sum(np.sum(np.abs(field.U)**2)) * field.grid.dx * field.grid.dy)
field_far = fraunhofer.propagate(field, z)
print(np.sum(np.sum(np.abs(field_far.U)**2 )) * field_far.grid.dx * field_far.grid.dy)

# -------------------------------------------------------------------------
# Display
# -------------------------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(8, 8))

show_amplitude(field,ax = axs[0,0],title = 'Object amplitude',zoom=1)
show_phase(field_far,ax = axs[0,1])
show_amplitude(field_far,ax = axs[1,0],zoom=4)
show_intensity(field_far,ax = axs[1,1],log=False,zoom=8)
plt.tight_layout()
plt.show()