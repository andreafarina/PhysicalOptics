"""
Example 06 - 4f system for spatial filtering..
"""

import matplotlib.pyplot as plt

from physical_optics.common.grid import Grid
from physical_optics.common.field import Field

from physical_optics.objects.apertures import (
    circular_aperture,
    rectangular_aperture,
    abbe_porter_grating, vertical_slit
)

from physical_optics.diffraction import angular_spectrum
from physical_optics.objects.lenses import thin_lens

from physical_optics.visualization.plot import show_intensity, show_phase, show_amplitude

# -------------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------------

wavelength = 633e-9     # [m]
f = 100e-3

Nx = 512 * 4
Ny = 512 * 4

dx = 2e-6                # [m]
dy = 2e-6                # [m]

# -------------------------------------------------------------------------
# Input field
# -------------------------------------------------------------------------

grid = Grid(Nx, Ny, dx, dy)

field = Field(grid, wavelength)

# abbe-porter grating
field.U *= abbe_porter_grating(grid,
                               period_x=200e-6,
                               period_y=200e-6,
                               line_width_x=50e-6,
                               line_width_y=50e-6,)
# field.U*= rectangular_aperture(
#     grid,
#     width=100e-6,
#     height=800e-6,
#     x0 = 0,
#     y0 = 0)


# -------------------------------------------------------------------------
# Angular spectrum propagation
# -------------------------------------------------------------------------
field_1 = angular_spectrum.propagate(field, f)
field_1.U *= thin_lens(grid,wavelength,f)
field_1 = angular_spectrum.propagate(field_1, f)
field_2 = field_1.copy()
field_2.U *= vertical_slit(
    grid,
    width=100e-6,
    x0 = 0,
    )
field_3 = field_2.copy()
field_3 = angular_spectrum.propagate(field_3, 2*f)
field_3.U *= thin_lens(grid,wavelength,f)
field_3 = angular_spectrum.propagate(field_3, f)

# -------------------------------------------------------------------------
# Display
# -------------------------------------------------------------------------

fig, axs = plt.subplots(2, 2, figsize=(8, 8))

show_amplitude(field, ax=axs[0, 0],title="Input intensity")
show_amplitude(field_1, ax=axs[0, 1],title="Fourier plane")
show_amplitude(field_2, ax=axs[1, 0],title="after filtering")
show_amplitude(field_3, ax=axs[1, 1],title="Output")


plt.tight_layout()
plt.show()