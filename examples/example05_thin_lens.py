"""
Example 05 - Angular spectrum propagation after a thin lens..
"""

import matplotlib.pyplot as plt

from physical_optics.common.grid import Grid
from physical_optics.common.field import Field

from physical_optics.objects.apertures import (
    circular_aperture,
    rectangular_aperture,
    sinusoidal_grating,
    binary_mask
)


from physical_optics.diffraction import (
    angular_spectrum,
    fresnel
)
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

radius = 500e-6          # [m]


# -------------------------------------------------------------------------
# Input field
# -------------------------------------------------------------------------

grid = Grid(Nx, Ny, dx, dy)

field = Field(grid, wavelength)

# field.U *= circular_aperture(grid, radius)
# field.U*= rectangular_aperture(
#     grid,
#     width=100e-6,
#     height=800e-6,
#     x0 = 0,
#     y0 = 0)
field.U *= sinusoidal_grating(grid, 1e4)

field.U *= binary_mask(grid,"../physical_optics/objects/F_mask_8x8.txt",dx,size=512)

# -------------------------------------------------------------------------
# Angular spectrum propagation
# -------------------------------------------------------------------------
# field_out = angular_spectrum.propagate(field, f)
# field_out.U *= thin_lens(grid,wavelength,f)
# field_out = angular_spectrum.propagate(field_out, f)

# -------------------------------------------------------------------------
# Fresnel propagation
# -------------------------------------------------------------------------
field_out = fresnel.propagate(field, f)
field_out.U *= thin_lens(grid,wavelength,f)
field_out = fresnel.propagate(field_out, f)

# -------------------------------------------------------------------------
# Display
# -------------------------------------------------------------------------

fig, axs = plt.subplots(2, 2, figsize=(5, 5))

show_intensity(field, ax=axs[0, 0], log=False,title="Input intensity")
show_phase(field_out, ax=axs[0, 1], title="Output phase")
show_amplitude(field_out, ax=axs[1, 0], title="Output amplitude")
show_intensity(field_out, ax=axs[1, 1], log=False, title="Output intensity")

plt.tight_layout()
plt.show()