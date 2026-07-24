""""
Example 07 - Fresnel propagation
"""

import matplotlib.pyplot as plt
import numpy as np

from physical_optics.common.grid import Grid
from physical_optics.common.field import Field

from physical_optics.objects.apertures import (
    circular_aperture,
    rectangular_aperture
)


from physical_optics.diffraction import fresnel

from physical_optics.visualization.plot import (
    show_intensity,
    show_phase,
    show_amplitude,
    show_image
)

# -------------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------------

wavelength = 633e-9   # [m]
z = 5e-3 * 2            # [m]

Nx = 512
Ny = 512

dx = 2e-6                # [m]
dy = 2e-6              # [m]

radius = 50e-6          # [m]


# -------------------------------------------------------------------------
# Input field
# -------------------------------------------------------------------------

grid = Grid(Nx, Ny, dx, dy)

field = Field(grid, wavelength)

# field.U *= circular_aperture(grid, radius)
field.U*= rectangular_aperture(
    grid,
    width=40e-6,
    height=80e-6,
    x0 = 0,
    y0 = 0)


# -------------------------------------------------------------------------
# Fresnel propagation:
# test fresnel.propagate_convolution and look at the computation time
# -------------------------------------------------------------------------
print(np.sum(np.abs(field.U)**2))
field_out = fresnel.propagate(field, z)
print(np.sum(np.abs(field_out.U)**2))
# -------------------------------------------------------------------------
# Display propagator phase
# -------------------------------------------------------------------------
phase = fresnel.propagation_phase(grid, wavelength, z)

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
show_image(
    np.real(phase),
    grid,
    domain="frequency",
    ax = axs[0],
    title = "Re(prop phase)")
show_image(np.imag(phase),
           grid,
           domain = "frequency",
           ax = axs[1],
           title = "Imag(prop phase)")
plt.tight_layout()

# -------------------------------------------------------------------------
# Display propagator transfer function
# -------------------------------------------------------------------------
H = fresnel.transfer_function(grid, wavelength, z)
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
show_image(
    np.abs(H),
    grid,
    domain="frequency",
    ax = axs[0],
    title = "abs(H)")
show_image(np.angle(H),
           grid,
           domain = "frequency",
           ax = axs[1],
           title = "angle(H)")
plt.tight_layout()

# -------------------------------------------------------------------------
# Display output
# -------------------------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(8, 8))

show_intensity(field, ax=axs[0, 0], log=False,title="Input intensity")
show_phase(field_out, ax=axs[0, 1], title="Output phase")
show_amplitude(field_out, ax=axs[1, 0], title="Output amplitude")
show_intensity(field_out, ax=axs[1, 1], log=False, title="Output intensity")

plt.tight_layout()

plt.show()