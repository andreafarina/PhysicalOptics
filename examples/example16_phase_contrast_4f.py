"""
Example 16 - 4f system for phase contrast ..
"""
import numpy as np
import matplotlib.pyplot as plt

from physical_optics.common.fft import fft2c
from physical_optics.common.grid import Grid
from physical_optics.common.field import Field

from physical_optics.objects.apertures import (
    circular_aperture,
    rectangular_aperture,
    image_aperture,
)

from physical_optics.diffraction import (
    angular_spectrum,
    fresnel,
)
from physical_optics.objects.lenses import thin_lens

from physical_optics.visualization.plot import (
    show_intensity,
    show_phase,
    show_amplitude,
    show_image
)

# -------------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------------
wavelength = 633e-9     # [m]
f = 10e-3

Nx = 512 * 8
Ny = 512 * 8

dx = 2e-6               # [m]
dy = 2e-6                # [m]

# -------------------------------------------------------------------------
# Input field
# -------------------------------------------------------------------------
grid = Grid(Nx, Ny, dx, dy)

field = Field(grid, wavelength)

size_image = (2e-3,2e-3)
field.U*= rectangular_aperture(
    grid,
    width=size_image[0],
    height=size_image[1],
    )
t = image_aperture(grid,"../physical_optics/objects/Convallaria.jpg", invert=True,size=size_image)
field.U *= np.exp(1j*t)
#field.U *= 1 + 1j*t*2*np.pi
# -------------------------------------------------------------------------
# Fresnel propagation
# -------------------------------------------------------------------------
field_1 = fresnel.propagate(field, f)       # object--> lens1
field_1.U *= thin_lens(grid,wavelength,f)   # lens1 phase transform
field_1 = fresnel.propagate(field_1, f)     # lens1 --> fourier plane
field_2 = field_1.copy()
field_2b = field_1.copy()
# propagate after phase mask
hc = 1 + circular_aperture(grid,5e-6)*np.exp(1j*np.pi/2)
field_2.U *= hc
field_out = field_2.copy()
field_out = fresnel.propagate(field_out, f) # fourier plane --> lens2
field_out.U *= thin_lens(grid,wavelength,f) # lens2 phase transform
field_out = fresnel.propagate(field_out, f)# lens2 --> image
#propagate without phase mask
field_out_b = fresnel.propagate(field_2b, f) # fourier plane --> lens2
field_out_b.U *= thin_lens(grid,wavelength,f) # lens2 phase transform
field_out_b = fresnel.propagate(field_out_b, f)# lens2 --> image
# -------------------------------------------------------------------------
# Display
# -------------------------------------------------------------------------

fig, axs = plt.subplots(2, 3, figsize=(12, 8))

show_intensity(field, ax=axs[0, 0],log = False,zoom=2,title="Input intensity")
show_phase(field, ax=axs[0, 1],zoom=4,title="Input Phase")
show_amplitude(field_1, ax=axs[0, 2],zoom=32,title="Fourier plane")
show_phase(field_1, ax=axs[1, 0],zoom=8,title="Fourier plane")
show_intensity(field_out_b, ax=axs[1, 1],log = False,zoom=2,title="Output without phase mask")
show_intensity(field_out, ax=axs[1, 2],log = False,zoom=2,title="Output with phase mask")

plt.tight_layout()
plt.show()