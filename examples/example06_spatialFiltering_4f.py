"""
Example 06 - 4f system for spatial filtering..
"""
import numpy as np
import matplotlib.pyplot as plt

from physical_optics.common.fft import fft2c
from physical_optics.common.grid import Grid
from physical_optics.common.field import Field

from physical_optics.objects.apertures import (
    circular_aperture,
    rectangular_aperture,
    abbe_porter_grating, vertical_slit
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

period = 200e-6
line_width = 50e-6
# -------------------------------------------------------------------------
# Input field
# -------------------------------------------------------------------------
grid = Grid(Nx, Ny, dx, dy)

field = Field(grid, wavelength)

# abbe-porter grating
field.U *= abbe_porter_grating(grid,
                               period_x=period,
                               period_y=period,
                               line_width_x=line_width,
                               line_width_y=line_width,)
field.U*= rectangular_aperture(
    grid,
    width=2e-3,
    height=2e-3,
    x0 = 0,
    y0 = 0)

# -------------------------------------------------------------------------
# Fresnel propagation
# -------------------------------------------------------------------------
field_1 = fresnel.propagate(field, f)       # object--> lens1
field_1.U *= thin_lens(grid,wavelength,f)   # lens1 phase transform
field_1 = fresnel.propagate(field_1, f)     # lens1 --> fourier plane
field_2 = field_1.copy()
# field_2.U *= vertical_slit(                 # spatial filter
#     grid,
#     width=10e-6,
#     x0 = 0,
#     )
field_out = field_2.copy()
field_out = fresnel.propagate(field_out, f) # fourier plane --> lens2
field_out.U *= thin_lens(grid,wavelength,f) # lens2 phase transform
field_out = fresnel.propagate(field_out, f)# lens2 --> image
# DEBUG: print expected positions
#print(f"Expected grating spectrum positions:{1/period*wavelength*f*1e3:.3f} mm")

# -------------------------------------------------------------------------
# Check direct Fourier transform from the Fourier plane
# -------------------------------------------------------------------------
# Fourier Transform after Fresnel propagation to the fourier plane
field_2f = field_2.copy()

field_outFFT = field_2f.copy()
field_outFFT.U = 1/(wavelength * f) * fft2c(field_outFFT.U) * field_outFFT.grid.dx * field_outFFT.grid.dy
# change the grid to go back to space domain
field_outFFT.grid= field_outFFT.grid.scaled_fourier_grid(wavelength,f)

# -------------------------------------------------------------------------
# Display
# -------------------------------------------------------------------------

fig, axs = plt.subplots(2, 2, figsize=(8, 8))

show_intensity(field, ax=axs[0, 0],log = False,title="Input intensity")
show_amplitude(field_1, ax=axs[0, 1],zoom=16,title="Fourier plane")
show_amplitude(field_2, ax=axs[1, 0],zoom=16,title="after filtering")
show_intensity(field_out, ax=axs[1, 1],log = False,title="Output")

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
show_image(np.abs(field_2f.U),field_2f.grid, ax=axs[0],zoom=16,domain="frequency",title="Fourier Plane")
show_intensity(field_outFFT, ax=axs[1],log = False,title="Output intensity with FFT")


plt.tight_layout()
plt.show()