"""
Example 12 - Imaging through a thin lens..
If you use transmission grating, pay attention to aliasing and grid sampling.
"""

import matplotlib.pyplot as plt
import numpy as np

from physical_optics.common.grid import Grid
from physical_optics.common.field import Field

from physical_optics.objects.apertures import (
    circular_aperture,
    fresnel_number,
    rectangular_aperture,
    sinusoidal_grating,
    binary_mask
)


from physical_optics.diffraction import (
    angular_spectrum,
    fresnel
)
from physical_optics.objects.lenses import thin_lens
from physical_optics.imaging.coherent import coherent_impulse_response

from physical_optics.visualization.plot import show_intensity, show_phase, show_amplitude, show_image

# -------------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------------

wavelength = 633e-9     # [m]
f = 100e-3
do = 2*f                # object distance
di = (1/f - 1/do)**(-1) # image distance

Nx = 512 * 8
Ny = 512 * 8

dx = 2e-5                # [m]
dy = 2e-5                # [m]
# # -------------------------------------------------------------------------
# # Define lens pupil
# # -------------------------------------------------------------------------
radius = 12.5e-3          # [m]
grid_pupil = Grid(Nx, Ny, dx, dy)
pupil = circular_aperture(grid_pupil, radius)
#CIR, new_grid = coherent_impulse_response(grid_pupil,pupil,wavelength,di,pad_factor=4)

# -------------------------------------------------------------------------
# Input field
# -------------------------------------------------------------------------
grid = Grid(Nx, Ny, dx, dy)
field = Field(grid, wavelength)
#field.U *= rectangular_aperture(grid, width=2e-3,height=2e-3)
#field.U *= sinusoidal_grating(grid, 5e4)
field.U *= binary_mask(grid,"../physical_optics/objects/F_mask_8x8.txt",dx,size=1024)
# # -------------------------------------------------------------------------
# # Propagate object to the lens
# # -------------------------------------------------------------------------
field_1 = fresnel.propagate(field, do,method="fourier")

# # -------------------------------------------------------------------------
# # apply the lens phase function
# # -------------------------------------------------------------------------
field_2 = field_1.copy()
field_2.U *= pupil
field_2.U *= thin_lens(field_2.grid, wavelength, f)

# # -------------------------------------------------------------------------
# # Propagate to fourier plane
# # -------------------------------------------------------------------------
field_3 = fresnel.propagate(field_2, f,method="fourier")
print(f"Fresnel Number: {fresnel_number(radius,wavelength,f):.3f}")
# # -------------------------------------------------------------------------
# # Propagate to image plane
# # -------------------------------------------------------------------------
field_out = fresnel.propagate(field_3, di-f,method="fourier")
# -------------------------------------------------------------------------
# Display The Coherent Impulse response
# -------------------------------------------------------------------------
# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# show_image(pupil,ax=axs[0],grid=grid_pupil,domain="space")
# show_image(np.abs(CIR),ax=axs[1],grid=new_grid,domain="space",zoom=32)
# print(f"Jinc zero:,{1.22*wavelength/(2*radius)*di*1e3:.4f} mm")

# -------------------------------------------------------------------------
# Display Output field
# -------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
show_amplitude(field, ax=axs[0],title="Input ampl.")
show_phase(field, ax=axs[1], title="Input phase")

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
show_amplitude(field_1, ax=axs[0], title="Ampl. on the lens")
show_phase(field_1, ax=axs[1], title="Phase on the lens")

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
show_amplitude(field_2, ax=axs[0], title="Ampl. after the lens")
show_phase(field_2, ax=axs[1], title="Phase after the lens")

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
show_amplitude(field_3, ax=axs[0], zoom=8,title="Ampl. on the Fourier plane")
show_phase(field_3, ax=axs[1], zoom=8,title="Phase on the Fourier plane")

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
show_amplitude(field_out, ax=axs[0], title="Output ampl.")
show_phase(field_out, ax=axs[1], title="Output phase")
print(field_out.power())
print(field_2.power())
print(field_3.power())
#show_intensity(field_out, ax=axs[2, 2], log=False, title="Output intensity")

plt.tight_layout()
plt.show()