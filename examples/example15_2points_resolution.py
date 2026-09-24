"""
Example 15 - Two points resolution..
Compare the resolution of two near point sources using coherent and incoherent illumination.
You can change the phase of one of the two point sources and look at the effect.
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
    binary_mask,
    point
)
from physical_optics.common.fft import fft2c, ifft2c

from physical_optics.diffraction import (
    angular_spectrum,
    fresnel
)
from physical_optics.objects.lenses import thin_lens
from physical_optics.imaging.coherent import coherent_transfer_function
from physical_optics.imaging.incoherent import OTF

from physical_optics.visualization.plot import (
    show_intensity, show_phase, show_amplitude, show_image, show_lineplot
)

# -------------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------------

wavelength = 633e-9     # [m]
di = 200e-3                # object distance

Nx = 512 * 8
Ny = 512 * 8

dx = 2e-5                # [m]
dy = 2e-5                # [m]

# # -------------------------------------------------------------------------
# # Define phase difference between the two points
# # -------------------------------------------------------------------------
phi = np.pi
# # -------------------------------------------------------------------------
# # Define lens pupil
# # -------------------------------------------------------------------------
radius = 12.5e-4          # [m]
grid_pupil = Grid(Nx, Ny, dx, dy)
pupil = circular_aperture(grid_pupil, radius)
CTF, grid = coherent_transfer_function(grid_pupil,pupil,wavelength,di)
OTF,_ = OTF(grid_pupil,pupil,wavelength,di)
# -------------------------------------------------------------------------
# Input field
# -------------------------------------------------------------------------
field = Field(grid, wavelength)
x0 = 1.22 * wavelength/(2*radius) * di
print(f"Theoretical resolution limit: {x0 * 1e6:.3f} um")

field.U *= (point(grid,x0/2) +
            np.exp(1j*phi)*point(grid,-x0/2))
# field.U *= (point(grid,1e-4*0.6) +
#             np.exp(1j*np.pi)*point(grid,-1e-4*0.6))
print(f"Fresnel Number: {fresnel_number(radius,wavelength,di):.3f}")
# # -------------------------------------------------------------------------
# # Propagate to image plane through the CTF
# # -------------------------------------------------------------------------
U_out = ifft2c(fft2c(field.U) * CTF)
field_out = Field(grid, wavelength)
field_out.U *= U_out
# # -------------------------------------------------------------------------
# # Propagate to image plane through the OTF
# # -------------------------------------------------------------------------
I_in = np.abs(field.U)**2
I_out = np.abs(ifft2c(fft2c(I_in) * OTF))

# -------------------------------------------------------------------------
# Display Output field
# -------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
show_amplitude(field, ax=axs[0],zoom=64,title="Input ampl.")
show_phase(field, ax=axs[1], zoom=64,title="Input phase")

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
show_amplitude(field_out, ax=axs[0], zoom=4, title="Output ampl.")
show_phase(field_out, ax=axs[1], zoom=4, title="Output phase")

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
show_intensity(field_out, ax=axs[0], zoom=4,log=False, title="CTF: Output intensity")
show_lineplot(np.abs(field_out.U)**2, grid, ax=axs[1], title="CTF: Output intensity", zoom = 8)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
show_image(I_out, grid, ax=axs[0], zoom=4, title="OTF: Output intensity")
show_lineplot(I_out, grid, ax=axs[1], title="OTF: Output intensity", zoom = 8)
plt.tight_layout()
plt.show()