"""
Example 13 - Plot the Coherent Impulse Response ad Coherent Transfer Function of a lens..
"""

import matplotlib.pyplot as plt
import numpy as np

from physical_optics.common.grid import Grid

from physical_optics.objects.apertures import (
    circular_aperture,
    rectangular_aperture,
)

from physical_optics.imaging.coherent import (
    coherent_impulse_response,
    coherent_transfer_function
)

from physical_optics.visualization.plot import show_intensity, show_phase, show_amplitude, show_image

# -------------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------------
wavelength = 633e-9     # [m]
f = 100e-3
do = 2*f                # object distance
di = (1/f - 1/do)**(-1) # image distance

Nx = 512 * 4
Ny = 512 * 4

dx = 2e-5                # [m]
dy = 2e-5                # [m]
# # -------------------------------------------------------------------------
# # Define lens pupil
# # -------------------------------------------------------------------------
radius = 12.5e-3          # [m]
grid = Grid(Nx, Ny, 1*dx, 1*dy)
pupil = circular_aperture(grid, radius)
CIR, new_grid = coherent_impulse_response(grid,pupil,wavelength,di,pad_factor=4)
CTF, new_gridF = coherent_transfer_function(grid,pupil,wavelength,di)
# -------------------------------------------------------------------------
# Display The Coherent Impulse response
# -------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
show_image(pupil,ax=axs[0],grid=grid,domain="space",title="Pupil profile")
show_image(np.abs(CIR),ax=axs[1],grid=new_grid,domain="space",zoom=32,title="Coherent impulse response")
print(f"Jinc zero:,{1.22*wavelength/(2*radius)*di*1e3:.4f} mm")
plt.tight_layout()
# -------------------------------------------------------------------------
# Display The Coherent Transfer Function
# -------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
show_image(pupil,ax=axs[0],grid=grid,domain="space",title="Pupil profile")
show_image(np.abs(CTF),ax=axs[1],grid=new_gridF,domain="frequency",zoom=1,title="Coherent transfer function")
print(f"Circ zero:,{radius/(wavelength*di)*1e-3:.4f} mm^-1")
plt.tight_layout()
plt.show()