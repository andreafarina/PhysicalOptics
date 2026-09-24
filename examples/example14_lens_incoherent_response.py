"""
Example 14 - Plot the Incoherent Impulse Response (PSF) ad Optical Transfer Function (OTF) of a lens..
"""

import matplotlib.pyplot as plt
import numpy as np

from physical_optics.common.grid import Grid

from physical_optics.objects.apertures import (
    circular_aperture,
    rectangular_aperture,
)

from physical_optics.imaging.incoherent import (
    PSF,
    OTF
)

from physical_optics.visualization.plot import (
    show_image, show_surface,
    show_lineplot
)

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
grid = Grid(Nx, Ny, 2*dx, 2*dy)
pupil = circular_aperture(grid, radius)
PSFf, new_grid = PSF(grid,pupil,wavelength,di,pad_factor=4)
OTF, new_gridF = OTF(grid,pupil,wavelength,di)
# -------------------------------------------------------------------------
# Display The Point Spread Function
# -------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#axs[0]= fig.add_subplot(1, 2, 1, projection="3d")
#show_surface(np.abs(PSFf),ax=axs[0],grid=new_grid,zoom=1,domain="space",title="PSF")
show_image(np.abs(PSFf),ax=axs[0],grid=new_grid,domain="space",zoom=32,title="PSF")
show_lineplot(np.abs(PSFf),ax=axs[1],grid=new_grid,zoom=64,domain="space",title="PSF")
print(f"Jinc zero: { 1.22*wavelength/(2*radius)*di*1e3:.4f} mm")
plt.tight_layout()
# -------------------------------------------------------------------------
# Display The Optical Transfer Function
# -------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0]= fig.add_subplot(1, 2, 1, projection="3d")
show_surface(np.abs(OTF),ax=axs[0],grid=new_gridF,domain="frequency",title="OTF")
show_lineplot(np.abs(OTF),ax=axs[1],grid=new_gridF,domain="frequency",title="OTF")

print(f"OTF zero: {2 * radius/(wavelength*di)*1e-3:.4f} mm^-1")
plt.tight_layout()
plt.show()