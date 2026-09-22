"""
Example 11 - Fresnel impulse response.
"""

import matplotlib.pyplot as plt
import numpy as np

from physical_optics.common.grid import Grid

from physical_optics.diffraction import fresnel

from physical_optics.visualization.plot import (
    show_surface,show_image,
)

# -------------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------------

wavelength = 633e-9   # [m]
dz = 5e-6            # [m]

Nx = 512 * 8
Ny = 512 * 8

dx = 2e-6/32            # [m]
dy = 2e-6/32         # [m]

# -------------------------------------------------------------------------
# Input grid
# -------------------------------------------------------------------------

grid = Grid(Nx, Ny, dx, dy)

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
ax1, ax2, ax3, ax4 = axes.ravel()

z = dz
h = fresnel.impulse_response(grid, wavelength, z)
show_image(np.real(h), grid, domain="space", ax=ax1, title="Re(h)")
show_image(np.imag(h), grid, domain="space", ax=ax2, title="Imag(h)")
show_image(np.abs(h), grid, domain="space", ax=ax3, title="Abs(h)")
show_image(np.angle(h), grid, domain="space", ax=ax4, title="Angle(h)")

real_image = ax1.images[0]
imag_image = ax2.images[0]
abs_image = ax3.images[0]
angle_image = ax4.images[0]
plt.show(block=False)
plt.pause(0.1)

for i in range(1, 10):
    z = dz * (i + 1)
    # -------------------------------------------------------------------------
    # Display propagator impulse response
    # -------------------------------------------------------------------------
    h = fresnel.impulse_response(grid, wavelength, z)
    real_image.set_data(np.real(h))
    imag_image.set_data(np.imag(h))
    abs_image.set_data(np.abs(h))
    angle_image.set_data(np.angle(h))
    real_image.set_clim(np.min(np.real(h)), np.max(np.real(h)))
    imag_image.set_clim(np.min(np.imag(h)), np.max(np.imag(h)))
    abs_image.set_clim(np.min(np.abs(h)), np.max(np.abs(h)))
    angle_image.set_clim(np.min(np.angle(h)), np.max(np.angle(h)))

    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.1)
