"""
Example 04 - Angular spectrum propagation.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from physical_optics.common.grid import Grid
from physical_optics.common.field import Field

from physical_optics.objects.apertures import (
    circular_aperture,
    rectangular_aperture
)


from physical_optics.diffraction import angular_spectrum

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
z = 5e-8             # [m]

Nx = 512 * 8
Ny = 512 * 8

dx = 2e-6 / 8           # [m]
dy = 2e-6 / 8          # [m]

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
print(field.power())

# -------------------------------------------------------------------------
# Angular spectrum propagation
# -------------------------------------------------------------------------

field_out = angular_spectrum.propagate(field, z)
print(field_out.power())
# -------------------------------------------------------------------------
# Display propagator phase
# -------------------------------------------------------------------------
phase = angular_spectrum.propagation_phase(grid, wavelength, z)

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
H = angular_spectrum.transfer_function(grid, wavelength, z)
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
fig.canvas.manager.set_window_title("Angular spectrum transfer function")
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



fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(
    grid.X * 1e3,
    grid.Y * 1e3,
    np.angle(H),
    cmap="viridis",
    linewidth=0,
    antialiased=True,
)

ax.set_xlabel("fx [mm-1]")
ax.set_ylabel("fy [mm-1]")
ax.set_zlabel("Amplitude")
ax.set_title("H amplitude")
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