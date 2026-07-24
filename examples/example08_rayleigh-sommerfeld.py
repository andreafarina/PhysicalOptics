""""
Example 08 - Rayleigh-Sommerfield propagation
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rayleigh

from physical_optics.common.grid import Grid
from physical_optics.common.field import Field

from physical_optics.objects.apertures import (
    circular_aperture,
    rectangular_aperture
)

from physical_optics.diffraction import (
    rayleigh_sommerfeld,
    angular_spectrum
)

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
z = 5e-2            # [m]

Nx = 512*8
Ny = 512*8

dx = 2e-6         # [m]
dy = 2e-6        # [m]

radius = 5e-6          # [m]

method = 'derivative'


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
# RS  and angular spectrum propagation
# -------------------------------------------------------------------------
print(np.sum(np.abs(field.U)**2))
field_out_rs = rayleigh_sommerfeld.propagate(field, z,method=method)
field_out_as = angular_spectrum.propagate(field, z)
print(np.sum(np.abs(field_out_rs.U)**2))
print(np.sum(np.abs(field_out_as.U)**2))

print(np.max(np.abs(field_out_rs.U - field_out_as.U)))

# -------------------------------------------------------------------------
# RMS error of field
# -------------------------------------------------------------------------
err = field_out_rs.U - field_out_as.U

print("RMS error:", np.sqrt(np.mean(np.abs(err)**2)))
print("Relative RMS:",
      np.sqrt(np.mean(np.abs(err)**2)) /
      np.sqrt(np.mean(np.abs(field_out_as.U)**2)))
show_image(
    np.abs(err),
    grid,
    domain="space",
    title="|RS - AS|"
)
#
# -------------------------------------------------------------------------
# Display impulse response
# -------------------------------------------------------------------------
if method == "exact":
    h = rayleigh_sommerfeld.impulse_response(grid, wavelength, z)
elif method == "goodman":
    h = rayleigh_sommerfeld.impulse_response_goodman(grid, wavelength, z)
elif method == "derivative":
    h = rayleigh_sommerfeld.impulse_response_derivative(grid, wavelength, z)


fig, axs = plt.subplots(2, 2, figsize=(8, 8))
fig.canvas.manager.set_window_title("RS impulse response")
show_image(
    np.real(h),
    grid,
    domain="space",
    ax = axs[0,0],
    title = "Re(h)")
show_image(np.imag(h),
           grid,
           domain = "space",
           ax = axs[0,1],
           title = "Imag(h)")
show_image(
    np.abs(h),
    grid,
    domain="space",
    ax = axs[1,0],
    title = "Abs(h)")
phase = np.angle(h)#np.unwrap(np.unwrap(np.angle(h), axis=0), axis=1)
show_image(phase,
           grid,
           domain = "space",
           ax = axs[1,1],
           title = "Angle(h)")
plt.tight_layout()

# -------------------------------------------------------------------------
# Display propagator transfer function
# -------------------------------------------------------------------------
H = rayleigh_sommerfeld.transfer_function(grid, wavelength, z)
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
fig.canvas.manager.set_window_title("RS transfer function")
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
# Display propagator of angular spectrum transfer function
# -------------------------------------------------------------------------
Has = angular_spectrum.transfer_function(grid, wavelength, z)
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
fig.canvas.manager.set_window_title("Angular spectrum transfer function")
show_image(
    np.abs(Has),
    grid,
    domain="frequency",
    ax = axs[0],
    title = "abs(Has)")
show_image(np.angle(Has),
           grid,
           domain = "frequency",
           ax = axs[1],
           title = "angle(Has)")
plt.tight_layout()
# -------------------------------------------------------------------------
# Error of transfer functions
# -------------------------------------------------------------------------
errH = np.abs(H - Has)

show_image(errH,
           grid,
           domain="frequency",
           title="|H_RS - H_AS|")
# -------------------------------------------------------------------------
# Display output
# -------------------------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(8, 8))
fig.canvas.manager.set_window_title("Final propoagation with RS")
show_intensity(field, ax=axs[0, 0], log=False,title="Input intensity")
show_phase(field_out_rs, ax=axs[0, 1], title="Output phase")
show_amplitude(field_out_rs, ax=axs[1, 0], title="Output amplitude")
show_intensity(field_out_rs, ax=axs[1, 1], log=False, title="Output intensity")

plt.tight_layout()

plt.show()