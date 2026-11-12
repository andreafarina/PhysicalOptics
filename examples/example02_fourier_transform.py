from scipy.stats import false_discovery_control

from physical_optics.common.grid import Grid
from physical_optics.common.field import Field
from physical_optics.common.fft import fft2c

from physical_optics.objects.apertures import (
    rectangular_aperture,
    circular_aperture,
    sinusoidal_grating,
    vertical_grating,
    square_grating,
    abbe_porter_grating)

from physical_optics.visualization.plot import (
    show_intensity,
    show_spectrum, show_surface,show_lineplot
)

import matplotlib.pyplot as plt
import numpy as np


# Grid
grid = Grid(512, 512, 5e-6, 5e-6)

# Optical field
field = Field(grid, 633e-9)

## Rectangular aperture
# field.U *= rectangular_aperture(
#     grid,
#     width=50e-6,
#     height=100e-6,
# )
# Circular aperture
# field.U *= circular_aperture(grid, radius=50e-6)

# sinusoidal_grating
grid = Grid(512, 512, 2e-6, 2e-6)
field.U *=sinusoidal_grating(grid,
                       frequency = 10e3,
                       modulation=1.0,
                       offset=0.5,
                       angle=0.,
                       x0=0.0,
                       y0=0.0)
# vertical grid
# field.U *= vertical_grating(grid,
#                             period=150e-6,
#                             width = 50e-6)
# # square grating
# field.U *= square_grating(grid,
#                             period_x=500e-6,
#                             period_y=500e-6,
#                             width_x= 50e-6,
#                             width_y = 50e-6)

# abbe-porter grating
# field.U *= abbe_porter_grating(grid,
#                                period_x=200e-6,
#                                period_y=200e-6,
#                                line_width_x=50e-6,
#                                line_width_y=50e-6,)
# Fourier transform
A = fft2c(field.U) * grid.dx * grid.dy

# Display
fig, axs = plt.subplots(1, 2, figsize=(8, 4))

show_intensity(field,ax = axs[0],log=False)
show_spectrum(grid, A, ax = axs[1],log=False)
plt.tight_layout()

show_surface(np.real(A),grid,domain="frequency",zoom=2)
show_lineplot(np.real(A), grid, direction="horizontal", domain="frequency")
plt.show()