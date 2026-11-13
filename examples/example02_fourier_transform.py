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
    show_intensity, show_image,
    show_spectrum, show_surface,show_lineplot
)

import matplotlib.pyplot as plt
import numpy as np


# Grid
grid = Grid(1024+1, 1024+1, 1e-6/4, 1e-6/4)

# Optical field
field = Field(grid, 633e-9)

## Rectangular aperture
# field.U *= rectangular_aperture(
#     grid,
#     width=5e-6,
#     height=10e-6,
# )
# Circular aperture
field.U *= circular_aperture(grid, radius=5e-6)

# sinusoidal_grating
# grid = Grid(512, 512, 2e-6, 2e-6)
# field.U *=sinusoidal_grating(grid,
#                        frequency = 10e3,
#                        modulation=1.0,
#                        offset=0.5,
#                        angle=0.,
#                        x0=0.0,
#                        y0=0.0)
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
fig = plt.figure(figsize=(10, 12))
# 3D plots
ax1 = fig.add_subplot(3, 2, 1, projection="3d")
ax2 = fig.add_subplot(3, 2, 2, projection="3d")
show_surface(np.real(field.U),grid,ax = ax1,domain="space",zoom=16)
show_surface(np.real(A),grid,ax = ax2,domain="frequency",zoom=2)

# 2D plots
ax3 = fig.add_subplot(3, 2, 3)
ax4 = fig.add_subplot(3, 2, 4)

show_image(np.real(field.U),grid,ax = ax3 ,domain="space",zoom=16)
show_image(np.real(A),grid,ax = ax4,domain="frequency")
# 1D plot
ax5 = fig.add_subplot(3, 2, 5)
ax6 = fig.add_subplot(3, 2, 6)
show_lineplot(np.real(field.U), grid, ax = ax5, direction="horizontal", domain="space",zoom=16)
show_lineplot(np.real(A), grid, ax = ax6, direction="horizontal", domain="frequency",zoom=2)

# plt.tight_layout()
# Export figure for the handbook
# OUTPUT_FILE = "../docs/handbook/fig/circle_Fourier"
# fig.savefig(OUTPUT_FILE+".pdf", bbox_inches="tight")
# fig.savefig(OUTPUT_FILE+".png", dpi=300, bbox_inches="tight")

plt.show()