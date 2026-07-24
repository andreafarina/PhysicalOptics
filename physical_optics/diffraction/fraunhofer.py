"""
Fraunhofer diffraction.
"""

from physical_optics.common.fft import fft2c
import numpy as np
from physical_optics.common.grid import Grid
from physical_optics.common.field import Field


def output_grid(field, z):
    """Return the output sampling grid for Fraunhofer propagation."""

    dx_out = field.wavelength * z / field.grid.Lx
    dy_out = field.wavelength * z / field.grid.Ly

    return Grid(
        field.grid.Nx,
        field.grid.Ny,
        dx_out,
        dy_out,
    )


def quadratic_phase(grid, wavelength, z):
    """Return the quadratic phase factor on the observation plane."""

    k = 2 * np.pi / wavelength

    return np.exp(
        1j * k / (2 * z)
        * (grid.X**2 + grid.Y**2)
    )


def propagation_factor(wavelength, z):
    """Return the Fraunhofer propagation prefactor."""

    k = 2 * np.pi / wavelength

    return np.exp(1j * k * z) / (1j * wavelength * z)


def propagate(field, z):
    """
    Propagate an optical field in the Fraunhofer approximation.

    Parameters
    ----------
    field : Field
        Input optical field.
    z : float
        Propagation distance [m].

    Returns
    -------
    Field
        Optical field propagated to a distance z in the
        Fraunhofer approximation.
    """

    grid_out = output_grid(field, z)

    field_out = Field(grid_out, field.wavelength)

    field_out.U = (
        propagation_factor(field.wavelength, z)
        * quadratic_phase(grid_out, field.wavelength, z)
        * fft2c(field.U) * field.grid.dx * field.grid.dy
    )

    return field_out