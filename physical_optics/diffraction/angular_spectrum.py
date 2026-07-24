"""
Angular spectrum propagation..
"""

import numpy as np
from physical_optics.common.fft import fft2c, ifft2c

def kz(grid, wavelength):
    return np.sqrt(
        (2 * np.pi / wavelength) ** 2
        - grid.KX ** 2
        - grid.KY ** 2
        + 0j
    )

def propagation_phase(grid, wavelength, z):
    """Return the angular spectrum propagation phase kz*z."""
    return kz(grid, wavelength) * z

def transfer_function(grid, wavelength, z):
    return np.exp(1j * propagation_phase(grid, wavelength, z))

def propagate(field, z):
    """
    Angular spectrum propagation.

    Parameters
    ----------
    field : Field
        Input optical field.
    z : float
        Propagation distance [m].

    Returns
    -------
    Field
        Propagated optical field.
    """

    # Angular spectrum
    A = fft2c(field.U)

    # Transfer function
    H = transfer_function(field.grid, field.wavelength, z)

    # Propagation
    A *= H

    # Output field
    field_out = field.copy()
    field_out.U = ifft2c(A)

    return field_out