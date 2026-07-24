"""
Fresnel propagation..
"""

import numpy as np
from scipy.signal import convolve2d
from physical_optics.common.fft import fft2c, ifft2c

def kz(grid, wavelength):
    return np.sqrt(
        (2 * np.pi / wavelength) ** 2
        - grid.KX ** 2
        - grid.KY ** 2
        + 0j
    )

def propagation_phase(grid, wavelength, z):
    """Return the fresnel propagation phase."""
    k = 2 * np.pi / wavelength
    ph1 = k * z
    ph2 = - np.pi * wavelength * z * (grid.FX ** 2 + grid.FY ** 2)

    return ph1 + ph2

def transfer_function(grid, wavelength, z):
    return np.exp(1j * propagation_phase(grid, wavelength, z))

def impulse_response(grid, wavelength, z):
    k = 2 * np.pi / wavelength
    f1 = np.exp(1j * k * z ) / (1j * wavelength * z)
    f2 = np.exp(
        1j * k / (2 * z) * (grid.X ** 2 + grid.Y ** 2)
        )
    return f1 * f2

def propagate(field, z):
    """
    Fresnel propagation using Fourier Transform.

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

def propagate_convolution(field, z):
    """
    Fresnel propagation using direct spatial convolution.
    """

    h = impulse_response(field.grid, field.wavelength, z)

    field_out = field.copy()
    field_out.U = convolve2d(
        field.U,
        h,
        mode="same",
        boundary="fill",
    )

    return field_out
