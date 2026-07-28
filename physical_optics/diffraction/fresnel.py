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
    """
    Return the Fresnel transfer function (Goodman Eq 4.21)
    """
    return np.exp(1j * propagation_phase(grid, wavelength, z))

def impulse_response(grid, wavelength, z):
    """
        Return the Fresnel impulse response (Goodman Eq. 4.16)
        """
    k = 2 * np.pi / wavelength
    f1 = np.exp(1j * k * z ) / (1j * wavelength * z)
    f2 = np.exp(
        1j * k / (2 * z) * (grid.X ** 2 + grid.Y ** 2)
        )
    return f1 * f2

def propagate(field, z, method="transfer_function"):
    """
    Fresnel propagation.

    Parameters
    ----------
    field : Field
        Input optical field.
    z : float
        Propagation distance [m].
    method : {"transfer_function", "convolution", "fourier"}, optional
        Numerical implementation of the Fresnel propagator.

        - "transfer_function": frequency-domain transfer function.
        - "convolution": spatial convolution with the impulse response.
        - "fourier": scaled Fourier transform (Goodman Eq. 4.17).

    Returns
    -------
    Field
        Propagated optical field.
    """

    if method == "transfer_function":
        return _propagate_transfer_function(field, z)

    if method == "convolution":
        return _propagate_convolution(field, z)

    if method == "fourier":
        return _propagate_fourier(field, z)

    raise ValueError(
        "method must be "
        "'transfer_function', "
        "'convolution' or "
        "'fourier'"
    )

def _propagate_transfer_function(field, z):
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

def _propagate_convolution(field, z):
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

def _propagate_fourier(field, z):
    """
    Fresnel propagation using the scaled Fourier transform
    formulation (Goodman Eq. 4.17).
    """

    grid = field.grid
    wavelength = field.wavelength
    k = 2 * np.pi / wavelength

    # Input quadratic phase
    Qin = np.exp(
        1j * k / (2 * z)
        * (grid.X**2 + grid.Y**2)
    )

    # Fourier transform
    A = fft2c(field.U * Qin) * grid.dx * grid.dy

    # Output grid
    grid_out = grid.scaled_fourier_grid(
        wavelength,
        z,
    )

    # Output quadratic phase
    Qout = np.exp(
        1j * k / (2 * z)
        * (grid_out.X**2 + grid_out.Y**2)
    )

    field_out = field.copy()
    field_out.grid = grid_out

    field_out.U = (
        np.exp(1j * k * z)
        / (1j * wavelength * z)
        * Qout
        * A
    )

    return field_out
