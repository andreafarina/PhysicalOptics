"""
Rayleigh-Sommerfeld propagation.
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


def impulse_response(grid, wavelength, z):
    """
    Exact Rayleigh-Sommerfeld impulse response.
    It uses Eq (3-34) and put the result in Eq (3-36). Goodman
    """
    k = 2 * np.pi / wavelength
    r = np.sqrt(grid.X**2 + grid.Y**2 + z**2)

    G = np.exp(1j*k*r) / r          # Spherical Green function term
    cos_nr = z / r                  # cos of the angle between the normal to the aperture and r
    A = 1 / (2 * np.pi)
    K = 1j*k - 1 / r                # complete term
    return -A * G * K * cos_nr

def impulse_response_goodman(grid, wavelength, z):
    """
    Rayleigh-Sommerfeld impulse response as proposed by Goodman r01>>wavelength.
    Eq (3-40) Goodman
    """
    k = 2 * np.pi / wavelength
    r = np.sqrt(grid.X**2 + grid.Y**2 + z**2)

    G = np.exp(1j*k*r) / r          # Spherical Green function term
    cos_nr = z / r                  # cos of the angle between the normal to the aperture and r
    A = 1 / (2 * np.pi)
    K = 1j*k                        # approximated term complete term
    AK = -1 / (1j * wavelength)     # the product A*K with mult/div by j
    return AK * G * cos_nr

def impulse_response_derivative(grid, wavelength, z):
    """
    Rayleigh-Sommerfeld impulse response using numerical derivative in z.
    Eq (3-36) Goodman
    """
    k = 2 * np.pi / wavelength
    eps = 1e-12
    rp = np.sqrt(grid.X**2 + grid.Y**2 + (z + eps)**2)
    rn = np.sqrt(grid.X ** 2 + grid.Y ** 2 + (z - eps) ** 2)

    Gp = np.exp(1j*k*rp) / rp           # Spherical Green function term
    Gn = np.exp(1j * k * rn) / rn       # Spherical Green function term
    dG = (Gp - Gn) / (2 * eps)
    return -1 / (2 * np.pi) * dG

def transfer_function(grid, wavelength, z, method="exact"):
    if method == "exact":
        h = impulse_response(grid, wavelength, z)
    elif method == "goodman":
        h = impulse_response_goodman(grid, wavelength, z)
    elif method == "derivative":
        h = impulse_response_derivative(grid, wavelength, z)
    else:
        raise ValueError("method must be 'exact', 'goodman' or 'derivative")

    return fft2c(h) * grid.dx * grid.dy

def propagate(field, z, method="exact"):
    """
    Rayleigh-Sommerfeld propagation using Fourier Transform.

    Parameters
    ----------
    field : Field
        Input optical field.
    z : float
        Propagation distance [m].
    method : {"exact", "goodman", "derivative"}, optional
        Rayleigh-Sommerfeld impulse response to use. Default is "exact".

    Returns
    -------
    Field
        Propagated optical field.
    """

    # Angular spectrum
    A = fft2c(field.U)

    # Transfer function
    H = transfer_function(field.grid, field.wavelength, z, method=method)

    # Propagation
    A *= H

    # Output field
    field_out = field.copy()
    field_out.U = ifft2c(A)

    return field_out

def propagate_convolution(field, z,method="exact"):
    """
    Rayleigh-Sommerfeld propagation using direct spatial convolution.
    """
    if method == "exact":
        h = impulse_response(field.grid, field.wavelength, z)
    elif method == "goodman":
        h = impulse_response_goodman(field.grid, field.wavelength, z)
    elif method == "derivative":
        h = impulse_response_derivative(field.grid, field.wavelength, z)
    else:
        raise ValueError("method must be 'exact', 'goodman' or 'derivative")
    h = impulse_response(field.grid, field.wavelength, z)

    field_out = field.copy()
    field_out.U = convolve2d(
        field.U,
        h,
        mode="same",
        boundary="fill",
    )

    return field_out
