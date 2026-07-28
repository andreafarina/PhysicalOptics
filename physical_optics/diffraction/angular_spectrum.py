"""
Angular Spectrum propagation.

This module implements the Angular Spectrum method, an exact
frequency-domain solution of the scalar Helmholtz equation.

The optical field is decomposed into plane-wave components, each
propagating independently according to its longitudinal wave-vector
component.

References
----------
J. W. Goodman,
Introduction to Fourier Optics,
3rd Edition,
Section 3.10.
"""

import numpy as np
from physical_optics.common.fft import fft2c, ifft2c

def kz(grid, wavelength):
    """
    Return the longitudinal wave-vector component.

    Computes the longitudinal component of the wave vector

        kz = sqrt(k² - kx² - ky²)

    where

        k = 2π / λ.

    Plane-wave components satisfying
        kx² + ky² <= k²
    correspond to propagating waves, whereas higher spatial
    frequencies produce an imaginary kz and therefore represent
    evanescent waves.

    Parameters
    ----------
    grid : Grid
        Spatial-frequency grid.
    wavelength : float
        Optical wavelength [m].

    Returns
    -------
    ndarray
        Longitudinal wave-vector component.

    References
    ----------
    Goodman, 3rd Edition,
    Section 3.10.
    """
    return np.sqrt(
        (2 * np.pi / wavelength) ** 2
        - grid.KX ** 2
        - grid.KY ** 2
        + 0j
    )

def propagation_phase(grid, wavelength, z):
    """
    Return the angular-spectrum propagation phase.

    Computes the phase

        kz · z

    accumulated by each plane-wave component during propagation.

    Parameters
    ----------
    grid : Grid
        Spatial-frequency grid.
    wavelength : float
        Optical wavelength [m].
    z : float
        Propagation distance [m].

    Returns
    -------
    ndarray
        Angular-spectrum propagation phase.

    References
    ----------
    Goodman, 3rd Edition,
    Section 3.10.
    """
    return kz(grid, wavelength) * z

def transfer_function(grid, wavelength, z):
    """
    Return the Angular Spectrum transfer function.

    Computes the transfer function

        H(kx, ky) = exp(i kz z)

    describing the propagation of every plane-wave component in
    the angular spectrum representation.

    Parameters
    ----------
    grid : Grid
        Spatial-frequency grid.
    wavelength : float
        Optical wavelength [m].
    z : float
        Propagation distance [m].

    Returns
    -------
    ndarray
        Angular Spectrum transfer function.

    References
    ----------
    Goodman, 3rd Edition,
    Section 3.10.
    """
    return np.exp(1j * propagation_phase(grid, wavelength, z))

def propagate(field, z):
    """
    Propagate an optical field using the Angular Spectrum method.

    The propagation is performed by

    1. Computing the Fourier transform of the input field.
    2. Multiplying the angular spectrum by the transfer function.
    3. Computing the inverse Fourier transform.

    This method is an exact scalar solution of the Helmholtz
    equation and naturally includes both propagating and
    evanescent plane-wave components.

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

    References
    ----------
    J. W. Goodman,
    Introduction to Fourier Optics,
    3rd Edition,
    Section 3.10.
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