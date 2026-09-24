"""Incoherent imaging functions."""

import numpy as np
from scipy.signal import correlate2d

from physical_optics.imaging.coherent import (
    coherent_impulse_response,
    coherent_transfer_function
)

from physical_optics.common.fft import fft2c


def OTF(grid, pupil, wavelength, distance, pad_factor=1):
    r"""Return the incoherent optical transfer function.

    The incoherent OTF is the normalized autocorrelation of the pupil:

    .. math::

        OTF = P \star P^*.
    """
    # ctf, grid_out = coherent_transfer_function(grid, pupil, wavelength, distance)
    # very slow the correlate 2D
    # otf = correlate2d(
    #     ctf,
    #     np.conj(ctf),
    #     mode="full"
    # )
    # Better using the correlation theorem
    h_c, grid_out = coherent_impulse_response(
        grid,
        pupil,
        wavelength,
        distance,
        pad_factor
    )
    otf = fft2c(np.abs(h_c) ** 2)

    # Normalize to OTF(0, 0) = 1
    center = (otf.shape[0] // 2, otf.shape[1] // 2)
    otf /= otf[center]

    return otf, grid_out


def PSF(
        grid,
        pupil,
        wavelength,
        distance,
        pad_factor=1
):
    r"""Compute the incoherent impulse response or Point-Spread Function(PSF).

    The incoherent PSF is the squared modulus of the coherent
    amplitude impulse response:

    .. math::

        h_i = |h_c|^2.
    """

    h_c, grid_out = coherent_impulse_response(
        grid,
        pupil,
        wavelength,
        distance,
        pad_factor
    )

    psf = np.abs(h_c) ** 2

    # Normalize to unit peak
    psf /= np.max(psf)

    return psf, grid_out