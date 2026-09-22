"""Coherent imaging functions.

Functions in this module describe the coherent, isoplanatic imaging model
used in Fourier optics.  In particular, the coherent amplitude transfer
function (CTF) is identified with the pupil function expressed in spatial
frequency coordinates, while the coherent impulse response is its inverse
Fourier transform.
"""

import numpy as np
from physical_optics.common.grid import Grid
from physical_optics.common.fft import fft2c, ifft2c


def coherent_transfer_function(grid, pupil, wavelength, distance):
    r"""Return the coherent amplitude transfer function.

    For a coherent, isoplanatic optical system, the amplitude transfer
    function (ATF) or coherent transfer function (CTF) is the pupil function
    expressed as a function of scaled spatial frequency.  It therefore acts as a
    multiplicative filter on the spatial spectrum of the input field.

    Parameters
    ----------
    grid : Grid
        Spatial-frequency grid. The grid coordinates are used only to define
        the expected sampling of the pupil; the function currently returns
        the supplied pupil unchanged.
    pupil : ndarray
        Complex pupil function sampled on the spatial-frequency grid.

    Returns
    -------
    ndarray
        Coherent amplitude transfer function, equal to ``pupil``.

    Notes
    -----
    In the scalar coherent imaging model,

    .. math::

        \mathrm{CTF}(f_x,f_y) = P(-\lambda d_i f_x,-\lambda d_i f_y).

    The corresponding coherent impulse response is

    .. math::

        h(x,y) = \mathcal{F}^{-1}\{P(f_x,f_y)\}.

    The precise normalization and coordinate scaling depend on the Fourier
    transform convention and on how the pupil is parameterized.
    """
    h = pupil
    grid_out = grid.scaled_fourier_grid(wavelength, distance)
    return h, grid_out


def coherent_impulse_response(grid, pupil, wavelength, distance, pad_factor=1):
    r"""Compute the coherent impulse response from the pupil function.

    The coherent impulse response for an isoplanatic coherent
    imaging system is the Fourier transform of the pupil function
    expressed in spatial scaled frequency coordinates.

    Parameters
    ----------
    grid : Grid
        Spatial grid associated with ``pupil``. Its sampling is
        used to define the output spatial grid through the discrete Fourier
        transform convention adopted by the toolbox.
    pupil : ndarray
        Complex pupil function sampled on the spatial-frequency grid.
    wavelength : float
        Wavelength [m].
    distance : float
        Distance from the exit pupil to the image plane [m].
    pad_factor : int, optional
        Zero-padding factor applied to the pupil before the Fourier transform.
        ``pad_factor=1`` performs no padding. Larger values provide a finer
        sampling of the impulse response without changing the optical
        resolution. Default is 1.

    Returns
    -------
    h:  ndarray
        Complex coherent impulse response in the spatial domain.
    grid_out:  Grid
        Scaled Spatial grid associated with ``pupil``.

    Notes
    -----
    The continuous relation is

    .. math::

        \tilde{h(x,y)} = \lambda^{-2} d^{-2}_i \mathcal{F}\{P(x,y)\}.\\
        f_x = \frac{x}{\lambda d_i}, f_y = \frac{y}{\lambda d_i}

    Thus, for a circular pupil, the coherent impulse response is proportional
    to a jinc function.
    """
    if not isinstance(pad_factor, int) or pad_factor < 1:
        raise ValueError("pad_factor must be a positive integer")

    if pad_factor > 1:
        ny, nx = pupil.shape
        padded_shape = (pad_factor * ny, pad_factor * nx)
        pupil_padded = np.zeros(padded_shape, dtype=pupil.dtype)
        y0 = (padded_shape[0] - ny) // 2
        x0 = (padded_shape[1] - nx) // 2
        pupil_padded[y0:y0 + ny, x0:x0 + nx] = pupil
    else:
        pupil_padded = pupil

    h = wavelength ** (-2) * distance ** (-2) * fft2c(pupil_padded)

    grid_out = grid.scaled_fourier_grid(wavelength, distance, pad_factor)

    return h, grid_out