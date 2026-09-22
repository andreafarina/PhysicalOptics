

import numpy as np


"""
Functions to generate the transmission of ideal optical apertures.

Each function returns the aperture transmission evaluated on the
provided Grid.
"""


def vertical_slit(grid, width, x0=0.0):
    """
    One-dimensional vertical slit.

    Parameters
    ----------
    grid : Grid
        Spatial grid.
    width : float
        Slit width [m].
    x0 : float, optional
        Slit center along x [m]. Default is 0.

    Returns
    -------
    ndarray
        Aperture transmission.
    """

    return (np.abs(grid.X - x0) <= width / 2).astype(float)

def horizontal_slit(grid, width, y0=0.0):
    """
    One-dimensional horizontal slit.

    Parameters
    ----------
    grid : Grid
        Spatial grid.
    width : float
        Slit width [m].
    y0 : float, optional
        Slit center along y [m]. Default is 0.

    Returns
    -------
    ndarray
        Aperture transmission.
    """

    return (np.abs(grid.Y - y0) <= width / 2).astype(float)


# --- Grating functions ---

def vertical_grating(grid, period, width, x0=0.0):
    """
    One-dimensional transmission grating with vertical slits.

    Parameters
    ----------
    grid : Grid
        Spatial grid.
    period : float
        Grating period [m].
    width : float
        Slit width [m].
    x0 : float, optional
        Grating center along x [m]. Default is 0.

    Returns
    -------
    ndarray
        Grating transmission.
    """

    x = np.mod(grid.X - x0 + period / 2, period) - period / 2
    return (np.abs(x) <= width / 2).astype(float)


def horizontal_grating(grid, period, width, y0=0.0):
    """
    One-dimensional transmission grating with horizontal slits.

    Parameters
    ----------
    grid : Grid
        Spatial grid.
    period : float
        Grating period [m].
    width : float
        Slit width [m].
    y0 : float, optional
        Grating center along y [m]. Default is 0.

    Returns
    -------
    ndarray
        Grating transmission.
    """

    y = np.mod(grid.Y - y0 + period / 2, period) - period / 2
    return (np.abs(y) <= width / 2).astype(float)


def square_grating(grid,
                   period_x,
                   width_x,
                   period_y=None,
                   width_y=None,
                   x0=0.0,
                   y0=0.0):
    """
    Two-dimensional square transmission grating.

    Parameters
    ----------
    grid : Grid
        Spatial grid.
    period_x : float
        Grating period along x [m].
    width_x : float
        Slit width along x [m].
    period_y : float, optional
        Grating period along y [m]. If None, period_x is used.
    width_y : float, optional
        Slit width along y [m]. If None, width_x is used.
    x0 : float, optional
        Grating center along x [m].
    y0 : float, optional
        Grating center along y [m].

    Returns
    -------
    ndarray
        Grating transmission.
    """

    if period_y is None:
        period_y = period_x

    if width_y is None:
        width_y = width_x

    return (
        vertical_grating(grid, period_x, width_x, x0)
        * horizontal_grating(grid, period_y, width_y, y0)
    )



def rectangular_aperture(grid, width, height, x0=0.0, y0=0.0):
    """
    Rectangular aperture.

    Parameters
    ----------
    grid : Grid
        Spatial grid.
    width : float
        Aperture width along x [m].
    height : float
        Aperture height along y [m].
    x0 : float, optional
        Aperture center along x [m]. Default is 0.
    y0 : float, optional
        Aperture center along y [m]. Default is 0.

    Returns
    -------
    ndarray
        Aperture transmission.
    """

    return (
        (np.abs(grid.X - x0) <= width / 2)
        &
        (np.abs(grid.Y - y0) <= height / 2)
    ).astype(float)

def abbe_porter_grating(grid,
                        period_x,
                        line_width_x,
                        period_y=None,
                        line_width_y=None,
                        x0=0.0,
                        y0=0.0):
    """
    Cross-line grating as commonly used in the Abbe-Porter experiment.

    The transmission consists of two orthogonal sets of transparent slits.
    The resulting pattern is a mesh rather than an array of isolated square
    apertures.
    """

    if period_y is None:
        period_y = period_x

    if line_width_y is None:
        line_width_y = line_width_x

    gx = vertical_grating(grid, period_x, line_width_x, x0)
    gy = horizontal_grating(grid, period_y, line_width_y, y0)


    return np.maximum(gx, gy)


# --- Sinusoidal grating ---
def sinusoidal_grating(grid,
                       frequency,
                       modulation=1.0,
                       offset=0.5,
                       angle=0.0,
                       x0=0.0,
                       y0=0.0):
    """
    Sinusoidal amplitude transmission grating.

    The grating transmission is sinusoidally modulated along an axis that is
    rotated by ``angle`` with respect to the x axis.

    Parameters
    ----------
    grid : Grid
        Spatial grid.
    frequency : float
        Grating frequency [m-1].
    modulation : float, optional
        Modulation depth. Default is 1.0.
        The transmission is given by
        ``offset + modulation / 2 * cos(...)``.
    offset : float, optional
        Mean transmission. Default is 0.5.
    angle : float, optional
        Grating orientation angle [rad], measured counterclockwise from the
        x axis. Default is 0.
    x0, y0 : float, optional
        Grating center [m]. Default is (0, 0).

    Returns
    -------
    ndarray
        Sinusoidal amplitude transmission.

    Notes
    -----
    For ``angle=0``, the transmission varies along x. The grating vector is
    therefore oriented along x, while the grating lines are parallel to y.
    The local coordinate normal to the grating lines is

        u = (x - x0) cos(angle) + (y - y0) sin(angle).

    The transmission is

        t(u) = offset + modulation / 2 * cos(2 pi frequency u ).
    """

    u = (
        (grid.X - x0) * np.cos(angle)
        + (grid.Y - y0) * np.sin(angle)
    )

    return offset + 0.5 * modulation * np.cos(2 * np.pi * u * frequency)


def circular_aperture(grid, radius, x0=0.0, y0=0.0):
    """
    Circular aperture.

    Parameters
    ----------
    grid : Grid
        Spatial grid.
    radius : float
        Aperture radius [m].
    x0 : float, optional
        Aperture center along x [m]. Default is 0.
    y0 : float, optional
        Aperture center along y [m]. Default is 0.

    Returns
    -------
    ndarray
        Aperture transmission.
    """

    return (
        (grid.X - x0) ** 2 + (grid.Y - y0) ** 2 <= radius ** 2
    ).astype(float)

def binary_mask(grid, filename, dx, size=None):
    """Load and scale a binary mask onto a spatial grid.

    The mask is read from a text file containing rows of ``0`` and ``1``.
    Each input pixel is expanded into a square physical pixel of side ``dx``.
    The mask can optionally be resized by an integer factor, e.g. from 8x8
    to 16x16, 32x32, etc., using nearest-neighbour replication so that the
    binary structure is preserved exactly.

    Parameters
    ----------
    grid : Grid
        Spatial grid on which the mask is evaluated.
    filename : str or pathlib.Path
        Path to the text file containing the binary mask. Each row must
        contain the same number of ``0`` and ``1`` entries.
    dx : float
        Physical size of one mask pixel [m].
    size : int, optional
        Desired number of pixels along each dimension. It must be a power of
        two and an integer multiple of the original mask size. If None, the
        original mask size is used.

    Returns
    -------
    ndarray
        Binary amplitude transmission sampled on ``grid``.

    Notes
    -----
    The physical size of the mask after resizing is

        size * dx.

    The mask is centered at the origin. Pixels are assigned using the
    coordinate of the pixel centre.
    """
    # Read the mask as individual binary pixels. This supports both
    # compact rows such as ``11100100`` and whitespace-separated values
    # such as ``1 1 1 0 0 1 0 0``.
    with open(filename, "r") as file:
        rows = [line.strip().replace(" ", "") for line in file if line.strip()]

    if not rows:
        raise ValueError("The binary mask file is empty")

    if any(set(row) - {"0", "1"} for row in rows):
        raise ValueError("The binary mask must contain only 0 and 1")

    ncols = len(rows[0])
    if any(len(row) != ncols for row in rows):
        raise ValueError("All rows of the binary mask must have the same length")

    mask = np.array([[int(value) for value in row] for row in rows], dtype=int)
    mask = np.flipud(mask)
    if mask.ndim != 2:
        raise ValueError("The binary mask must be a two-dimensional array")

    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError("The binary mask must contain only 0 and 1")

    nrows, ncols = mask.shape

    if nrows != ncols:
        raise ValueError("The binary mask must be square")

    if dx <= 0:
        raise ValueError("dx must be positive")

    if size is None:
        size = nrows

    if size < nrows or size % nrows != 0:
        raise ValueError(
            "size must be an integer multiple of the original mask size"
        )

    if size & (size - 1):
        raise ValueError("size must be a power of two")

    scale = size // nrows
    mask = np.repeat(np.repeat(mask, scale, axis=0), scale, axis=1)

    extent = size * dx
    x0 = -extent / 2
    y0 = -extent / 2

    ix = np.floor((grid.X - x0) / dx).astype(int)
    iy = np.floor((grid.Y - y0) / dx).astype(int)

    valid = (
        (ix >= 0) & (ix < size) &
        (iy >= 0) & (iy < size)
    )

    transmission = np.zeros_like(grid.X, dtype=float)
    transmission[valid] = mask[iy[valid], ix[valid]]

    return transmission


def fresnel_number(a, wavelength, distance):
    """
    Calculate the Fresnel number.

    Parameters
    ----------
    a : float
        Characteristic aperture size [m].
    wavelength : float
        Wavelength [m].
    distance : float
        Propagation distance [m].

    Returns
    -------
    float
        Fresnel number.
    """

    if a <= 0:
        raise ValueError("Aperture size must be positive.")

    if wavelength <= 0:
        raise ValueError("Wavelength must be positive.")

    if distance <= 0:
        raise ValueError("Distance must be positive.")

    return a ** 2 / (wavelength * distance)
