

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