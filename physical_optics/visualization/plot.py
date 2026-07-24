import numpy as np
import matplotlib.pyplot as plt


"""
Visualization utilities for optical fields.
"""


def show_image(image, grid, domain="space", ax=None, title="", cmap=None):
    """Low-level visualization routine.

    Parameters
    ----------
    image : 2D array
        Image to display.
    grid : object
        Grid object with spatial and frequency coordinates.
    domain : str, optional
        Domain of the image, either "space" or "frequency".
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    title : str, optional
        Title of the plot.
    cmap : str or Colormap, optional
        Colormap to use.
    """
    if domain == "space":
        extent = [
            grid.x[0] * 1e3,
            grid.x[-1] * 1e3,
            grid.y[0] * 1e3,
            grid.y[-1] * 1e3,
        ]
        xlabel = "x [mm]"
        ylabel = "y [mm]"
    elif domain == "frequency":
        extent = [
            grid.fx[0] / 1e3,
            grid.fx[-1] / 1e3,
            grid.fy[0] / 1e3,
            grid.fy[-1] / 1e3,
        ]
        xlabel = "fx [cycles/mm]"
        ylabel = "fy [cycles/mm]"
    else:
        raise ValueError("domain must be 'space' or 'frequency'")

    if ax is None:
        fig, ax = plt.subplots()
    im = ax.imshow(
        image,
        extent=extent,
        origin="lower",
        aspect="equal",
        cmap=cmap,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.colorbar(im, ax=ax)


def show_amplitude(field, ax=None, title="Amplitude"):
    """Display the field amplitude."""

    show_image((np.abs(field.U)), field.grid, domain="space", ax=ax, title=title)


def show_intensity(field, log=True, ax=None, title="Intensity"):
    """Display the field intensity."""
    if log:
        image = 2*np.log10(np.abs(field.U) + 1e-12)
    else:
        image = np.abs(field.U) ** 2
    show_image(image ** 2, field.grid, domain="space", ax=ax, title=title)


def show_phase(field, ax=None, title="Phase [rad]"):
    """Display the field phase."""

    show_image(np.angle(field.U), field.grid, domain="space", ax=ax, title=title)


def show_spectrum(grid, spectrum, log=True, ax=None, title="Spectrum"):
    """Display the magnitude of a Fourier spectrum."""

    if log:
        image = np.log10(np.abs(spectrum) + 1e-12)
    else:
        image = np.abs(spectrum)
    show_image(image, grid, domain="frequency", ax=ax, title=title)
