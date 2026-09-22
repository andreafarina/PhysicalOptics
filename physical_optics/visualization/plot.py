import numpy as np
import matplotlib.pyplot as plt


"""
Visualization utilities for optical fields.
"""

# -------------------------------------------------------------------------
# Private helper functions
# -------------------------------------------------------------------------

def _apply_zoom__(ax, grid, zoom):
    """
    Apply a visualization zoom by changing the axis limits.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    grid : Grid
        Grid associated with the field.
    zoom : float
        Zoom factor (1 = full view, 2 = half width, 4 = quarter width, ...).
    """

    if zoom <= 1:
        return

    Lx = grid.Nx * grid.dx
    Ly = grid.Ny * grid.dy

    ax.set_xlim(-Lx / (2 * zoom), Lx / (2 * zoom))
    ax.set_ylim(-Ly / (2 * zoom), Ly / (2 * zoom))

def _apply_zoom(ax, zoom):
    if zoom <= 1:
        return

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    xc = 0.5 * (xmin + xmax)
    yc = 0.5 * (ymin + ymax)

    hx = (xmax - xmin) / (2 * zoom)
    hy = (ymax - ymin) / (2 * zoom)

    ax.set_xlim(xc - hx, xc + hx)
    ax.set_ylim(yc - hy, yc + hy)

def show_image(image, grid, domain="space", ax=None, title="", cmap=None, zoom=1):
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
    _apply_zoom(ax, zoom)
    ax.set_title(title)
    plt.colorbar(im, ax=ax)

def show_surface(
    image,
    grid,
    domain="space",
    ax=None,
    title="",
    cmap="viridis",
    zoom = 1,
    ):
    if domain == "space":
        X = grid.X * 1e3
        Y = grid.Y * 1e3
        xlabel = "x [mm]"
        ylabel = "y [mm]"
    elif domain == "frequency":
        X = grid.FX / 1e3
        Y = grid.FY / 1e3
        xlabel = "fx [cycles/mm]"
        ylabel = "fy [cycles/mm]"
    else:
        raise ValueError("domain must be 'space' or 'frequency'")

    if ax is None:
        fig, ax = plt.subplots()
    ax.plot_surface(
        X,Y,
        image,
        cmap=cmap,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _apply_zoom(ax, zoom)

def show_amplitude(field, log=False,ax=None, title="Amplitude",zoom=1):
    """Display the field amplitude."""
    if log:
        image = np.log10(np.abs(field.U) + 1e-12)
    else:
        image = np.abs(field.U)
    show_image(image, field.grid, domain="space", ax=ax, title=title,zoom=zoom)


def show_intensity(field, log=True, ax=None, title="Intensity", zoom=1):
    """Display the field intensity."""
    if log:
        image = 2*np.log10(np.abs(field.U) + 1e-12)
    else:
        image = np.abs(field.U) ** 2
    show_image(image, field.grid, domain="space", ax=ax, title=title,zoom=zoom)


def show_phase(field, ax=None, title="Phase [rad]",zoom=1):
    """Display the field phase."""
    phase = np.angle(field.U)
    phase[np.abs(field.U) == 0] = np.nan #discard points where amplitude is zero
    show_image(phase, field.grid, domain="space", ax=ax, title=title, zoom=zoom)


def show_spectrum(grid, spectrum, log=True, ax=None, title="Spectrum",zoom=1):
    """Display the magnitude of a Fourier spectrum."""

    if log:
        image = np.log10(np.abs(spectrum) + 1e-12)
    else:
        image = np.abs(spectrum)
    show_image(image, grid, domain="frequency", ax=ax, title=title,zoom=zoom)

def show_lineplot(image, grid, direction="horizontal",
                  ax=None, title="", domain="space", **kwargs):
    """Plot the central horizontal or vertical line of a 2D image.

    Parameters
    ----------
    image : 2D array
        Image or 2D data array from which the line is extracted.
    grid : Grid
        Grid associated with ``image``.
    direction : {"horizontal", "vertical"}, optional
        Direction of the line. ``"horizontal"`` extracts the central row;
        ``"vertical"`` extracts the central column.
    ax : matplotlib.axes.Axes, optional
        Axes on which to draw the plot. If None, a new figure is created.
    title : str, optional
        Title of the plot.
    domain : {"space", "frequency"}, optional
        Domain of the data.
    **kwargs
        Additional arguments passed to ``ax.plot()``.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the line plot.
    """

    if domain not in ("space", "frequency"):
        raise ValueError("domain must be 'space' or 'frequency'")

    if direction == "horizontal":

        # Central row
        index = image.shape[0] // 2
        data = image[index, :]

        if domain == "space":
            coordinate = grid.x * 1e3
            xlabel = "x [mm]"
        else:
            coordinate = grid.fx / 1e3
            xlabel = "fx [cycles/mm]"

    elif direction == "vertical":

        # Central column
        index = image.shape[1] // 2
        data = image[:, index]

        if domain == "space":
            coordinate = grid.y * 1e3
            xlabel = "y [mm]"
        else:
            coordinate = grid.fy / 1e3
            xlabel = "fy [cycles/mm]"

    else:
        raise ValueError(
            "direction must be 'horizontal' or 'vertical'"
        )

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(coordinate, data, **kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.grid(True)

    return ax