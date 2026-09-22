import numpy as np


class Grid:
    """
    Two-dimensional Cartesian grid.

    The origin is located at the center of the grid. Spatial frequencies
    follow the centered FFT convention.

    Parameters
    ----------
    Nx : int
        Number of samples along x.
    Ny : int
        Number of samples along y.
    dx : float
        Sampling interval along x [m].
    dy : float
        Sampling interval along y [m].
    """

    def __init__(self, Nx, Ny, dx, dy):
        if Nx <= 0 or Ny <= 0:
            raise ValueError("Nx and Ny must be positive.")

        if dx <= 0 or dy <= 0:
            raise ValueError("dx and dy must be positive.")

        self.Nx = Nx
        self.Ny = Ny

        self.dx = dx
        self.dy = dy

        # Grid size
        self.Lx = Nx * dx
        self.Ly = Ny * dy

        # Spatial coordinates
        self.x = (np.arange(Nx) - Nx / 2) * dx
        self.y = (np.arange(Ny) - Ny / 2) * dy

        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="xy")

        # Spatial frequencies
        self.fx = np.fft.fftshift(np.fft.fftfreq(Nx, d=dx))
        self.fy = np.fft.fftshift(np.fft.fftfreq(Ny, d=dy))

        self.FX, self.FY = np.meshgrid(self.fx, self.fy, indexing="xy")

        # Angular spatial frequencies
        self.kx = 2 * np.pi * self.fx
        self.ky = 2 * np.pi * self.fy

        self.KX, self.KY = np.meshgrid(self.kx, self.ky, indexing="xy")

    def fourier_to_space_grid(self, wavelength, z, pad_factor=1):
        """
        Return the output spatial grid associated with the scaled Fourier transform
        used by the Fresnel and Fraunhofer propagators.

        The output sampling is
            dx = λ z / (Nx dx)
            dy = λ z / (Ny dy)

        With zero-padding by ``pad_factor``, the output sampling interval is
        reduced by the same factor while the number of samples is increased
        by ``pad_factor``. The physical output extent is therefore increased
        by ``pad_factor``.

        Parameters
        ----------
        wavelength : float
            Wavelength [m].
        z : float
            Propagation distance [m].
        pad_factor : int, optional
            Zero-padding factor. ``pad_factor=1`` gives the original grid.
            Default is 1.

        Returns
        -------
        Grid
            Output spatial grid.
        """
        if not isinstance(pad_factor, int) or pad_factor < 1:
            raise ValueError("pad_factor must be a positive integer")

        Nx = pad_factor * self.Nx
        Ny = pad_factor * self.Ny

        dx = wavelength * z / (Nx * self.dx)
        dy = wavelength * z / (Ny * self.dy)

        return Grid(
            Nx=Nx,
            Ny=Ny,
            dx=dx,
            dy=dy,
        )

    def scaled_fourier_grid(self, wavelength, z, pad_factor=1):
        """
        Return the output spatial grid associated with the scaled Fourier transform
        used by the Fresnel and Fraunhofer propagators.

        The output sampling is
            dx = λ z / (Nx dx)
            dy = λ z / (Ny dy)

        With zero-padding by ``pad_factor``, the output sampling interval is
        reduced by the same factor while the number of samples is increased
        by ``pad_factor``. The physical output extent is therefore increased
        by ``pad_factor``.

        Parameters
        ----------
        wavelength : float
            Wavelength [m].
        z : float
            Propagation distance [m].
        pad_factor : int, optional
            Zero-padding factor. ``pad_factor=1`` gives the original grid.
            Default is 1.

        Returns
        -------
        Grid
            Output spatial grid.
        """
        if not isinstance(pad_factor, int) or pad_factor < 1:
            raise ValueError("pad_factor must be a positive integer")

        Nx = pad_factor * self.Nx
        Ny = pad_factor * self.Ny

        dx = wavelength * z / (Nx * self.dx)
        dy = wavelength * z / (Ny * self.dy)

        return Grid(
            Nx=Nx,
            Ny=Ny,
            dx=dx,
            dy=dy,
        )