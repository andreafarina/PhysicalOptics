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

    def scaled_fourier_grid(self, wavelength, z):
        """
        Return the output spatial grid associated with the scaled Fourier transform
        used by the Fresnel and Fraunhofer propagators.

        The output sampling is
            dx = λ z / (Nx dx)
            dy = λ z / (Ny dy)

        Returns
        -------
        GridO
            utput spatial grid.
    """

        dx = wavelength * z / (self.Nx * self.dx)
        dy = wavelength * z / (self.Ny * self.dy)

        return Grid(
            Nx=self.Nx,
            Ny=self.Ny,
            dx=dx,
            dy=dy,
        )