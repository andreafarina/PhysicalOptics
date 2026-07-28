import numpy as np


class Field:
    """
    Scalar optical field.

    Parameters
    ----------
    grid : Grid
        Spatial grid.
    wavelength : float
        Wavelength [m].
    """

    def __init__(self, grid, wavelength):

        if wavelength <= 0:
            raise ValueError("Wavelength must be positive.")

        self.grid = grid
        self.wavelength = wavelength
        self.k = 2 * np.pi / wavelength

        # Complex field
        self.U = np.ones((grid.Ny, grid.Nx), dtype=complex)

    def copy(self):
        """
        Create a copy of the optical field.

        Returns
        -------
        Field
            A new Field object with the same grid, wavelength and a
            copy of the complex field.
        """

        field = Field(self.grid, self.wavelength)
        field.U = self.U.copy()
        return field


    def power(self):
        """
        Return the total optical intensity

            ∫∫ |U|² dx dy
        """
        return (
                np.sum(np.abs(self.U) ** 2)
                * self.grid.dx
                * self.grid.dy
        )