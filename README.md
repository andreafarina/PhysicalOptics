# PhysicalOptics

A Python library for teaching and experimenting with scalar physical optics.

PhysicalOptics provides a simple and educational framework for simulating
wave propagation, diffraction, interference and imaging using numerical
methods commonly employed in Fourier optics.

The library is designed for undergraduate and graduate courses in Physical
Optics and follows the notation of Goodman's *Introduction to Fourier Optics*.

## Features

- Simple Grid and Field classes
- Centered FFT utilities
- Optical elements
  - Apertures
  - Thin lenses
- Scalar diffraction propagators
  - Fraunhofer
  - Angular Spectrum
  - Fresnel
  - Rayleigh-Sommerfeld (planned)
- Visualization utilities
- Educational examples

## Installation

Clone the repository

```bash
git clone https://github.com/<username>/PhysicalOptics.git
```

Create an environment

```bash
pip install numpy matplotlib
```

(or install using your preferred environment manager)

## Quick example

```python
from physical_optics.common.grid import Grid
from physical_optics.common.field import Field

from physical_optics.objects.apertures import circular_aperture
from physical_optics.diffraction import fraunhofer
from physical_optics.visualization.plot import show_intensity

grid = Grid(1024, 1024, 2e-6, 2e-6)

field = Field(grid, wavelength=633e-9)

field.U *= circular_aperture(grid, radius=50e-6)

field_out = fraunhofer.propagate(field, z=0.5)

show_intensity(field_out)
```

## Philosophy

This library prioritizes:

- readability over optimization
- explicit physics over hidden implementations
- educational value over abstraction

Whenever possible, the physical building blocks of an algorithm are exposed as
public functions rather than hidden inside a propagator.

## Current modules

```
physical_optics
├── common
├── diffraction
├── imaging
├── objects
└── visualization
```

## Roadmap

- [x] Core classes
- [x] Angular Spectrum propagation
- [x] Fraunhofer propagation
- [x] Fresnel propagation
- [x] Rayleigh-Sommerfeld propagation
- [x] Imaging systems
- [ ] Jupyter notebooks
- [ ] Documentation

## References

J. W. Goodman,
*Introduction to Fourier Optics*,
4th Edition,
W. H. Freeman.

## License

MIT License.
