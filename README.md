# galsim-timeit
A Python package meant to be used with [GalSim](https://github.com/GalSim-developers/GalSim).

Intended to time different FFT and photon shooting routines.

## Installation

Clone the `GalSim` repo.
`git clone` inside the GalSim repo.
Run `pip install .`

To run in development mode, instead of `pip install .`, run 
```
pip install -e ./
```
This makes it so that all changes are reflected immediately and you do not have to keep running `pip install .` every time.

Read more here:
https://python-packaging-tutorial.readthedocs.io/en/latest/setup_py.html

## Usage
```python
from timer import Timer

```

