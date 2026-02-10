"""
Finite-element Green's function approach for critical-dimension metrology
of three-dimensional gratings on multilayer films.

Based on: Chang et al., J. Opt. Soc. Am. A, Vol. 23, No. 3, March 2006.
Errata corrections applied as documented in errata.md.

Supports GPU acceleration via MLX on Apple Silicon. Falls back to
NumPy/SciPy with Accelerate framework when MLX is not available.
"""

from .transfer_matrix import compute_reflection_coefficients
from .greens_function import GreensFunctionTensor
from .solver import LippmannSchwinger
from .simulation import GratingSimulation
from .mlx_backend import use_mlx, set_backend, get_backend
