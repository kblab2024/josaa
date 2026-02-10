"""
Finite-element Green's function approach for critical-dimension metrology
of three-dimensional gratings on multilayer films.

Based on: Chang et al., J. Opt. Soc. Am. A, Vol. 23, No. 3, March 2006.
Errata corrections applied as documented in errata.md.

Optimized for Apple Silicon (M3) via NumPy/SciPy with Accelerate framework.
"""

from .transfer_matrix import compute_reflection_coefficients
from .greens_function import GreensFunctionTensor
from .solver import LippmannSchwinger
from .simulation import GratingSimulation
