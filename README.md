# Green's Function Method for 3D Grating CD Metrology

Implementation of the finite-element Green's function approach for critical-dimension
metrology of three-dimensional gratings on multilayer films, as described in:

> **Chang et al.**, "Efficient finite-element, Green's function approach for
> critical-dimension metrology of three-dimensional gratings on multilayer films,"
> *J. Opt. Soc. Am. A*, Vol. 23, No. 3, March 2006.

## Errata Corrections

The following mathematical errors documented in `errata.md` have been corrected
in this implementation:

1. **Gxz derivation (§2)**: The paper erroneously labels two lines as "Gxx = ..."
   in the Gxz derivation section. Corrected to "Gxz = ...".

2. **Gxz same-layer expression**: The last term was missing the ḡₙ(z') function
   factor, breaking the (z, z') symmetry. The missing factor has been restored.

3. **Eq. (14) — Gxz relation**: Corrected the relationship
   Gxz = (ikₙ/qₙ²) ∂z Gxx to be consistent with the corrected Gxz expression.

4. **Eq. (23) — TM decoupled equation**: The left-hand side erroneously reads
   "Hy − Ēz⁰(z)", mixing magnetic and electric field quantities. Corrected to
   "Hy(z) − Hy⁰(z)" for dimensional and physical consistency.

5. **qₙ² definition**: Confirmed and used the correct definition qₙ² ≡ kₙ² − k²
   throughout the implementation.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ with NumPy and SciPy.

### MLX GPU Acceleration (Apple Silicon)

On Apple Silicon Macs (M1/M2/M3/M4), the package supports GPU acceleration via
[MLX](https://github.com/ml-explore/mlx), Apple's array framework for machine
learning. MLX is automatically installed on Apple Silicon systems and provides
Metal GPU acceleration for array operations.

```bash
# MLX is included as a conditional dependency for Apple Silicon
pip install -r requirements.txt
```

When MLX is available, the following operations are GPU-accelerated:
- Batch Green's function tensor computation (`compute_full_tensor_batch`)
- Vectorized matrix-vector products in the GMRES solver
- Array operations (exp, sqrt, matrix multiply) via Metal GPU

When MLX is not available (e.g., on Linux/Windows), the package falls back
to NumPy/SciPy automatically with no code changes required.

#### Backend Selection

```python
from gf_method import use_mlx, set_backend, get_backend

# Check if MLX is active
print(f"MLX active: {use_mlx()}")

# Explicitly select backend
set_backend('numpy')   # Force NumPy backend
set_backend('mlx')     # Force MLX backend (requires Apple Silicon)

# Get the active array module
xp = get_backend()     # Returns mlx.core or numpy
```

## Usage

```python
from gf_method import ContactHoleSimulation

# Setup: 2D periodic array of contact holes in poly-Si on c-Si substrate
sim = ContactHoleSimulation(
    period_x=1000,        # nm, x-period
    period_y=1000,        # nm, y-period
    hole_diameter=307.5,  # nm
    hole_depth=409,       # nm
    epsilon_film=12.0 + 0.5j,      # poly-Si dielectric constant
    epsilon_substrate=12.0 + 0.5j,  # c-Si dielectric constant
    Nx=15, Ny=15          # plane waves per direction (N=225 total)
)

# Compute reflectivity vs wavelength
wavelengths = [400, 450, 500, 550, 600, 650, 700]  # nm
results = sim.compute_reflectivity(wavelengths, theta=0.0)

# Results contain:
#   results['r_TE']  - complex TE reflection amplitude
#   results['r_TM']  - complex TM reflection amplitude
#   results['R_TE']  - TE reflectivity |r|²
#   results['R_TM']  - TM reflectivity |r|²
```

## Module Structure

- **`gf_method/mlx_backend.py`** — MLX/NumPy backend abstraction for GPU
  acceleration on Apple Silicon via Metal
- **`gf_method/transfer_matrix.py`** — Transfer-matrix recursion for forward/backward
  reflection coefficients (Eqs. 8-11)
- **`gf_method/greens_function.py`** — Tensor Green's function Gxx, Gyy, Gxz, Gzx, Gzz
  in reciprocal space (§2, with errata corrections), including batch computation
- **`gf_method/solver.py`** — Lippmann-Schwinger equation solver with GMRES iteration,
  analytical segment integration, and precomputed batch matrices (§3, Eqs. 19-28)
- **`gf_method/cylindrical.py`** — Cylindrical grating support with Bessel/Struve
  function integrals (§4, Eq. 29)
- **`gf_method/simulation.py`** — High-level simulation driver for contact hole
  reflectivity calculations

## Testing

```bash
pytest tests/ -v
```

## Performance

The method scales as O(N log N) where N is the number of plane waves, compared to
O(N³) for RCWA. On Apple Silicon, MLX provides GPU acceleration via Metal for
vectorized array operations, batch Green's function computation, and matrix-vector
products. When MLX is not available, NumPy/SciPy with the Accelerate framework
is used for optimized BLAS/LAPACK operations.
