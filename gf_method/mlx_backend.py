"""
MLX backend for GPU-accelerated array operations on Apple Silicon.

Provides a unified array operations interface that uses MLX when available
(for GPU acceleration on Apple Silicon) and falls back to NumPy otherwise.

MLX (https://github.com/ml-explore/mlx) is Apple's array framework for
machine learning on Apple Silicon, providing GPU acceleration via Metal.

Usage:
    from gf_method.mlx_backend import xp, use_mlx, to_numpy

    # xp is either mlx.core or numpy depending on availability
    a = xp.array([1.0, 2.0, 3.0])
    b = xp.exp(a)

    # Convert back to numpy for scipy interop
    b_np = to_numpy(b)
"""

import numpy as np

# Try to import MLX; fall back to NumPy if unavailable
_MLX_AVAILABLE = False
try:
    import mlx.core as mx
    # Verify MLX is functional (not just installed but broken)
    _test = mx.array([1.0])
    mx.eval(_test)
    _MLX_AVAILABLE = True
except Exception:
    pass

# Active backend: 'mlx' or 'numpy'
_backend = 'mlx' if _MLX_AVAILABLE else 'numpy'


def use_mlx():
    """Return True if MLX backend is active and available."""
    return _backend == 'mlx' and _MLX_AVAILABLE


def get_backend():
    """
    Return the active array module (mlx.core or numpy).

    Returns
    -------
    module
        mlx.core if MLX is active, numpy otherwise.
    """
    if use_mlx():
        return mx
    return np


def set_backend(backend_name):
    """
    Set the active backend.

    Parameters
    ----------
    backend_name : str
        'mlx' to use MLX (requires Apple Silicon), 'numpy' for NumPy.

    Raises
    ------
    ValueError
        If backend_name is not 'mlx' or 'numpy'.
    RuntimeError
        If 'mlx' is requested but not available.
    """
    global _backend
    if backend_name not in ('mlx', 'numpy'):
        raise ValueError(f"Unknown backend: {backend_name!r}. Use 'mlx' or 'numpy'.")
    if backend_name == 'mlx' and not _MLX_AVAILABLE:
        raise RuntimeError(
            "MLX is not available. Install mlx (pip install mlx) and ensure "
            "you are running on Apple Silicon."
        )
    _backend = backend_name


def to_numpy(arr):
    """
    Convert an array to a NumPy array.

    Parameters
    ----------
    arr : array-like
        Input array (MLX or NumPy).

    Returns
    -------
    np.ndarray
        NumPy array.
    """
    if _MLX_AVAILABLE and isinstance(arr, mx.array):
        return np.array(arr)
    return np.asarray(arr)


def to_backend(arr, dtype=None):
    """
    Convert a NumPy array to the active backend's array type.

    Parameters
    ----------
    arr : array-like
        Input array.
    dtype : dtype, optional
        Desired dtype. For MLX complex support, complex128 maps to
        complex64 (MLX's supported complex type).

    Returns
    -------
    array
        Array in the active backend format.
    """
    if use_mlx():
        np_arr = np.asarray(arr)
        if dtype is not None:
            np_arr = np_arr.astype(dtype)
        # MLX supports float32, float16, complex64
        # Map float64 -> float32, complex128 -> complex64
        if np_arr.dtype == np.float64:
            np_arr = np_arr.astype(np.float32)
        elif np_arr.dtype == np.complex128:
            np_arr = np_arr.astype(np.complex64)
        return mx.array(np_arr)
    arr = np.asarray(arr)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


# Convenience alias: xp is the active array module
xp = get_backend()
