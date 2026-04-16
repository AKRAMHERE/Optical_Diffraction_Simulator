"""
utils.py — Helper Functions for the Diffraction Simulator
==========================================================
Contains low-level utilities for grid creation, normalization,
and other reusable operations that don't belong in the physics
or GUI layer.
"""

import numpy as np


def create_grid(N: int, extent: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a 2D symmetric spatial grid of size N×N.

    The grid spans from -extent to +extent in both x and y.
    Uses meshgrid so that every (i, j) index maps to a (x, y) coordinate.

    Parameters
    ----------
    N      : number of grid points per side
    extent : half-width of the spatial domain (default 1.0)

    Returns
    -------
    (x, y) : two 2D numpy arrays of shape (N, N)
              x[i, j] = x-coordinate of grid point (i, j)
              y[i, j] = y-coordinate of grid point (i, j)

    Example
    -------
    For N=4, extent=1:
        coords = [-1, -0.33, 0.33, 1]
        x, y = meshgrid(coords, coords)
    """
    coords = np.linspace(-extent, extent, N)
    x, y = np.meshgrid(coords, coords)
    return x, y


def normalize_intensity(intensity: np.ndarray) -> np.ndarray:
    """
    Normalise a 2D intensity array so its maximum value equals 1.0.

    This is necessary because raw FFT output magnitudes depend on
    grid size and aperture area — normalisation makes the colour
    scale consistent across different parameter choices.

    Parameters
    ----------
    intensity : 2D numpy array of non-negative floats

    Returns
    -------
    Normalised 2D array in range [0, 1]
    """
    peak = intensity.max()
    if peak == 0:
        # Degenerate case: aperture is fully closed → return zeros
        return intensity
    return intensity / peak


def apply_log_scale(intensity: np.ndarray,
                    epsilon: float = 1e-6) -> np.ndarray:
    """
    Apply a logarithmic stretch to reveal faint diffraction rings.

    The log transform compresses the huge dynamic range of diffraction
    patterns (the central peak is often 10⁶× brighter than outer rings).

    I_log = log(1 + I) / log(1 + I_max)   (normalised to [0,1])

    Parameters
    ----------
    intensity : 2D array, values in [0, 1]
    epsilon   : small floor to avoid log(0)

    Returns
    -------
    Log-scaled array in [0, 1]
    """
    clipped = np.clip(intensity, epsilon, None)
    log_vals = np.log10(clipped + 1.0)
    return normalize_intensity(log_vals)


def ring_stats(intensity: np.ndarray) -> dict:
    """
    Compute basic statistics on the diffraction pattern for display.

    Parameters
    ----------
    intensity : normalised 2D intensity array

    Returns
    -------
    dict with keys:
        'peak'   : peak intensity (always 1.0 after normalisation)
        'mean'   : mean intensity
        'energy' : total energy (sum of all values, proxy for aperture area)
    """
    return {
        "peak":   float(intensity.max()),
        "mean":   float(intensity.mean()),
        "energy": float(intensity.sum()),
    }
