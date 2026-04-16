"""
simulation.py — Optical Diffraction Physics Engine
====================================================
Implements scalar diffraction theory using Fourier optics.

Physics Summary:
    When light passes through a small aperture, it bends — this is diffraction.
    In the Fraunhofer (far-field) approximation, the diffracted intensity
    pattern is the squared magnitude of the Fourier Transform of the aperture.

    Mathematically:
        I(u, v) = |FFT{ A(x, y) }|²

    where:
        A(x, y) = aperture function (1 inside the opening, 0 outside)
        I(u, v) = intensity in the far-field / focal plane
        FFT     = 2D Fast Fourier Transform (via NumPy)
"""

import numpy as np
from utils import create_grid, normalize_intensity


class DiffractionSimulator:
    """
    Encapsulates the physics and computation of 2D scalar diffraction.

    Supported apertures:
        - Circular  (Airy pattern result)
        - Rectangular (sinc² pattern result)
        - Double slit (interference + diffraction)
    """

    APERTURE_TYPES = ["Circular", "Rectangular", "Double Slit"]

    def __init__(self):
        # Store the last computed results for reuse (e.g. cross-section plot)
        self.intensity = None       # 2D normalized intensity array
        self.x_axis = None          # Spatial coordinates along x
        self.y_axis = None          # Spatial coordinates along y
        self.params = {}            # Last used simulation parameters

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, wavelength_nm: float, aperture_radius: float,
            grid_N: int, aperture_type: str,
            rect_width: float = None, rect_height: float = None,
            slit_separation: float = None, slit_width: float = None) -> dict:
        """
        Run a full diffraction simulation.

        Parameters
        ----------
        wavelength_nm   : wavelength of light in nanometres (e.g. 550 nm)
        aperture_radius : radius/half-width of aperture in arbitrary units
        grid_N          : number of grid points per side (NxN grid)
        aperture_type   : one of 'Circular', 'Rectangular', 'Double Slit'
        rect_width      : width  for rectangular aperture (optional)
        rect_height     : height for rectangular aperture (optional)
        slit_separation : centre-to-centre distance for double slit (optional)
        slit_width      : width of each slit for double slit (optional)

        Returns
        -------
        dict with keys:
            'intensity'   : 2D numpy array, normalised [0, 1]
            'x_axis'      : 1D array of spatial x-coords (freq domain)
            'y_axis'      : 1D array of spatial y-coords (freq domain)
            'cross_x'     : 1D intensity slice along horizontal centre
            'cross_y'     : 1D intensity slice along vertical centre
            'aperture'    : 2D array of the aperture mask (for reference)
        """
        # 1. Build a symmetric spatial grid centred at origin
        x, y = create_grid(grid_N)

        # 2. Construct the aperture function A(x, y)
        aperture = self._build_aperture(
            x, y, aperture_type, aperture_radius,
            rect_width, rect_height,
            slit_separation, slit_width
        )

        # 3. Compute 2D FFT and shift zero-frequency to centre
        #    np.fft.fftshift moves the DC component from corner → centre
        fft_result = np.fft.fftshift(np.fft.fft2(aperture))

        # 4. Intensity = |FFT|²  (power spectrum)
        #    This is the physical observable — detectors measure intensity
        raw_intensity = np.abs(fft_result) ** 2

        # 5. Normalise so maximum = 1 (makes colour scaling consistent)
        intensity = normalize_intensity(raw_intensity)

        # 6. Build frequency-domain axes for labelled plots
        #    freq bins run from -N/2 to N/2 in units of 1/grid_spacing
        freq = np.fft.fftshift(np.fft.fftfreq(grid_N))
        x_axis = freq
        y_axis = freq

        # 7. Extract 1D cross-sections through the centre pixel
        centre = grid_N // 2
        cross_x = intensity[centre, :]   # horizontal slice
        cross_y = intensity[:, centre]   # vertical slice

        # Cache results
        self.intensity = intensity
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.params = {
            "wavelength_nm": wavelength_nm,
            "aperture_radius": aperture_radius,
            "grid_N": grid_N,
            "aperture_type": aperture_type,
        }

        return {
            "intensity": intensity,
            "x_axis": x_axis,
            "y_axis": y_axis,
            "cross_x": cross_x,
            "cross_y": cross_y,
            "aperture": aperture,
        }

    # ------------------------------------------------------------------
    # Private helpers — aperture construction
    # ------------------------------------------------------------------

    def _build_aperture(self, x, y, aperture_type: str,
                        radius: float,
                        rect_width, rect_height,
                        slit_sep, slit_width) -> np.ndarray:
        """
        Dispatch to the correct aperture builder.
        Returns a 2D float array: 1.0 inside opening, 0.0 outside.
        """
        if aperture_type == "Circular":
            return self._circular_aperture(x, y, radius)
        elif aperture_type == "Rectangular":
            w = rect_width  if rect_width  is not None else radius
            h = rect_height if rect_height is not None else radius
            return self._rectangular_aperture(x, y, w, h)
        elif aperture_type == "Double Slit":
            sep = slit_sep   if slit_sep   is not None else radius * 2
            sw  = slit_width if slit_width is not None else radius * 0.5
            return self._double_slit_aperture(x, y, sep, sw, radius)
        else:
            raise ValueError(f"Unknown aperture type: {aperture_type}")

    @staticmethod
    def _circular_aperture(x: np.ndarray, y: np.ndarray,
                           radius: float) -> np.ndarray:
        """
        Circular aperture — produces an Airy pattern after diffraction.

        A(x,y) = 1  if  sqrt(x² + y²) ≤ radius
                 0  otherwise
        """
        r_squared = x**2 + y**2   # vectorised distance² from centre
        return (r_squared <= radius**2).astype(float)

    @staticmethod
    def _rectangular_aperture(x: np.ndarray, y: np.ndarray,
                               width: float, height: float) -> np.ndarray:
        """
        Rectangular aperture — produces a 2D sinc² pattern.

        A(x,y) = 1  if  |x| ≤ width/2  AND  |y| ≤ height/2
                 0  otherwise
        """
        mask_x = np.abs(x) <= width  / 2.0
        mask_y = np.abs(y) <= height / 2.0
        return (mask_x & mask_y).astype(float)

    @staticmethod
    def _double_slit_aperture(x: np.ndarray, y: np.ndarray,
                               separation: float, slit_width: float,
                               slit_height: float) -> np.ndarray:
        """
        Double-slit aperture — interference fringes modulated by diffraction.

        Two thin vertical slits separated by `separation`, each of width
        `slit_width` and height `slit_height`.
        """
        half_sep = separation / 2.0
        half_w   = slit_width / 2.0

        # Left slit centred at  x = -half_sep
        left  = (np.abs(x + half_sep) <= half_w) & (np.abs(y) <= slit_height)
        # Right slit centred at x = +half_sep
        right = (np.abs(x - half_sep) <= half_w) & (np.abs(y) <= slit_height)

        return (left | right).astype(float)
