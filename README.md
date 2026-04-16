# Optical Diffraction Simulator

A clean, interactive Python application that simulates 2D optical diffraction
using Fourier optics — built with NumPy, Matplotlib, and PyQt5.

---

## Physics in Simple Terms

When light passes through a small opening (aperture), it doesn't travel in
a perfectly straight line — it **bends and spreads**. This is called
**diffraction**.

In the far field (Fraunhofer regime), the intensity pattern you'd observe on
a screen is mathematically identical to the **squared magnitude of the
Fourier Transform** of the aperture shape:

```
I(u, v) = | FFT{ A(x, y) } |²
```

Where:
- `A(x, y)` = aperture function — **1** where light passes through, **0** where blocked  
- `FFT`     = 2D Fast Fourier Transform  
- `I(u, v)` = intensity at position (u, v) in the focal / far-field plane

### Aperture patterns:
| Aperture       | Diffraction Pattern         |
|----------------|-----------------------------|
| Circular       | **Airy pattern** (rings)    |
| Rectangular    | 2D **sinc²** fringes        |
| Double slit    | Interference + diffraction  |

---

## Installation

```bash
pip install numpy matplotlib PyQt5
```

Then run:

```bash
python main.py
```

---

## Parameters

| Parameter        | Description |
|------------------|-------------|
| **Wavelength λ** | Colour of light (380–780 nm). Affects the diffraction scale conceptually — in this Fraunhofer model, changing λ while keeping aperture size fixed changes the angular width of diffraction rings. |
| **Aperture radius** | Physical size of the opening. **Smaller aperture → wider diffraction pattern** (inverse relationship). |
| **Grid N×N**     | Simulation resolution. Higher N = smoother, more detailed result, but slower. 256 is a good balance. |
| **Rect Width/Height** | Dimensions of rectangular aperture. Wider → narrower sinc fringes in that axis. |
| **Slit separation** | Distance between double-slit centres. Larger separation → more tightly-spaced interference fringes. |
| **Slit width**   | Width of each individual slit in double-slit mode. |
| **Log scale**    | Apply logarithmic stretching to reveal faint outer rings (the Airy rings are ~1000× dimmer than the central disc). |

---

## File Structure

```
diffraction_sim/
├── main.py         ← Entry point: starts the Qt application
├── simulation.py   ← Physics engine: aperture construction + FFT
├── utils.py        ← Helpers: grid creation, normalisation, log scale
├── gui.py          ← UI layout: control panel + embedded Matplotlib canvas
└── README.md       ← This file
```

---

## Features

- ✅ Circular, rectangular, and double-slit apertures  
- ✅ Interactive sliders for all physical parameters  
- ✅ 2D intensity heatmap with colourbar  
- ✅ Horizontal and vertical 1D cross-sections  
- ✅ Aperture preview panel  
- ✅ Logarithmic intensity scale toggle  
- ✅ Five colourmap options  
- ✅ Auto-update mode (no button press needed)  
- ✅ Save output image to PNG/JPEG  
- ✅ Dark, minimal GUI  

---

## Physical Intuition: Key Relationships

1. **Smaller aperture → broader diffraction** — this is the Heisenberg
   uncertainty principle in optics: narrow spatial localisation means wide
   angular spread.

2. **Airy disc radius** ≈ 1.22 λ/D where D is the aperture diameter.
   This is the resolution limit of telescopes and microscopes.

3. **Double-slit fringes** = envelope from single-slit diffraction ×
   cosine² interference term. Young's double-slit experiment in 2D!
