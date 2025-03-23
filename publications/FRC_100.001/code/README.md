# Code Appendix for FRC 100.001

This directory contains the Python code used to generate the figures in the FRC 100.001 paper: "Fractal Resonance Cognition: A Framework for Complex Systems Analysis."

## Requirements

The code requires the following libraries:
- NumPy (for numerical computations)
- Matplotlib (for plotting)
- SciPy (for curve fitting)

You can install them with:
```
pip install numpy matplotlib scipy
```

## Files

- `figure_generation.py` - The main Python script that generates all figures
- `figures/` - Directory containing the output figures

## Usage

To generate all figures, run:
```
python figure_generation.py
```

The figures will be saved in the `figures/` directory with a resolution of 300 DPI, suitable for publication.

## Description

The code generates three main figures:

1. **Figure 1: Fractal Resonance Potential**
   - Shows two forms of the fractal resonance potential V_FRC(x):
     - A simple oscillatory potential with a Gaussian envelope
     - A Weierstrass-like fractal potential with self-similar oscillations

2. **Figure 2: Perturbed Wavefunction**
   - Compares the unperturbed and perturbed ground state wavefunctions of a 1D harmonic oscillator
   - Illustrates how the fractal resonance potential introduces oscillatory features

3. **Figure 3: Power-Law Correlation**
   - Demonstrates a power-law correlation C(r) ~ r^(-alpha), a key signature of fractal resonance
   - Includes a curve fit to estimate the power-law exponent

## Notes

- For reproducibility, a random seed is set to ensure consistent results
- Parameters in the simulations (e.g., m, ω, ℏ) are set to 1 for simplicity, representing arbitrary units

© 2025 Hadi Servat, All Rights Reserved