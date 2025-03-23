# Code Appendix for FRC 100.002

This directory contains the Python code used to generate the figures in the FRC 100.002 paper: "Fractal Resonance Cognition in Quantum Chaos: Nodal Patterns in the Stadium Billiard."

## Requirements

The code requires the following libraries:
- NumPy (for numerical computations)
- Matplotlib (for plotting)
- SciPy (for sparse matrix operations and eigenvalue computations)

You can install them with:
```
pip install numpy matplotlib scipy
```

## Files

- `stadium_billiard.py` - The main Python script that generates all figures
- `figures/` - Directory containing the output figures

## Usage

To generate all figures, run:
```
python stadium_billiard.py
```

The figures will be saved in the `figures/` directory with a resolution of 300 DPI, suitable for publication.

## Description

The code generates two main figures:

1. **Figure 1: Nodal Patterns**
   - Shows the nodal patterns of the 50th eigenfunction in the stadium billiard
   - Compares the unperturbed case (σ=0) and the perturbed case with FRC potential (σ=0.1)
   - Reveals how fractal resonance induces self-similar clustering in the nodal lines

2. **Figure 2: Fractal Dimension Analysis**
   - Uses the box-counting method to compute the fractal dimension of the nodal lines
   - Shows log(N(ε)) vs log(1/ε) with a linear fit to determine the fractal dimension
   - Confirms that D ≈ 1.91 ± 0.03, consistent with FRC predictions

## Method Details

The code solves the modified Helmholtz equation for the stadium billiard:
- Uses finite difference method on a 200x200 grid
- Implements the 5-point stencil to approximate the Laplacian
- Applies boundary conditions by setting ψ = 0 outside the stadium
- Computes the eigenfunctions using SciPy's sparse eigenvalue solver
- Implements box-counting algorithm to compute fractal dimensions

## Notes

- For reproducibility, a random seed is set to ensure consistent results
- Parameters are in arbitrary units (ℏ = 1, m = 1) for simplicity
- The perturbation strength σ = 0.1 is chosen to be small enough to maintain the system's chaotic character but large enough to induce fractal patterns

© 2025 Hadi Servat, All Rights Reserved