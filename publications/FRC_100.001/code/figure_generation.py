"""
Figure Generation Code for FRC 100.001: "Fractal Resonance Cognition: A Framework for Complex Systems Analysis"

This script generates the three key figures used in the FRC 100.001 paper:
  1. Fractal Resonance Potential
  2. Perturbed Wavefunction
  3. Power-Law Correlation

Author: Hadi Servat
Copyright: © 2025 Hadi Servat, All Rights Reserved
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# Create figures directory if it doesn't exist
os.makedirs("figures", exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Figure 1: Fractal Resonance Potential
# This figure shows two forms of the fractal resonance potential V_FRC(x):
# (a) A simple oscillatory potential with a Gaussian envelope.
# (b) A Weierstrass-like fractal potential with self-similar oscillations.
x = np.linspace(-5, 5, 1000)  # Spatial coordinate, range [-5, 5] with 1000 points
sigma, k, beta, phi = 1.0, 10, 0.1, 0.0  # Parameters for simple V_FRC
# Simple potential: V_FRC(x) = sigma * cos(k*x + phi) * exp(-beta*x^2)
V_simple = sigma * np.cos(k * x + phi) * np.exp(-beta * x**2)

# Weierstrass-like potential: V_FRC(x) = sigma * sum_n [lambda^(-n*alpha) * cos(lambda^n * k * x)]
V_weierstrass = np.zeros_like(x)
lambda_, alpha, N = 2.0, 0.5, 5  # Parameters: lambda > 1, 0 < alpha < 1, N is number of terms
for n in range(N + 1):
    V_weierstrass += (lambda_**(-n * alpha)) * np.cos((lambda_**n) * k * x)

# Plotting Figure 1
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(x, V_simple, 'b-', label=r'$\sigma=1, k=10, \beta=0.1$')
plt.title('(a) Simple $V_{\mathrm{FRC}}$')
plt.xlabel('$x$')
plt.ylabel('$V_{\mathrm{FRC}}(x)$')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(x, V_weierstrass, 'r-', label=r'$\sigma=1, \lambda=2, \alpha=0.5$')
plt.title('(b) Weierstrass-like $V_{\mathrm{FRC}}$')
plt.xlabel('$x$')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/figure1_potential.png', dpi=300)
plt.close()

# Figure 2: Perturbed Wavefunction (Using Perturbation Theory)
# This figure compares the unperturbed and perturbed ground state wavefunctions of a 1D harmonic oscillator.
# Parameters are in arbitrary units for simplicity.
m, omega, hbar = 1.0, 1.0, 1.0  # Arbitrary units: mass, frequency, reduced Planck constant
# Unperturbed ground state: psi_0(x) = (m*omega/(pi*hbar))^0.25 * exp(-m*omega*x^2/(2*hbar))
psi_0 = (m * omega / (np.pi * hbar))**0.25 * np.exp(-m * omega * x**2 / (2 * hbar))
# Perturbation potential (same as simple V_FRC in Figure 1)
V_FRC = sigma * np.cos(k * x + phi) * np.exp(-beta * x**2)

# Compute first-order correction to the wavefunction using perturbation theory
# First-order correction: psi_1 = sum_{n!=0} |n><n|V|0> / (E_0 - E_n) * psi_n
# Approximate with the first excited state contribution (n=1) for simplicity
E_0 = hbar * omega * (0.5)  # Ground state energy
E_1 = hbar * omega * (1.5)  # First excited state energy
# First excited state: psi_1(x) = (m*omega/(pi*hbar))^0.25 * sqrt(2) * sqrt(m*omega/hbar) * x * exp(-m*omega*x^2/(2*hbar))
psi_1 = (m * omega / (np.pi * hbar))**0.25 * np.sqrt(2) * (np.sqrt(m * omega / hbar) * x) * np.exp(-m * omega * x**2 / (2 * hbar))
matrix_element = np.trapz(psi_1 * V_FRC * psi_0, x)  # <1|V|0>
coeff = matrix_element / (E_0 - E_1)  # Perturbation coefficient
psi_perturbed = psi_0 + coeff * psi_1  # First-order perturbed wavefunction

# Plotting Figure 2
plt.figure(figsize=(8, 4))
plt.plot(x, psi_0, 'k-', label='Unperturbed ($\sigma=0$)')
plt.plot(x, psi_perturbed, 'b--', label='Perturbed ($\sigma=1$)')
plt.xlabel('$x$')
plt.ylabel('$\psi_0(x)$')
plt.title('Ground State Wavefunction')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('figures/figure2_wavefunction.png', dpi=300)
plt.close()

# Figure 3: Power-Law Correlation with Fit
# This figure demonstrates a power-law correlation C(r) ~ r^(-alpha), a key signature of fractal resonance.
r = np.logspace(-1, 1, 100)  # Separation distance, logarithmic range [0.1, 10]
# Synthetic correlation: C(r) = r^(-0.9) with added noise
C_r = r**(-0.9) * (1 + 0.05 * np.random.randn(len(r)))

# Fit the power-law to estimate the exponent: C(r) = A * r^(-alpha)
def power_law(r, A, alpha):
    return A * r**(-alpha)
popt, pcov = curve_fit(power_law, r, C_r, p0=[1.0, 0.9])
A_fit, alpha_fit = popt
alpha_err = np.sqrt(pcov[1, 1])  # Standard error of alpha
C_fit = power_law(r, A_fit, alpha_fit)

# Plotting Figure 3
plt.figure(figsize=(6, 4))
plt.loglog(r, C_r, 'b.', label=f'$C(r) \\sim r^{{-{alpha_fit:.2f} \pm {alpha_err:.2f}}}$')
plt.loglog(r, C_fit, 'k-', alpha=0.5)
plt.xlabel('$r$')
plt.ylabel('$C(r)$')
plt.title('Power-Law Correlation')
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.savefig('figures/figure3_correlation.png', dpi=300)
plt.close()

print("Figure generation complete. All figures saved to the 'figures' directory.")
