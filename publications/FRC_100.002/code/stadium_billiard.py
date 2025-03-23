"""
Figure Generation Code for FRC 100.002: "Fractal Resonance Cognition in Quantum Chaos: 
Nodal Patterns in the Stadium Billiard"

This script generates the three key figures used in the FRC 100.002 paper:
  1. Nodal Patterns - Comparing unperturbed and perturbed stadium billiard
  2. Fractal Dimension Analysis - Box-counting method to compute fractal dimension
  3. FRC Potential Visualization - Visualizing the fractal resonance potential

Author: Hadi Servat
Copyright: © 2025 Hadi Servat, All Rights Reserved
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, lil_matrix
from scipy.sparse.linalg import eigs
import os

# Create figures directory if it doesn't exist
figures_path = "../figures"
os.makedirs(figures_path, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Define the stadium billiard geometry
L, R = 2.0, 1.0  # Length of straight sides and radius of semicircular ends
nx, ny = 150, 150  # Increased grid points to 150x150 for better resolution
x = np.linspace(-L/2 - R, L/2 + R, nx)  # x-coordinate range
y = np.linspace(-R, R, ny)  # y-coordinate range
dx, dy = x[1] - x[0], y[1] - y[0]  # Grid spacing
X, Y = np.meshgrid(x, y)  # 2D grid

# Create a mask for the stadium billiard domain
print("Creating stadium billiard mask...")
mask = np.zeros((ny, nx), dtype=bool)
for i in range(ny):
    for j in range(nx):
        if abs(Y[i, j]) <= R:  # Within the straight sides
            if abs(X[i, j]) <= L/2:
                mask[i, j] = True
            elif abs(X[i, j]) <= L/2 + R:  # Within the semicircular ends
                if (X[i, j] - L/2)**2 + Y[i, j]**2 <= R**2 or (X[i, j] + L/2)**2 + Y[i, j]**2 <= R**2:
                    mask[i, j] = True

# Flatten the mask for finite difference method
mask_flat = mask.flatten()

# Define the FRC potential: V_FRC(x,y) = sigma * sum_n [lambda^(-n*alpha) * cos(lambda^n * kx * x) * cos(lambda^n * ky * y)]
print("Computing FRC potential...")
sigma, lambda_, alpha, N, kx, ky = 0.1, 2.0, 0.6, 4, 8.0, 12.0  # Updated parameters
V_FRC = np.zeros((ny, nx))
for n in range(N + 1):
    V_FRC += (lambda_**(-n * alpha)) * np.cos((lambda_**n) * kx * X) * np.cos((lambda_**n) * ky * Y)
V_FRC_flat = V_FRC.flatten()

# Finite difference Laplacian (5-point stencil) for 2D Helmholtz equation
# -nabla^2 psi + V psi = k^2 psi
print("Setting up Laplacian operator...")
N_total = nx * ny
diagonals = [np.ones(N_total), -4 * np.ones(N_total), np.ones(N_total)]
offsets = [-nx, 0, nx]
D2x = diags(diagonals, offsets, shape=(N_total, N_total)) / dx**2  # Second derivative in x

# Create D2y and handle boundary connections
diagonals_y = [np.ones(N_total - 1), -2 * np.ones(N_total), np.ones(N_total - 1)]
offsets_y = [-1, 0, 1]
D2y = diags(diagonals_y, offsets_y, shape=(N_total, N_total)) / dy**2

# Convert D2y to lil_matrix to allow item assignment
D2y = D2y.tolil()
for i in range(1, ny):
    D2y[i*nx - 1, i*nx] = 0  # Remove connections across x-boundaries
    D2y[i*nx, i*nx - 1] = 0
D2y = D2y.tocsr()  # Convert back to CSR format for efficiency

Laplacian = -(D2x + D2y)

# Apply boundary conditions by setting Laplacian to identity outside the domain
Laplacian = Laplacian.tolil()
for i in range(N_total):
    if not mask_flat[i]:
        Laplacian[i, :] = 0
        Laplacian[i, i] = 1
Laplacian = Laplacian.tocsr()

# Solve for eigenfunctions: unperturbed case (sigma = 0)
print("Computing unperturbed eigenvalues...")
H_unperturbed = Laplacian.copy()
k2_unperturbed, psi_unperturbed = eigs(H_unperturbed, k=20, which='SM')  # Compute 20 eigenvalues
print("Unperturbed eigenvalues computed.")
psi_15_unperturbed = np.real(psi_unperturbed[:, 15].reshape(ny, nx))  # 15th eigenfunction
psi_15_unperturbed[~mask] = 0  # Enforce boundary conditions

# Solve for eigenfunctions: perturbed case (with FRC potential)
print("Computing perturbed eigenvalues...")
V_FRC_flat[~mask_flat] = 0  # Set potential to 0 outside the domain
H_perturbed = Laplacian + sigma * diags(V_FRC_flat, 0)
k2_perturbed, psi_perturbed = eigs(H_perturbed, k=20, which='SM')
print("Perturbed eigenvalues computed.")
psi_15_perturbed = np.real(psi_perturbed[:, 15].reshape(ny, nx))
psi_15_perturbed[~mask] = 0

# Figure 1: Nodal Patterns
# Plot the nodal lines (where psi = 0) for unperturbed and perturbed cases
print("Generating Figure 1: Nodal Patterns...")
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.contour(X, Y, psi_15_unperturbed, levels=[0], colors='k')
plt.title('(a) Unperturbed ($\\sigma=0$)')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.gca().set_aspect('equal')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.contour(X, Y, psi_15_perturbed, levels=[0], colors='k')
plt.title('(b) Perturbed ($\\sigma=0.1$)')
plt.xlabel('$x$')
plt.gca().set_aspect('equal')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(figures_path, 'figure1_nodal_patterns.png'), dpi=300)
plt.close()

# Figure 2: Fractal Dimension Analysis (Box-Counting Method)
# Compute the fractal dimension of the nodal lines in the perturbed case
print("Computing fractal dimension using box-counting method...")
def box_counting(psi, mask, epsilons):
    """Computes the box-counting dimension of the nodal lines in a wavefunction."""
    N_eps = []
    for epsilon in epsilons:
        # Downsample the grid to boxes of size epsilon
        step = max(1, int(epsilon / dx))
        N = 0
        for i in range(0, ny, step):
            for j in range(0, nx, step):
                # Check if the box contains a nodal line (psi changes sign)
                if i+step <= ny and j+step <= nx:  # Ensure within bounds
                    sub_psi = psi[i:i+step, j:j+step]
                    sub_mask = mask[i:i+step, j:j+step]
                    if np.any(sub_mask) and np.any(sub_psi > 0) and np.any(sub_psi < 0):
                        N += 1
        N_eps.append(N)
    return N_eps

# Define box sizes (epsilons) for box-counting
epsilons = np.logspace(np.log10(dx*2), np.log10((L/2 + R)/4), 15)  # Adjusted range
N_eps_unperturbed = box_counting(psi_15_unperturbed, mask, epsilons)
N_eps_perturbed = box_counting(psi_15_perturbed, mask, epsilons)

# Filter out zero values to avoid log(0)
valid_indices_unperturbed = np.where(np.array(N_eps_unperturbed) > 0)[0]
valid_indices_perturbed = np.where(np.array(N_eps_perturbed) > 0)[0]

epsilons_unperturbed = np.array(epsilons)[valid_indices_unperturbed]
N_eps_unperturbed = np.array(N_eps_unperturbed)[valid_indices_unperturbed]

epsilons_perturbed = np.array(epsilons)[valid_indices_perturbed]
N_eps_perturbed = np.array(N_eps_perturbed)[valid_indices_perturbed]

# Compute fractal dimensions
log_eps_unperturbed = np.log(1/epsilons_unperturbed)
log_N_unperturbed = np.log(N_eps_unperturbed)
coeffs_unperturbed = np.polyfit(log_eps_unperturbed, log_N_unperturbed, 1)
D_unperturbed = coeffs_unperturbed[0]  # Fractal dimension is the slope

log_eps_perturbed = np.log(1/epsilons_perturbed)
log_N_perturbed = np.log(N_eps_perturbed)
coeffs_perturbed = np.polyfit(log_eps_perturbed, log_N_perturbed, 1)
D_perturbed = coeffs_perturbed[0]

# Bootstrap error estimation
bootstrap_D = []
for _ in range(10):
    indices = np.random.choice(len(log_eps_perturbed), len(log_eps_perturbed), replace=True)
    coeffs_boot = np.polyfit(log_eps_perturbed[indices], log_N_perturbed[indices], 1)
    bootstrap_D.append(coeffs_boot[0])
D_err = np.std(bootstrap_D)

print(f"Fractal dimension of unperturbed system: D ≈ {D_unperturbed:.2f}")
print(f"Fractal dimension of perturbed system: D ≈ {D_perturbed:.2f} ± {D_err:.2f}")

# Plotting Figure 2
print("Generating Figure 2: Fractal Dimension Analysis...")
plt.figure(figsize=(6, 4))
plt.plot(log_eps_perturbed, log_N_perturbed, 'b.', label=f'Perturbed, $D = {D_perturbed:.2f} \\pm {D_err:.2f}$')
plt.plot(log_eps_perturbed, np.polyval(coeffs_perturbed, log_eps_perturbed), 'b-', alpha=0.5)
plt.plot(log_eps_unperturbed, log_N_unperturbed, 'r.', label=f'Unperturbed, $D = {D_unperturbed:.2f}$')
plt.plot(log_eps_unperturbed, np.polyval(coeffs_unperturbed, log_eps_unperturbed), 'r-', alpha=0.5)
plt.xlabel('$\\log(1/\\epsilon)$')
plt.ylabel('$\\log N(\\epsilon)$')
plt.title('Box-Counting Analysis')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(figures_path, 'figure2_fractal_dimension.png'), dpi=300)
plt.close()

# Figure 3: FRC Potential Visualization
# Plot the FRC potential V_FRC(x,y) within the stadium billiard domain
print("Generating Figure 3: FRC Potential Visualization...")
plt.figure(figsize=(6, 4))
V_FRC_masked = V_FRC.copy()
V_FRC_masked[~mask] = np.nan  # Set potential to NaN outside the domain for visualization
plt.contourf(X, Y, V_FRC_masked, cmap='viridis', levels=20)
plt.colorbar(label='$V_{\\mathrm{FRC}}(x,y)$')
plt.title('Fractal Resonance Potential')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.gca().set_aspect('equal')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(figures_path, 'figure3_frc_potential.png'), dpi=300)
plt.close()

print(f"Figure generation complete. All figures saved to the '{figures_path}' directory.")
