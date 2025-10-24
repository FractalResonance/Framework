import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp, zeta, zetazero, gamma, pi
from multiprocessing import Pool, cpu_count

# Precision and defaults
mp.dps = 50

# FRC 567.901 core mappings
# S(s) = k_* ln |ζ(s)|,  C(s) = |ζ(s)|^{-2}


def get_zeros(n: int,
              cache_path: str = "artifacts/riemann_567901/known_zeros.npy",
              prefer_bulk: str = "artifacts/riemann_567901/known_zeros_5000.npy") -> np.ndarray:
    """Load first n zeros from cache, prefer a bulk 5000-file if available; otherwise compute and cache.

    This avoids slow on-the-fly root-finding when a larger precomputed set exists.
    """
    # Prefer a precomputed 5000-zeros file when present
    if prefer_bulk and os.path.exists(prefer_bulk):
        try:
            bulk = np.load(prefer_bulk)
            if bulk.size >= n:
                return bulk[:n].astype(float)
        except Exception:
            pass

    zeros = None
    if os.path.exists(cache_path):
        try:
            cached = np.load(cache_path)
            if cached.size >= n:
                return cached[:n]
            zeros = list(cached.astype(float))
        except Exception:
            zeros = []
    else:
        zeros = []

    # Compute remaining zeros incrementally (can be slow for large n)
    start_k = len(zeros) + 1
    for k in range(start_k, n + 1):
        t = float(zetazero(k).imag)
        zeros.append(t)
        if k % 50 == 0:
            np.save(cache_path, np.array(zeros, dtype=np.float64))
    np.save(cache_path, np.array(zeros, dtype=np.float64))
    return np.array(zeros, dtype=np.float64)


def zeta_abs(sigma: float, tau: float) -> float:
    """|ζ(σ + iτ)| as float."""
    val = zeta(mp.mpc(sigma, tau))
    return float(abs(val))

def xi_abs(sigma: float, tau: float) -> float:
    """|ξ(s)| as float, where ξ(s) = 1/2 s(s-1) π^{-s/2} Γ(s/2) ζ(s)."""
    s = mp.mpc(sigma, tau)
    xi = 0.5 * s * (s - 1) * (pi ** (-s / 2)) * gamma(s / 2) * zeta(s)
    return float(abs(xi))


def compute_maps(sigma_min: float, sigma_max: float, tau_center: float,
                 tau_halfspan: float, sigma_points: int, tau_points: int,
                 k_star: float = 1.0, eps: float = 0.0, sigma_offset: float = 0.0):
    sigmas = np.linspace(sigma_min, sigma_max, sigma_points)
    taus = np.linspace(tau_center - tau_halfspan, tau_center + tau_halfspan, tau_points)
    Z = np.empty((tau_points, sigma_points), dtype=np.float64)
    for i, tau in enumerate(taus):
        for j, sig in enumerate(sigmas):
            # Positive control: evaluate field at shifted sigma if requested
            Z[i, j] = zeta_abs(sig - sigma_offset, tau)
    # Entropy / coherence fields
    S = k_star * np.log(Z + 1e-30)
    C = 1.0 / (Z * Z + max(0.0, eps))
    return sigmas, taus, Z, S, C


def plot_heatmap(sigmas, taus, field, title, fname, critical_sigma=0.5, zero_tau=None, cmap="magma"):
    plt.figure(figsize=(8, 5))
    extent = [sigmas[0], sigmas[-1], taus[0], taus[-1]]
    plt.imshow(field, origin='lower', aspect='auto', extent=extent, cmap=cmap)
    plt.colorbar(label=title)
    plt.axvline(critical_sigma, color='cyan', lw=1.2, ls='--', label='Re(s)=1/2')
    if zero_tau is not None:
        plt.axhline(zero_tau, color='lime', lw=1.0, ls=':')
    plt.xlabel('σ = Re(s)')
    plt.ylabel('τ = Im(s)')
    plt.title(title)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(fname, dpi=160)
    plt.close()


def plot_slice(sigmas, Z_line, title, fname, critical_sigma=0.5):
    plt.figure(figsize=(7, 4))
    plt.plot(sigmas, Z_line, label='|ζ(σ + iτ₀)|')
    plt.axvline(critical_sigma, color='cyan', lw=1.2, ls='--', label='Re(s)=1/2')
    plt.xlabel('σ')
    plt.ylabel('|ζ|')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=160)
    plt.close()


def _per_zero_task(args_pack):
    (idx, tau0, args) = args_pack
    sigmas, taus, Z, S, C = compute_maps(
        args.sigma_min, args.sigma_max, tau0, args.tau_span,
        args.sigma_points, args.tau_points, k_star=args.k_star,
        eps=args.eps, sigma_offset=args.sigma_offset)

    # Save plots
    plot_heatmap(sigmas, taus, S, f'S(σ, τ) near zero #{idx+1} (τ≈{tau0:.2f})',
                 f'artifacts/riemann_567901/S_heatmap_zero_{idx+1:04d}.png', zero_tau=tau0)
    plot_heatmap(sigmas, taus, C, f'C(σ, τ) near zero #{idx+1} (τ≈{tau0:.2f})',
                 f'artifacts/riemann_567901/C_heatmap_zero_{idx+1:04d}.png', zero_tau=tau0, cmap='viridis')

    # Slice at τ = τ0
    j_mid = int(np.argmin(np.abs(taus - tau0)))
    plot_slice(sigmas, Z[j_mid, :], f'|ζ| slice at τ≈{tau0:.2f} (zero #{idx+1})',
               f'artifacts/riemann_567901/zeta_slice_zero_{idx+1:04d}.png')

    # Argmax σ of coherence per τ row (maximization diagnostic)
    argmax_idx = np.argmax(C, axis=1)
    sigma_argmax_per_tau = sigmas[argmax_idx]

    # Sub-grid quadratic fit around argmax (when neighbors exist)
    sigma_hat = np.full_like(sigma_argmax_per_tau, np.nan, dtype=float)
    for r, j in enumerate(argmax_idx):
        if 0 < j < (len(sigmas)-1):
            x1, x2, x3 = sigmas[j-1], sigmas[j], sigmas[j+1]
            y1, y2, y3 = C[r, j-1], C[r, j], C[r, j+1]
            denom = (x1 - x2)*(x1 - x3)*(x2 - x3)
            if denom != 0.0:
                a = (x3*(y2 - y1) + x2*(y1 - y3) + x1*(y3 - y2)) / denom
                b = (x3*x3*(y1 - y2) + x2*x2*(y3 - y1) + x1*x1*(y2 - y3)) / denom
                if a < 0:
                    sigma_hat[r] = float(-b/(2*a))
    # Derivative-zero stationary test per τ row (maxima via sign change and negative curvature)
    def stationary_sigma(row_c: np.ndarray) -> float:
        # central diffs for first derivative
        d = np.zeros_like(row_c)
        d[1:-1] = (row_c[2:] - row_c[:-2]) / (sigmas[2:] - sigmas[:-2])
        # second derivative
        d2 = np.zeros_like(row_c)
        h = sigmas[1] - sigmas[0]
        d2[1:-1] = (row_c[2:] - 2*row_c[1:-1] + row_c[:-2]) / (h*h)
        # sign changes of d around zero; maxima have d2<0
        sc = d[:-1] * d[1:] <= 0
        cand = np.where(sc)[0]
        if cand.size == 0:
            return np.nan
        # refine each crossing by linear interpolation of derivative
        picks = []
        for j in cand:
            j1, j2 = j, j+1
            denom = (d[j2] - d[j1])
            if denom == 0:
                sstar = sigmas[j1]
            else:
                sstar = sigmas[j1] - d[j1] * (sigmas[j2]-sigmas[j1]) / denom
            # evaluate curvature at nearest grid point
            jj = j1 if abs(sigmas[j1]-sstar) < abs(sigmas[j2]-sstar) else j2
            if d2[jj] < 0:
                picks.append(sstar)
        if not picks:
            return np.nan
        # choose the stationary point closest to 0.5
        picks = np.array(picks, dtype=float)
        return float(picks[np.argmin(np.abs(picks - 0.5))])

    sigma_stationary = np.array([stationary_sigma(C[r, :]) for r in range(C.shape[0])], dtype=float)

    return {
        'idx': idx,
        'tau0': float(tau0),
        'sigmas': sigmas,
        'taus': taus,
        'sigma_argmax_per_tau': sigma_argmax_per_tau,
        'sigma_hat_per_tau': sigma_hat,
        'sigma_stationary_per_tau': sigma_stationary,
    }


def main():
    parser = argparse.ArgumentParser(description='FRC 567.901 — Entropy/Coherence maps around ζ zeros')
    parser.add_argument('--zeros', type=int, default=1000, help='number of zeros to cache/use (default 1000; supports 5000)')
    parser.add_argument('--plots', type=int, default=20, help='number of zero-neighborhood plots to generate')
    parser.add_argument('--skip-plots', action='store_true', help='only cache/load zeros and exit')
    parser.add_argument('--sigma-min', type=float, default=0.35)
    parser.add_argument('--sigma-max', type=float, default=0.65)
    parser.add_argument('--tau-span', type=float, default=12.0, help='half-span around each zero for local maps')
    parser.add_argument('--sigma-points', type=int, default=161)
    parser.add_argument('--tau-points', type=int, default=161)
    parser.add_argument('--k-star', type=float, default=1.0)
    parser.add_argument('--parallel', type=int, default=min(4, cpu_count()), help='parallel workers for per-zero maps')
    parser.add_argument('--dps', type=int, default=40, help='mpmath precision (decimal digits)')
    parser.add_argument('--eps', type=float, default=0.0, help='epsilon smoothing for C = 1/(|ζ|^2+eps)')
    parser.add_argument('--sigma-offset', type=float, default=0.0, help='positive-control: evaluate field at σ-δ')
    # Operator test (Hilbert–Pólya style)
    parser.add_argument('--operator-test', action='store_true', help='compute Xi-based potential and eigenvalues')
    parser.add_argument('--op-n', type=int, default=801, help='grid points for operator test')
    parser.add_argument('--op-sigma', type=float, default=2.0, help='Gaussian kernel width for smoothing f(t)')
    parser.add_argument('--op-range', type=float, nargs=2, default=None, help='t-range [tmin, tmax] for operator test')
    parser.add_argument('--op-center-k', type=int, default=40, help='if no range, use zeros[0..k] to set range')
    parser.add_argument('--op-eigs', type=int, default=60, help='number of smallest eigenvalues to report')
    args = parser.parse_args()

    mp.dps = int(args.dps)
    os.makedirs('artifacts/riemann_567901', exist_ok=True)
    zeros = get_zeros(args.zeros)
    print(f"Loaded {len(zeros)} zeros. First: {zeros[0]:.6f}, Last: {zeros[-1]:.6f}")
    np.save('zeros_cached.npy', zeros)

    if args.skip_plots:
        print('skip-plots set: cached zeros only; no figures generated.')
        return

    # Local neighborhoods around the first K zeros
    K = min(args.plots, len(zeros))
    tasks = [(idx, float(zeros[idx]), args) for idx in range(K)]
    if args.parallel and args.parallel > 1:
        with Pool(processes=int(args.parallel)) as pool:
            results = list(pool.map(_per_zero_task, tasks))
    else:
        results = [ _per_zero_task(t) for t in tasks ]

    # Coarse global map (first and last plotted zero bounds)
    if K > 0:
        tau_low = max(0.0, zeros[0] - args.tau_span)
        tau_high = zeros[K-1] + args.tau_span
        tau_center = 0.5 * (tau_low + tau_high)
        tau_halfspan = 0.5 * (tau_high - tau_low)
        sigmas, taus, Zg, Sg, Cg = compute_maps(
            args.sigma_min, args.sigma_max, tau_center, tau_halfspan,
            args.sigma_points, args.tau_points, k_star=args.k_star)
        plot_heatmap(sigmas, taus, Sg, f'Global S(σ, τ) across first {K} zeros', 'artifacts/riemann_567901/S_global.png')
        plot_heatmap(sigmas, taus, Cg, f'Global C(σ, τ) across first {K} zeros', 'artifacts/riemann_567901/C_global.png', cmap='viridis')

    # Maximization analysis across all local maps
    if K > 0 and results:
        sigma_star = []
        sigma_hat = []
        sigma_stat = []
        for r in results:
            sigma_star.append(r['sigma_argmax_per_tau'])
            sigma_hat.append(r['sigma_hat_per_tau'])
            sigma_stat.append(r['sigma_stationary_per_tau'])
        sigma_star = np.concatenate(sigma_star)
        sigma_hat = np.concatenate(sigma_hat)
        sigma_stat = np.concatenate(sigma_stat)
        # Histogram of σ* - 0.5 (how tightly it centers on 1/2)
        delta = sigma_star - 0.5
        plt.figure(figsize=(7, 4))
        plt.hist(delta, bins=50, color='#1f77b4', alpha=0.9)
        plt.xlabel('σ* - 1/2 (argmax of C per τ row)')
        plt.ylabel('count')
        plt.title('Distribution of coherence-maximizing σ (near zeros)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('artifacts/riemann_567901/sigma_argmax_hist.png', dpi=160)
        plt.close()

        # Boundary fractions for window artifact test (A)
        frac_left = float(np.mean(np.isclose(sigma_star, args.sigma_min)))
        frac_right = float(np.mean(np.isclose(sigma_star, args.sigma_max)))

        # Sub-grid stats (B)
        sigma_hat_valid = sigma_hat[np.isfinite(sigma_hat)]
        sigma_stat_valid = sigma_stat[np.isfinite(sigma_stat)]
        stats = {
            'sigma_star_count': int(sigma_star.size),
            'sigma_star_mean': float(sigma_star.mean()),
            'sigma_star_std': float(sigma_star.std()),
            'sigma_star_min': float(sigma_star.min()),
            'sigma_star_max': float(sigma_star.max()),
            'frac_left': frac_left,
            'frac_right': frac_right,
            'sigma_hat_count': int(sigma_hat_valid.size),
            'sigma_hat_mean': float(np.mean(sigma_hat_valid)) if sigma_hat_valid.size else None,
            'sigma_hat_std': float(np.std(sigma_hat_valid)) if sigma_hat_valid.size else None,
            'sigma_hat_min': float(np.min(sigma_hat_valid)) if sigma_hat_valid.size else None,
            'sigma_hat_max': float(np.max(sigma_hat_valid)) if sigma_hat_valid.size else None,
            'sigma_stationary_count': int(sigma_stat_valid.size),
            'sigma_stationary_mean': float(np.mean(sigma_stat_valid)) if sigma_stat_valid.size else None,
            'sigma_stationary_std': float(np.std(sigma_stat_valid)) if sigma_stat_valid.size else None,
            'sigma_stationary_min': float(np.min(sigma_stat_valid)) if sigma_stat_valid.size else None,
            'sigma_stationary_max': float(np.max(sigma_stat_valid)) if sigma_stat_valid.size else None,
            'sigma_min': args.sigma_min,
            'sigma_max': args.sigma_max,
            'eps': args.eps,
            'sigma_offset': args.sigma_offset,
            'dps': args.dps,
        }
        # KS diagnostics versus uniform on [sigma_min, sigma_max]
        def ks_two_sample(x: np.ndarray, a: float, b: float, seed: int = 0):
            x = np.sort(x)
            if x.size == 0:
                return None
            rng = np.random.default_rng(seed)
            u = np.sort(rng.uniform(a, b, x.size))
            i = j = 0
            n = x.size
            m = u.size
            d = 0.0
            cdf_x = cdf_u = 0.0
            while i < n and j < m:
                if x[i] <= u[j]:
                    i += 1; cdf_x = i / n
                else:
                    j += 1; cdf_u = j / m
                d = max(d, abs(cdf_x - cdf_u))
            n_eff = n * m / (n + m)
            p_approx = float(np.exp(-2.0 * n_eff * d * d))
            return {'n': int(n), 'D': float(d), 'p_approx': p_approx}

        ks_hat = ks_two_sample(sigma_hat_valid, args.sigma_min, args.sigma_max)
        ks_stat = ks_two_sample(sigma_stat_valid, args.sigma_min, args.sigma_max)
        if ks_hat is not None:
            stats['ks_sigma_hat'] = ks_hat
        if ks_stat is not None:
            stats['ks_sigma_stationary'] = ks_stat
        with open('artifacts/riemann_567901/coherence_maxima.json', 'w') as f:
            json.dump(stats, f, indent=2)

        np.savez_compressed('artifacts/riemann_567901/coherence_maxima.npz', sigma_star=sigma_star, sigma_hat=sigma_hat, sigma_stationary=sigma_stat)

    # Save a compact NPZ for downstream analyses
    np.savez_compressed(
        'artifacts/riemann_567901/frc567901_outputs.npz',
        zeros=zeros,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        tau_span=args.tau_span,
        sigma_points=args.sigma_points,
        tau_points=args.tau_points,
        k_star=args.k_star,
        plots=K,
        parallel=args.parallel,
        dps=args.dps,
    )
    print('Saved figures to artifacts/riemann_567901 and artifacts/riemann_567901/frc567901_outputs.npz')

    # Optional: Operator test (Hilbert–Pólya prototype)
    if args.operator_test:
        run_operator_test(zeros, args)


 # ---------------------- Operator test utilities ----------------------
def gaussian_kernel(x: np.ndarray, sigma: float) -> np.ndarray:
    k = np.exp(-0.5 * (x / max(1e-9, sigma)) ** 2)
    return k / np.sum(k)


def build_potential_from_xi(t: np.ndarray, sigma_smooth: float) -> np.ndarray:
    """Compute V(t) from f(t) = -ln |Xi(1/2 + i t)| convolved with Gaussian.
    V = (1/4)(f')^2 - (1/2) f'' using central differences.
    """
    # Sample |Xi| on grid
    xi_vals = np.array([xi_abs(0.5, float(tt)) for tt in t]) + 1e-30
    f = -np.log(xi_vals)
    # Smooth with Gaussian kernel (reflect padding)
    # choose kernel size ~ 6 sigma_smooth
    half_w = max(3, int(3 * sigma_smooth / max(1e-12, (t[1]-t[0]))))
    xk = np.linspace(-half_w*(t[1]-t[0]), half_w*(t[1]-t[0]), 2*half_w+1)
    gk = gaussian_kernel(xk, sigma_smooth)
    f_s = np.convolve(np.r_[f[half_w:0:-1], f, f[-2:-half_w-2:-1]], gk, mode='same')
    f_s = f_s[half_w:-half_w]
    h = t[1] - t[0]
    # First and second derivatives (central differences)
    fp = np.empty_like(f_s)
    fpp = np.empty_like(f_s)
    fp[1:-1] = (f_s[2:] - f_s[:-2]) / (2*h)
    fpp[1:-1] = (f_s[2:] - 2*f_s[1:-1] + f_s[:-2]) / (h*h)
    # Neumann boundaries (copy interior neighbor)
    fp[0] = fp[1]; fp[-1] = fp[-2]
    fpp[0] = fpp[1]; fpp[-1] = fpp[-2]
    V = 0.25 * fp**2 - 0.5 * fpp
    return V


def schrodinger_matrix(t: np.ndarray, V: np.ndarray) -> np.ndarray:
    n = len(t)
    h = t[1] - t[0]
    main = 2.0 / (h*h) + V
    off = -1.0 / (h*h)
    H = np.zeros((n, n), dtype=np.float64)
    np.fill_diagonal(H, main)
    i = np.arange(n-1)
    H[i, i+1] = off
    H[i+1, i] = off
    return H


def run_operator_test(zeros: np.ndarray, args):
    os.makedirs('artifacts/riemann_567901', exist_ok=True)
    # Determine t-range
    if args.op_range is not None:
        tmin, tmax = args.op_range
    else:
        # Use zeros up to k to set a range padded by 10 on each side
        k = min(args.op_center_k, len(zeros)-1)
        tmin = max(0.0, float(zeros[0] - 10.0))
        tmax = float(zeros[k] + 10.0)
    t = np.linspace(tmin, tmax, int(args.op_n))
    V = build_potential_from_xi(t, sigma_smooth=float(args.op_sigma))
    H = schrodinger_matrix(t, V)
    # Compute eigenvalues (smallest M)
    M = min(int(args.op_eigs), len(t)-2)
    # Use dense solver then take smallest M
    w, _ = np.linalg.eigh(H)
    w = np.sort(w)[:M]
    # Compare sqrt(lambda) to nearby known zeros in [tmin, tmax]
    zeros_band = zeros[(zeros >= tmin) & (zeros <= tmax)]
    sqrt_w = np.sqrt(np.clip(w, a_min=0.0, a_max=None))
    # For each sqrt_w, find nearest zero distance
    if zeros_band.size > 0:
        dists = []
        for val in sqrt_w:
            dists.append(float(np.min(np.abs(zeros_band - val))))
        dists = np.array(dists)
        report = {
            'tmin': tmin,
            'tmax': tmax,
            'grid_n': int(args.op_n),
            'sigma_smooth': float(args.op_sigma),
            'eigs_reported': int(M),
            'zeros_in_band': int(zeros_band.size),
            'mean_abs_distance': float(np.mean(dists)),
            'median_abs_distance': float(np.median(dists)),
            'max_abs_distance': float(np.max(dists)),
        }
    else:
        report = {
            'tmin': tmin,
            'tmax': tmax,
            'grid_n': int(args.op_n),
            'sigma_smooth': float(args.op_sigma),
            'eigs_reported': int(M),
            'zeros_in_band': 0,
        }
        dists = np.array([])
    # Save artifacts
    np.save('artifacts/riemann_567901/operator_eigs.npy', w)
    np.save('artifacts/riemann_567901/operator_sqrt_eigs.npy', sqrt_w)
    np.save('artifacts/riemann_567901/operator_t_grid.npy', t)
    np.save('artifacts/riemann_567901/operator_V.npy', V)
    with open('artifacts/riemann_567901/operator_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print('Operator test report:', json.dumps(report))


if __name__ == '__main__':
    main()
