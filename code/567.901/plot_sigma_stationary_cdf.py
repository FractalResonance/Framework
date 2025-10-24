import os
import numpy as np
import matplotlib.pyplot as plt

def ecdf(x: np.ndarray):
    xs = np.sort(x)
    ys = np.arange(1, xs.size + 1) / xs.size
    return xs, ys

def main():
    base = 'artifacts/riemann_567901'
    zpath = os.path.join(base, 'coherence_maxima.npz')
    data = np.load(zpath)
    st = data.get('sigma_stationary')
    if st is None:
        raise SystemExit('sigma_stationary not found in coherence_maxima.npz')
    st = st[np.isfinite(st)]
    if st.size == 0:
        raise SystemExit('no finite sigma_stationary values')

    xs, ys = ecdf(st)
    # Uniform reference over observed window [min,max]
    lo, hi = float(np.min(st)), float(np.max(st))
    xu = np.linspace(lo, hi, 200)
    yu = (xu - lo) / (hi - lo)

    plt.figure(figsize=(7,4))
    plt.plot(xs, ys, label='σ_stationary CDF', color='#2ca02c')
    plt.plot(xu, yu, label='Uniform CDF (same support)', color='#ff7f0e', ls='--')
    plt.axvline(0.5, color='crimson', lw=1.2, ls=':')
    plt.xlabel('σ')
    plt.ylabel('Cumulative probability')
    plt.title('σ_stationary CDF vs Uniform')
    plt.legend()
    out = os.path.join(base, 'sigma_stationary_cdf.png')
    plt.tight_layout(); plt.savefig(out, dpi=160); plt.close()
    print('wrote', out)

if __name__ == '__main__':
    main()

