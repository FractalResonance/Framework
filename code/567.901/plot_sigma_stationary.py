import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    base = 'artifacts/riemann_567901'
    os.makedirs(base, exist_ok=True)
    z = np.load(os.path.join(base, 'coherence_maxima.npz'))
    st = z.get('sigma_stationary')
    if st is None:
        raise SystemExit('sigma_stationary not found in coherence_maxima.npz')
    st = st[np.isfinite(st)]
    if st.size == 0:
        raise SystemExit('no finite sigma_stationary values')

    plt.figure(figsize=(7,4))
    plt.hist(st, bins=20, color='#1f77b4', alpha=0.9)
    plt.axvline(0.5, color='crimson', lw=1.5, ls='--', label='σ = 1/2')
    plt.xlabel('σ_stationary')
    plt.ylabel('count')
    plt.title('Histogram of σ_stationary (tight band)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    out = os.path.join(base, 'sigma_stationary_hist.png')
    plt.tight_layout(); plt.savefig(out, dpi=160); plt.close()
    print('wrote', out)

if __name__ == '__main__':
    main()

