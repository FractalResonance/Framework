import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    base = 'artifacts/riemann_567901'
    z = np.load(os.path.join(base, 'coherence_maxima_off.npz'))
    st = z['sigma_stationary']
    st = st[np.isfinite(st)]
    plt.figure(figsize=(7,4))
    plt.hist(st, bins=20, color='#7f7f7f', alpha=0.9)
    plt.xlabel('σ_stationary (off-critical)')
    plt.ylabel('count')
    plt.title('Histogram of σ_stationary (σ∈[0.6,0.7])')
    plt.grid(True, alpha=0.3)
    out = os.path.join(base, 'sigma_stationary_hist_off.png')
    plt.tight_layout(); plt.savefig(out, dpi=160); plt.close()
    print('wrote', out)

if __name__ == '__main__':
    main()

