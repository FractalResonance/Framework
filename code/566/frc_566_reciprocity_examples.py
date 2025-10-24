import numpy as np
import matplotlib.pyplot as plt

def chemical_toy(kb=1.380649e-23, T=300.0):
    # synthetic equilibrium pairs: p -> q
    rng = np.random.default_rng(0)
    Hp = rng.normal(2.0, 0.2, 20)  # nats proxy
    Hq = Hp + rng.normal(-0.1, 0.05, 20)
    dlnC = -(Hq - Hp)  # k_* = 1 here (information proxy)
    dG = -kb*T * dlnC
    return dlnC, dG

def plot_dG_vs_dlnC(dlnC, dG, out='artifacts/566/dG_vs_dlnC.png'):
    import os
    os.makedirs('artifacts/566', exist_ok=True)
    plt.figure(figsize=(5,4))
    plt.scatter(dlnC, dG, c='#1f77b4')
    m = np.polyfit(dlnC, dG, 1)[0]
    x = np.linspace(min(dlnC), max(dlnC), 100)
    plt.plot(x, m*x, c='#ff7f0e', label=f'slope≈{m:.3e}')
    plt.xlabel('Δ ln C')
    plt.ylabel('ΔG (J)')
    plt.legend(); plt.tight_layout(); plt.savefig(out, dpi=160); plt.close()

if __name__ == '__main__':
    dlnC, dG = chemical_toy()
    plot_dG_vs_dlnC(dlnC, dG)

