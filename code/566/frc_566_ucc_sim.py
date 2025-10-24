import numpy as np
import matplotlib.pyplot as plt

def ucc_step(lnC, D=1.0, S=0.0, dx=1.0, dt=0.1):
    # 1D Neumann BCs
    d2 = np.zeros_like(lnC)
    d2[1:-1] = (lnC[2:] - 2*lnC[1:-1] + lnC[:-2]) / (dx*dx)
    d2[0] = d2[1]; d2[-1] = d2[-2]
    lnC_new = lnC + dt*(D*d2 + S)
    return lnC_new

def energy_like(lnC, dx=1.0):
    g = np.gradient(lnC, dx)
    return np.trapz(g*g, dx=dx)

def run(out='artifacts/566/ucc_dissipation.png'):
    import os
    os.makedirs('artifacts/566', exist_ok=True)
    x = np.linspace(0, 1, 201)
    lnC = np.exp(-50*(x-0.5)**2)  # hump
    D = 0.05
    E = []
    for _ in range(400):
        lnC = ucc_step(lnC, D=D, S=0.0, dx=x[1]-x[0], dt=0.0005)
        E.append(energy_like(lnC, dx=x[1]-x[0]))
    plt.figure(figsize=(5,4))
    plt.plot(E, c='#1f77b4')
    plt.xlabel('step'); plt.ylabel('∫||∇ ln C||²')
    plt.title('UCC dissipation (Neumann)')
    plt.tight_layout(); plt.savefig(out, dpi=160); plt.close()

if __name__ == '__main__':
    run()

