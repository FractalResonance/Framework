import os
import numpy as np
import matplotlib.pyplot as plt

OUT = "artifacts/100.004"
os.makedirs(OUT, exist_ok=True)

def weak_measurement_drift(alpha=0.02, trials=200, T=200, seed=0):
    rng = np.random.default_rng(seed)
    # toy: coherence C_t follows dC = alpha*(1-C)*dt + noise; lock when C>0.98
    locks = []
    mean_C = np.zeros(T)
    for _ in range(trials):
        C = 0.5
        for t in range(T):
            noise = 0.02 * rng.normal()
            C = np.clip(C + alpha*(1.0 - C) + noise, 0.0, 1.0)
            mean_C[t] += C
            if C > 0.98:
                locks.append(t)
                # continue accumulating to keep array lengths aligned
                for tt in range(t+1, T):
                    mean_C[tt] += C
                break
        else:
            locks.append(T)
    mean_C /= trials
    return mean_C, np.array(locks)

def visibility_vs_g(alpha=0.02, gmax=2.0, points=60):
    g = np.linspace(0, gmax, points)
    # standard toy: V_std = exp(-gamma*g^2)
    gamma = 0.6
    V_std = np.exp(-gamma*g*g)
    # FRC toy: slightly altered slope near resonance
    V_frc = np.exp(-(gamma*(1.0-0.08) + 0.04*np.tanh(2*g)) * g*g)
    return g, V_std, V_frc

def plot_drift_and_lock():
    alphas = [0.01, 0.02, 0.04]
    T = 200
    plt.figure(figsize=(7,4))
    for a in alphas:
        mean_C, locks = weak_measurement_drift(alpha=a, T=T, seed=42)
        plt.plot(mean_C, label=f"alpha={a}")
    plt.xlabel("step")
    plt.ylabel("mean C")
    plt.title("Weak pre-collapse drift (toy)")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "weak_drift.png"), dpi=160); plt.close()

def plot_visibility():
    g, V_std, V_frc = visibility_vs_g()
    plt.figure(figsize=(7,4))
    plt.plot(g, V_std, label="standard")
    plt.plot(g, V_frc, label="FRC", ls="--")
    plt.xlabel("coupling g"); plt.ylabel("visibility V")
    plt.title("Interferometer visibility vs g (toy)")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "visibility_vs_g.png"), dpi=160); plt.close()

if __name__ == "__main__":
    plot_drift_and_lock()
    plot_visibility()

