import os, numpy as np, matplotlib.pyplot as plt
OUT='artifacts/100.006'; os.makedirs(OUT, exist_ok=True)

# toy amplitudes
amps = np.array([0.2, 0.5, 0.3])
amp2 = amps**2/np.sum(amps**2)

# (i) direct equilibrium sampler with |alpha|^2 bias
rng = np.random.default_rng(0)
N = 20000
choices = rng.choice(len(amps), size=N, p=amp2)
props = np.bincount(choices, minlength=len(amps))/N

plt.figure(figsize=(5.2,4))
plt.plot(amp2, props, 'o', label='empirical')
plt.plot([0,1],[0,1],'k--', lw=1)
plt.xlabel('|alpha|^2'); plt.ylabel('empirical proportion')
plt.title('Proportions vs |alpha|^2 (toy)')
plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(f'{OUT}/proportions_vs_amp2.png', dpi=160); plt.close()

# (ii) drift dynamics: microstates move toward sector attractors with small noise
M = 6000; iters = 200
state = rng.uniform(0,1,size=(M,))  # sector coordinate in [0,1); three bins
history = []
for t in range(iters):
    # small drift toward nearest sector center {1/6,3/6,5/6} with weights amp2
    centers = np.array([1/6,3/6,5/6])
    attract = centers[np.argmin(np.abs(state[:,None]-centers[None,:]),axis=1)]
    state += 0.02*(attract - state) + 0.01*rng.normal(size=state.shape)
    state = np.mod(state,1.0)
    bins = np.digitize(state, [1/3,2/3])
    p = np.bincount(bins, minlength=3)/M
    history.append(p)
H = np.array(history)

plt.figure(figsize=(6,4))
for j in range(3):
    plt.plot(H[:,j], label=f'p{j} (emp)')
    plt.hlines(amp2[j], 0, iters-1, colors='k', linestyles='--', lw=0.8)
plt.xlabel('iteration'); plt.ylabel('proportion')
plt.title('Convergence to |alpha|^2 (toy drift)')
plt.grid(True, alpha=0.3); plt.legend()
plt.tight_layout(); plt.savefig(f'{OUT}/equilibrium_convergence.png', dpi=160); plt.close()
