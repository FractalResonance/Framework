import os, numpy as np, matplotlib.pyplot as plt
OUT='artifacts/100.006.002'; os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)
amp2 = np.array([0.2,0.5,0.3])**2; amp2 = amp2/amp2.sum()

def finite_time_bias(T, M=8000):
    state = rng.uniform(0,1,size=M)
    centers = np.array([1/6,3/6,5/6])
    for _ in range(T):
        attract = centers[np.argmin(np.abs(state[:,None]-centers[None,:]),axis=1)]
        state += 0.02*(attract - state) + 0.01*rng.normal(size=state.shape)
        state = np.mod(state,1.0)
    bins = np.digitize(state, [1/3,2/3])
    p = np.bincount(bins, minlength=3)/M
    return np.abs(p-amp2).mean()
Ts = np.arange(1,120,6)
bias_T = np.array([finite_time_bias(int(T)) for T in Ts])
plt.figure(figsize=(5.5,4)); plt.plot(Ts, bias_T, 'o-')
plt.xlabel('locking horizon T'); plt.ylabel('mean |p - |alpha|^2|')
plt.title('Finite-time locking bias'); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(f'{OUT}/bias_vs_time.png', dpi=160); plt.close()

def skew_bias(delta, M=9000, steps=200):
    base = np.array([1,1,1], dtype=float)
    w = base + delta*np.array([1,-1,0])
    w = np.clip(w, 0.01, None); w = w/w.sum()
    bins = rng.choice(3, size=M, p=w)
    centers = np.array([1/6,3/6,5/6]); state = centers[bins] + 0.03*rng.normal(size=M)
    state = np.mod(state,1.0)
    for _ in range(steps):
        attract = centers[np.argmin(np.abs(state[:,None]-centers[None,:]),axis=1)]
        state += 0.02*(attract - state) + 0.01*rng.normal(size=state.shape)
        state = np.mod(state,1.0)
    bins = np.digitize(state, [1/3,2/3])
    p = np.bincount(bins, minlength=3)/M
    return np.abs(p-amp2).mean()
D = np.linspace(0,0.8,10)
bias_D = np.array([skew_bias(d) for d in D])
plt.figure(figsize=(5.5,4)); plt.plot(D, bias_D, 's-')
plt.xlabel('initial skew delta'); plt.ylabel('mean |p - |alpha|^2|')
plt.title('Non-ergodic ensemble bias'); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(f'{OUT}/bias_vs_skew.png', dpi=160); plt.close()

def ramp_bias(rate, M=8000, steps=200):
    state = rng.uniform(0,1,size=M)
    centers = np.array([1/6,3/6,5/6])
    for t in range(steps):
        shift = rate * t/steps
        c = centers + np.array([shift,0,-shift])/6.0
        attract = c[np.argmin(np.abs(state[:,None]-c[None,:]),axis=1)]
        state += 0.02*(attract - state) + 0.01*rng.normal(size=state.shape)
        state = np.mod(state,1.0)
    bins = np.digitize(state, [1/3,2/3])
    p = np.bincount(bins, minlength=3)/M
    return np.abs(p-amp2).mean()
R = np.linspace(0,1.2,10)
bias_R = np.array([ramp_bias(r) for r in R])
plt.figure(figsize=(5.5,4)); plt.plot(R, bias_R, 'd-')
plt.xlabel('ramp rate (arb.)'); plt.ylabel('mean |p - |alpha|^2|')
plt.title('Non-stationary coupling bias'); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(f'{OUT}/bias_vs_ramp.png', dpi=160); plt.close()

print({'finite_time': float(bias_T[-1]), 'skew_0.8': float(bias_D[-1]), 'ramp_1.2': float(bias_R[-1])})
