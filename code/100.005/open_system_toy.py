import os
import numpy as np
import matplotlib.pyplot as plt
OUT="artifacts/100.005"; os.makedirs(OUT, exist_ok=True)
# toy entropy accounting during resonant locking
def simulate(T=300,trials=200,seed=0):
    rng=np.random.default_rng(seed)
    S_sys=np.zeros(T); S_env=np.zeros(T)
    for _ in range(trials):
        s=0.9; e=0.1
        for t in range(T):
            s=max(0.0, s-0.004-0.001*rng.normal())
            e=min(1.0, e+0.003+0.001*rng.normal())
            S_sys[t]+=s; S_env[t]+=e
    S_sys/=trials; S_env/=trials; S_tot=S_sys+S_env
    return S_sys,S_env,S_tot
S_sys,S_env,S_tot=simulate()
t=np.arange(len(S_sys))
plt.figure(figsize=(7,4))
plt.plot(t,S_sys,label='S_sys'); plt.plot(t,S_env,label='S_env'); plt.plot(t,S_tot,label='S_total',lw=2)
plt.xlabel('step'); plt.ylabel('entropy (toy units)'); plt.title('Entropy accounting during phase-locking (toy)')
plt.legend(); plt.grid(True,alpha=0.3); plt.tight_layout(); plt.savefig(f"{OUT}/entropy_accounting.png",dpi=160)
