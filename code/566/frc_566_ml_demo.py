import numpy as np

def demo_stats():
    # Placeholder: generate synthetic ΔS and Δ ln C with slope ≈ -1
    rng = np.random.default_rng(0)
    dlnC = rng.normal(0.05, 0.01, 50)
    dS = -1.0*dlnC + rng.normal(0.0, 0.01, 50)
    slope = np.polyfit(dlnC, dS, 1)[0]
    return float(slope), float(np.corrcoef(dlnC, dS)[0,1])

if __name__ == '__main__':
    m, r = demo_stats()
    print({'slope': m, 'corr': r})

