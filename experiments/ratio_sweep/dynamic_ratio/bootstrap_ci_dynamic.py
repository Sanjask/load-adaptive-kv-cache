import json
import numpy as np

IN_JSON = "dynamic_sweep_results.json"
N_BOOT = 10000
SEED = 0
TARGETS = [0.70, 0.80, 0.85, 0.90, 0.95, 0.98]
rng = np.random.default_rng(SEED)

with open(IN_JSON) as f:
    records = json.load(f)["records"]

def per_example_mean(tau, field):
    by_ex = {}
    for r in records:
        if r["coverage_target"] != tau:
            continue
        by_ex.setdefault(r["idx"], []).append(r[field])
    return np.array([np.mean(v) for v in by_ex.values()])

def bootstrap_ci(values, n_boot=N_BOOT, alpha=0.05):
    n = len(values)
    boots = np.array([values[rng.integers(0, n, size=n)].mean() for _ in range(n_boot)])
    return float(values.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

print(f"{'target':>8} {'mean_dF1':>10} {'95% CI':>26} {'signif?':>9}")
for tau in TARGETS:
    dF1 = per_example_mean(tau, "f1_minus_full")
    mean, lo, hi = bootstrap_ci(dF1)
    signif = not (lo <= 0 <= hi)
    print(f"{tau:>8} {mean:>10.4f}   [{lo:>7.4f}, {hi:>7.4f}]   {'YES' if signif else 'no':>9}")
