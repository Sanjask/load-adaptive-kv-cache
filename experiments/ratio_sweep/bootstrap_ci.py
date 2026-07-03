"""
Bootstrap confidence intervals for the static sweep.
Runs on CPU / your Mac -- no GPU needed. Reads static_sweep_results.json.

For each ratio, computes a bootstrap CI on the mean paired F1 difference
(blend F1 - full prefill F1), aggregated per example so resampling is over
examples (the independent unit), not over reps.
"""
import json
import numpy as np

IN_JSON = "static_sweep_results.json"
N_BOOT = 10000
SEED = 0
RATIOS = [0.05, 0.10, 0.16, 0.25, 0.40, 0.60, 1.0]

rng = np.random.default_rng(SEED)

with open(IN_JSON) as f:
    data = json.load(f)
records = data["records"]


def per_example_mean(ratio, field):
    """Average `field` across reps within each example, return array over examples."""
    by_ex = {}
    for r in records:
        if r["ratio"] != ratio:
            continue
        by_ex.setdefault(r["idx"], []).append(r[field])
    return np.array([np.mean(v) for v in by_ex.values()])


def bootstrap_ci(values, n_boot=N_BOOT, alpha=0.05):
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        boots[b] = sample.mean()
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return float(values.mean()), lo, hi


print(f"{'ratio':>7} {'mean_dF1':>10} {'95% CI':>26} {'signif?':>9}")
out = {}
for ratio in RATIOS:
    dF1 = per_example_mean(ratio, "f1_minus_full")
    mean, lo, hi = bootstrap_ci(dF1)
    # "indistinguishable from full prefill" == CI straddles 0
    signif = not (lo <= 0 <= hi)
    out[ratio] = {"mean_dF1": mean, "ci_low": lo, "ci_high": hi, "differs_from_full": signif}
    print(f"{ratio:>7} {mean:>10.4f}   [{lo:>7.4f}, {hi:>7.4f}]   {'YES' if signif else 'no':>9}")

with open("bootstrap_ci.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved bootstrap_ci.json")
print("Interpretation: 'no' in signif column = F1 statistically indistinguishable from full prefill.")