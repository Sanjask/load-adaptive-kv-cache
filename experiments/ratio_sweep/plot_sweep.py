import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("ratio_sweep_results.json") as f:
    summary = json.load(f)

ratios = sorted(summary["ratios"].keys(), key=float)
ttfts = [summary["ratios"][r]["mean_ttft"] for r in ratios]
f1s = [summary["ratios"][r]["mean_f1"] for r in ratios]

full_ttft = summary["full_prefill"]["mean_ttft"]
full_f1 = summary["full_prefill"]["mean_f1"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# --- Plot 1: TTFT and F1 vs ratio ---
ax = axes[0]
ax.plot([float(r) for r in ratios], ttfts, "o-", color="#2b6cb0", label="TTFT (blend)")
ax.axhline(full_ttft, ls="--", color="#2b6cb0", alpha=0.5, label="TTFT (full prefill)")
ax.set_xlabel("Recompute ratio")
ax.set_ylabel("Mean TTFT (s)", color="#2b6cb0")
ax.tick_params(axis="y", labelcolor="#2b6cb0")
ax2 = ax.twinx()
ax2.plot([float(r) for r in ratios], f1s, "s-", color="#c05621", label="F1 (blend)")
ax2.axhline(full_f1, ls="--", color="#c05621", alpha=0.5, label="F1 (full prefill)")
ax2.set_ylabel("Mean F1", color="#c05621")
ax2.tick_params(axis="y", labelcolor="#c05621")
ax.set_title("TTFT & F1 vs Recompute Ratio")

# --- Plot 2: Pareto frontier (TTFT vs F1) ---
ax = axes[1]
ax.plot(ttfts, f1s, "o-", color="#2f855a")
for r, t, f in zip(ratios, ttfts, f1s):
    ax.annotate(f"{float(r):.2f}", (t, f), textcoords="offset points", xytext=(6, 4), fontsize=9)
ax.scatter([full_ttft], [full_f1], color="red", zorder=5, label="full prefill")
ax.annotate("full", (full_ttft, full_f1), textcoords="offset points", xytext=(6, 4), fontsize=9, color="red")
ax.set_xlabel("Mean TTFT (s)  — lower is better")
ax.set_ylabel("Mean F1  — higher is better")
ax.set_title("Speed/Quality Frontier")
ax.legend()

plt.tight_layout()
plt.savefig("ratio_sweep_frontier.png", dpi=150)
print("Saved ratio_sweep_frontier.png")
