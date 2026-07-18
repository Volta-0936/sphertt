"""Headline figure for experiment 007."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = r"D:\sphertt-0.1.0\prototype\007"


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


r7 = load(rf"{BASE}\results007.jsonl")
r5 = load(r"D:\sphertt-0.1.0\prototype\005\results005.jsonl")

naive = sorted([(r["N"], r["mc300"]) for r in r5 if r.get("exp") == "H3"
                and r["n_dims"] >= 8])
guided = sorted([(r["N"], r["mc300"]) for r in r7
                 if r["part"] in ("A", "C") and r["beta"] == 0.99]
                + [(r["N"], r["mc300"]) for r in r7
                   if r["part"] == "A" and r["beta"] == 0.98 and False])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
ax.semilogx([n for n, _ in naive], [m for _, m in naive], "o-", color="C3",
            label=r"naive design ($\beta$=0.8, $\chi_x$=16), exp 005")
ax.semilogx([n for n, _ in guided], [m for _, m in guided], "o-", color="C2",
            lw=2, label=r"theory-guided ($\beta$=0.99, $\chi_x$=48)")
ax.axvline(8 * 2 ** 30 / 8, color="k", ls=":", lw=1)
ax.text(8 * 2 ** 30 / 8 * 0.13, 42, "dense state\n> 8 GB", fontsize=8,
        ha="center")
ax.set_xlabel("N (reservoir units)")
ax.set_ylabel("MC (window 300)")
ax.set_title("memory beyond the state wall:\n"
             "same engine, 800x via the delta decomposition")
ax.legend(fontsize=8, loc="center left")
ax.annotate("MC = 74 at N > 10$^9$\nmodel = 861 KB", xy=(1.07e9, 73.7),
            xytext=(1.5e6, 55), fontsize=9,
            arrowprops=dict(arrowstyle="->", lw=0.8))

ax = axes[1]
for r in r7:
    if r["part"] == "C" and r["n_dims"] == 15:
        prof = np.asarray(r["mc_profile"])
        ax.plot(np.arange(1, len(prof) + 1), prof, color="C2", lw=1)
ax.set_xlabel("delay k")
ax.set_ylabel(r"$r^2(k)$")
ax.set_title(r"memory function at N = 1,073,741,824"
             "\n" r"($\beta$=0.99, $\chi_x$=48, model 861 KB)")
fig.tight_layout()
fig.savefig(rf"{BASE}\fig13_beyond_the_wall.png", dpi=150)
print("fig13 written")
