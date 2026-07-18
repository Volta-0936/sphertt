"""Figure: amplification-law validation (predicted vs measured)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

rows = [json.loads(l) for l in
        open(r"D:\sphertt-0.1.0\prototype\010\results010.jsonl",
             encoding="utf-8")]
pts = [(r["amp_predicted"], r["amp_measured"], r["beta"]) for r in rows
       if r["amp_predicted"] is not None]
pred = np.array([p for p, _, _ in pts])
meas = np.array([m for _, m, _ in pts])
beta = np.array([b for _, _, b in pts])
ratio = pred / meas
within2 = np.mean((ratio > 0.5) & (ratio < 2.0))
print(f"n={len(pts)}, within factor 2: {within2:.1%}, "
      f"median ratio {np.median(ratio):.2f}")

fig, ax = plt.subplots(figsize=(5.4, 4.6))
lo, hi = 1.5, 45
xs = np.array([lo, hi])
ax.fill_between(xs, xs / 2, xs * 2, color="0.9", label="factor-2 band")
ax.plot(xs, xs, "k--", lw=0.8)
mid = (beta >= 0.4) & (beta <= 0.6)
sc = ax.scatter(meas[~mid], pred[~mid], c=beta[~mid], cmap="viridis",
                s=28, vmin=0.2, vmax=1.0, label=r"$\beta \notin [0.4,0.6]$")
ax.scatter(meas[mid], pred[mid], c=beta[mid], cmap="viridis", s=42,
           marker="^", vmin=0.2, vmax=1.0,
           label=r"$\beta \in [0.4,0.6]$ (runaway zone)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("measured amplification $A$")
ax.set_ylabel(
    r"predicted $\min\{(1-\bar g^2)^{-1/2},\ \sqrt{2}/\bar\delta\}$")
ax.set_title(f"amplification law: {within2:.0%} of {len(pts)} points "
             f"within factor 2\n(median ratio {np.median(ratio):.2f}; "
             r"$N=1024$, $[2]^{10}$, $\beta\times\chi_x\times$2 seeds)",
             fontsize=10)
plt.colorbar(sc, ax=ax, label=r"$\beta$")
ax.legend(fontsize=7, loc="upper left")
fig.tight_layout()
fig.savefig(r"D:\sphertt-0.1.0\prototype\010\fig5_amp_validation.png",
            dpi=150)
print("fig written")
