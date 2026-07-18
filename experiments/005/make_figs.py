"""Figures for prototype/005, generated from the JSONL results only."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = r"D:\sphertt-0.1.0\prototype\005"


def load(name):
    with open(rf"{BASE}\{name}", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


r5 = load("results005.jsonl")
r5b = load("results005b.jsonl")
r5c = load("results005c.jsonl")

# ---------------------------------------------------------------- fig7
h2_tt = [r for r in r5 if r["exp"] == "H2" and r["engine"] == "tt-state"]
h2_d = [r["narma10"] for r in r5 if r["exp"] == "H2"
        and r["engine"] == "dense-state"]
chis = sorted({r["chi_x"] for r in h2_tt})
mean = [np.mean([r["narma10"] for r in h2_tt if r["chi_x"] == c])
        for c in chis]
lo = [min(r["narma10"] for r in h2_tt if r["chi_x"] == c) for c in chis]
hi = [max(r["narma10"] for r in h2_tt if r["chi_x"] == c) for c in chis]
dm = [np.mean([r["delta_mean"] for r in h2_tt if r["chi_x"] == c])
      for c in chis]

fig, ax = plt.subplots(figsize=(6, 4))
ax.fill_between(chis, lo, hi, alpha=0.25, color="C0")
ax.plot(chis, mean, "o-", color="C0", label="TT-state (native)")
ax.axhspan(min(h2_d), max(h2_d), color="C1", alpha=0.25,
           label="dense-state (seeds)")
ax.set_xlabel(r"state bond dimension $\chi_x$")
ax.set_ylabel("NARMA10 test NRMSE")
ax2 = ax.twinx()
ax2.plot(chis, dm, "s--", color="C3", alpha=0.7)
ax2.set_ylabel(r"per-step rounding error $\bar\delta$", color="C3")
ax.set_title(r"N=1024 ($[4]^5$, max rank 16): exact at $\chi_x=16$")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(rf"{BASE}\fig7_ttstate_chi.png", dpi=150)

# ---------------------------------------------------------------- fig8
h3 = [r for r in r5 if r["exp"] == "H3"]
h3.sort(key=lambda r: r["N"])
N = [r["N"] for r in h3]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
axes[0].loglog(N, [r["ms_per_step"] for r in h3], "o-")
axes[0].set_xlabel("N"); axes[0].set_ylabel("ms / step")
axes[0].set_title("cost: log-linear in N")
axes[1].semilogx(N, [r["delta_mean"] for r in h3], "o-", color="C3")
axes[1].set_xlabel("N"); axes[1].set_ylabel(r"$\bar\delta$")
axes[1].set_title(r"rounding error at fixed $\chi_x=16$, $\beta=0.8$")
axes[2].semilogx(N, [r["mc300"] for r in h3], "o-", color="C2")
axes[2].set_xlabel("N"); axes[2].set_ylabel("MC (300-delay window)")
axes[2].set_title("memory: dies with deep truncation")
fig.tight_layout()
fig.savefig(rf"{BASE}\fig8_ttstate_scaling.png", dpi=150)

# ---------------------------------------------------------------- fig9
dense = {r["beta"]: r for r in r5c if r["engine"] == "dense-state"}
tt8 = {r["beta"]: r for r in r5c
       if r["engine"] == "tt-state" and r["chi_w"] == 8}
ref = next(r for r in r5 if r["exp"] == "H3" and r["n_dims"] == 8)
tt8.setdefault(0.8, {"mc300": ref["mc300"], "narma10": ref["narma10"]})
tt2 = {r["beta"]: r for r in r5c
       if r["engine"] == "tt-state" and r["chi_w"] == 2}
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
for d, lab, c in [(dense, "dense-state (ceiling)", "C1"),
                  (tt8, r"TT-state $\chi_x=16,\chi_w=8$", "C0"),
                  (tt2, r"TT-state $\chi_x=16,\chi_w=2$", "C4")]:
    if d:
        b = sorted(d)
        axes[0].plot(b, [d[x]["mc300"] for x in b], "o-", color=c, label=lab)
        axes[1].plot(b, [d[x]["narma10"] for x in b], "o-", color=c)
axes[0].set_xlabel(r"$\beta$"); axes[0].set_ylabel("MC (window 300)")
axes[0].legend(fontsize=8)
axes[1].set_xlabel(r"$\beta$"); axes[1].set_ylabel("NARMA10 NRMSE")
fig.suptitle(r"H5: raising $\beta$ to rescue deep truncation (N=65536)")
fig.tight_layout()
fig.savefig(rf"{BASE}\fig9_beta_rescue.png", dpi=150)
print("figs written")
