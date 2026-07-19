"""Figures 14-15 for Paper II v2, from experiment 012/015 results."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = r"D:\sphertt-0.1.0\prototype\015"

# ---------------- fig14: structured-perturbation gain and its reversal
fig, ax = plt.subplots(figsize=(6.4, 4.2))
groups = [
    ("d=8\n" r"$\beta$=0.95", [65.5, 40.5, 67.8], [128.1, 72.1, 106.3]),
    ("d=8\n" r"$\beta$=0.99", [113.6, 70.4, 103.1], [186.0, 119.3, 214.0]),
    ("d=15\n" r"$\beta$=0.99 (def.)", [103.7], [63.9]),
]
xs = np.arange(len(groups))
w = 0.34
for i, (label, rnd, krn) in enumerate(groups):
    ax.bar(i - w / 2, np.mean(rnd), w, color="C3", alpha=0.75,
           label="random-tt" if i == 0 else None)
    ax.bar(i + w / 2, np.mean(krn), w, color="C2", alpha=0.75,
           label="kron-sum" if i == 0 else None)
    ax.scatter([i - w / 2] * len(rnd), rnd, color="k", s=14, zorder=3)
    ax.scatter([i + w / 2] * len(krn), krn, color="k", s=14, zorder=3)
ax.set_xticks(xs)
ax.set_xticklabels([g[0] for g in groups], fontsize=9)
ax.set_ylabel("MC (window 300)")
ax.set_title("structured perturbation: large gain at moderate depth,\n"
             "reversal at d=15 (dots: individual seeds)", fontsize=10)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(rf"{BASE}\fig14_structure_reversal.png", dpi=150)

# ---------------- fig15: the three-resource law (noise-matched depth gap)
fig, ax = plt.subplots(figsize=(6.4, 4.4))
d8_curve = [(0.031, 277.4, "kron $\\beta$=.99"),
            (0.047, 252.6, ".98"), (0.064, 245.5, ".95"),
            (0.087, 230.9, ".92"), (0.105, 216.1, ".90")]
ax.plot([p[0] for p in d8_curve], [p[1] for p in d8_curve], "o-",
        color="C0", label=r"$d=8$ noise curve ($\chi_x$=48)")
ax.scatter([0.097], [63.9], color="C2", s=70, marker="s",
           label=r"$d=15$ kron", zorder=3)
ax.scatter([0.116], [103.7], color="C3", s=70, marker="^",
           label=r"$d=15$ random", zorder=3)
ax.axhline(297, color="k", ls=":", lw=0.9)
ax.text(0.033, 288, "dense ceiling (297)", fontsize=8)
ax.annotate("genuine depth cost\n($\\times\\,2.1$ at matched noise)",
            xy=(0.11, 110), xytext=(0.062, 150), fontsize=9,
            arrowprops=dict(arrowstyle="->", lw=0.9))
ax.set_xlabel(r"per-step rounding error $\bar\delta$")
ax.set_ylabel("MC (window 300, definitive protocol)")
ax.set_title("the third resource: depth\n"
             r"($T$=16000, validated $\alpha$, seed 0)", fontsize=10)
ax.set_ylim(0, 310)
ax.legend(fontsize=8, loc="center right")
fig.tight_layout()
fig.savefig(rf"{BASE}\fig15_three_resource.png", dpi=150)
print("figs written")
