"""Analysis for experiment 006: decomposition fit + figures 10-12."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = r"D:\sphertt-0.1.0\prototype\006"


def load(name):
    try:
        with open(rf"{BASE}\{name}", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []


rA = [r for r in load("results006.jsonl") if r["part"] == "A"]
rB = [r for r in load("results006.jsonl") if r["part"] == "B"]
rB2 = [r for r in load("results006.jsonl") if r["part"] == "B2"]
rD = load("results006b.jsonl")

# ------------------------------------------------- fig10: exact attractor
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
chis = sorted(int(c) for c in rA[0]["delta_attr_mean"])
for r in rA:
    axes[0].plot(chis, [r["delta_attr_mean"][str(c)] for c in chis], "o-",
                 label=f"β={r['beta']}")
axes[0].set_xlabel(r"$\chi$"); axes[0].set_ylabel("one-shot error")
axes[0].set_title("exact attractor: near-incompressible\n(N=65536, all β)")
axes[0].legend(fontsize=8)
for r in rA:
    axes[1].semilogy(np.asarray(r["schmidt_mid"]), label=f"β={r['beta']}")
if rD:
    for r in rD:
        if r["chi_x"] == 16 and r["beta"] in (0.8, 1.0):
            axes[1].semilogy(np.asarray(r["spec_z"]), "--",
                             label=f"trunc. z, β={r['beta']}")
axes[1].set_xlabel("index"); axes[1].set_ylabel("Schmidt value (mid bond)")
axes[1].set_title("exact spectrum flat; truncated system's\nown z-spectrum decays")
axes[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig(rf"{BASE}\fig10_attractor.png", dpi=150)

# ------------------------------------------------- decomposition fit
print("=== decomposition: delta^2 = flow^2 + stock^2 ===")
by_chi = {}
for r in rB:
    if r["chi_in"] == 4:
        by_chi.setdefault(r["chi_x"], {})[r["beta"]] = r["delta_bar"]
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
colors = {8: "C0", 16: "C1", 32: "C2"}
for chi_x, d in sorted(by_chi.items()):
    stock = d.get(1.0, 0.0)
    betas = sorted(b for b in d if b < 1.0)
    x = np.array([1.0 - b for b in betas])
    flow = np.array([np.sqrt(max(d[b] ** 2 - stock ** 2, 0.0))
                     for b in betas])
    mask = flow > 0
    c = float(np.sum(flow[mask] * x[mask]) / np.sum(x[mask] ** 2))
    pred = np.sqrt((c * x) ** 2 + stock ** 2)
    meas = np.array([d[b] for b in betas])
    resid = np.abs(pred - meas) / meas
    print(f"chi_x={chi_x}: stock={stock:.4f}  c={c:.3f}  "
          f"max rel resid={resid.max():.2%}")
    col = colors.get(chi_x, "C3")
    axes[0].plot(x, meas, "o", color=col)
    xs = np.linspace(0, x.max(), 100)
    axes[0].plot(xs, np.sqrt((c * xs) ** 2 + stock ** 2), "-", color=col,
                 label=rf"$\chi_x$={chi_x}: $\sqrt{{({c:.2f}(1-\beta))^2 + {stock:.3f}^2}}$")
    axes[1].plot(meas, pred, "o", color=col)
axes[0].set_xlabel(r"$1-\beta$"); axes[0].set_ylabel(r"$\bar\delta$")
axes[0].legend(fontsize=8); axes[0].set_title("measured vs 2-term model")
lim = [0, max(r["delta_bar"] for r in rB) * 1.1]
axes[1].plot(lim, lim, "k--", lw=0.8)
axes[1].set_xlabel("measured"); axes[1].set_ylabel("model")
axes[1].set_title(r"$\bar\delta^2 = (c(1-\beta))^2 + \delta_{stock}^2$")
fig.tight_layout()
fig.savefig(rf"{BASE}\fig11_decomposition.png", dpi=150)

# chi_in dependence of the stock term
stock_in = {r["chi_in"]: r["delta_bar"] for r in rB if r["beta"] == 1.0
            and r["chi_x"] == 16}
print("stock term vs chi_in (beta=1, chi_x=16):", stock_in)
for r in rB2:
    g = f"{r['g_bar']:.4f}" if r["g_bar"] is not None else "n/a"
    print(f"B2 beta={r['beta']}: g_bar={g} "
          f"err_steady={r['err_steady']:.3f}")

# ---------------------------------------- MC scaling law (005 + 006c/d)
# points: (beta, chi_x, mc300), chi_w=8 only
pts = []
for r in load(r"..\005\results005c.jsonl"):
    if r.get("engine") == "tt-state" and r.get("chi_w") == 8:
        pts.append((r["beta"], r["chi_x"], r["mc300"]))
for r in load(r"..\005\results005.jsonl"):
    if r.get("exp") == "H3" and r.get("n_dims") == 8:
        pts.append((0.8, r["chi_x"], r["mc300"]))
for r in load("results006c.jsonl"):
    if r.get("engine") == "tt-state":
        pts.append((r["beta"], r["chi_x"], r["mc300"]))
for r in load("results006d.jsonl"):
    pts.append((r["beta"], r["chi_x"], r["mc300"]))
if pts:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bcol = {0.8: "C3", 0.9: "C0", 0.95: "C2", 1.0: "C7"}
    for beta in sorted({b for b, _, _ in pts}):
        sel = sorted((c, m) for b, c, m in pts if b == beta)
        chis = np.array([c for c, _ in sel])
        mcs = np.array([m for _, m in sel])
        ax.loglog(chis ** 2, mcs, "o-", color=bcol.get(beta, "C5"),
                  label=rf"$\beta$={beta}")
    xs = np.array([200.0, 1300.0])
    ax.loglog(xs, 0.060 * xs, "k--", lw=0.9)
    ax.text(500, 0.060 * 500 * 1.6, r"slope 2 ($K_0\chi_x^2$)", fontsize=7,
            rotation=24)
    ax.axhline(297, color="k", lw=0.8, ls=":")
    ax.text(70, 320, "dense ceiling (297)", fontsize=7)
    ax.set_xlabel(r"$\chi_x^2$")
    ax.set_ylabel("MC (window 300)")
    ax.set_title("two-resource law: rank budget vs retention quality\n"
                 r"(N=65536, $\chi_w$=8)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(rf"{BASE}\fig12_mc_chi2_law.png", dpi=150)
print("analysis done, figures written")
