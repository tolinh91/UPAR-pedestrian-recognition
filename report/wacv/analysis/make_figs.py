"""Figures of the WACV Track 1 paper from fig1_cases.json and numbers.json (matplotlib, print).
../../../../Upar-Challenge-Track-2/.venv/bin/python make_figs.py"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).parent; FIG = HERE.parent / "figs"; FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
                     "font.size": 7, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "xtick.major.size": 2, "ytick.major.size": 2, "pdf.fonttype": 42})
BLUE, ORANGE, INK, MUTED = "#1F6FB4", "#D9661F", "#222222", "#8A8A8A"

# ---------------- Figure 1: the method on three unseen PETA images
cs = json.load(open(HERE / "fig1_cases.json")); tau = cs["tau"]
TITLES = {"agree": "both encoders agree", "ens": "the encoders disagree", "rule": "the rule decides"}
NICE = {"Accessory-Bag": "bag", "Accessory-Backpack": "backpack", "UpperBody-Color-Pink": "pink top"}
fig = plt.figure(figsize=(6.9, 2.75))
outer = fig.add_gridspec(1, 3, wspace=0.08, left=0.085, right=0.995, top=0.9, bottom=0.11)
for col, c in enumerate(cs["cases"]):
    g = outer[col].subgridspec(2, 3, height_ratios=[2.3, 1.0], hspace=0.22, wspace=0.05)
    img = np.asarray(Image.open(HERE / f"fig1_{c['kind']}.png")).astype(float) / 255
    H, W = img.shape[:2]
    for j, (key, name) in enumerate([(None, "input"), ("D", "DINOv3 query"), ("C", "CLIP query")]):
        ax = fig.add_subplot(g[0, j]); ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.4)
        if key:
            a = np.array(c[f"att{key}"]); a = a / a.max()
            up = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)) / 255
            ax.imshow(up, cmap="magma", alpha=0.6, vmin=0, vmax=1)
        ax.set_title(name, fontsize=6.5, pad=2)
    ax = fig.add_subplot(g[1, :])
    zm = (c["zD"] + c["zC"]) / 2
    vals = [c["zD"], c["zC"], zm, zm + tau * c["b"]]
    labs = ["DINOv3", "CLIP", "mean", r"mean + $\tau b_k$"]
    cols = [BLUE, ORANGE, MUTED, INK]
    yy = np.arange(4)[::-1]
    ax.barh(yy, vals, color=cols, height=0.62)
    ax.axvline(0, color=INK, lw=0.6)
    lim = 7.5
    ax.set_xlim(-lim, lim); ax.set_yticks(yy); ax.set_yticklabels(labs if col == 0 else [], fontsize=6)
    for y_, v in zip(yy, vals):
        ax.text(v + (0.25 if v >= 0 else -0.25), y_, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=5.8, color=INK)
    ax.set_xticks([-5, 0, 5]); ax.tick_params(axis="x", labelsize=6); ax.set_xlabel("logit (decision: > 0)", fontsize=6, labelpad=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.text(0.085 + col * 0.3033 + 0.152, 0.955, f"{NICE[c['attr']]}: label yes, {TITLES[c['kind']]}", ha="center", fontsize=7, fontweight="bold")
fig.savefig(FIG / "method_cases.pdf"); fig.savefig(FIG / "method_cases.png", dpi=200)

# ---------------- Figure 2: Avg vs tau on the two held-out folds
N = json.load(open(HERE / "numbers.json"))
fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.75), sharey=False)
for ax, fold, title in zip(axes, ("PETA", "Market"), ("PETA held out", "Market1501 held out")):
    d = N["tau"][fold]
    for name, col, ls in (("H+", BLUE, "-"), ("CLIP-L", ORANGE, "-"), ("H+ + CLIP-L", INK, "-")):
        t = sorted(float(k) for k in d[name]["curve"]); v = [d[name]["curve"][str(k) if str(k) in d[name]["curve"] else k] for k in t]
        ax.plot(t, v, color=col, lw=1.2 if name == "H+ + CLIP-L" else 0.9, ls=ls, label=name.replace("H+", "DINOv3-H+"))
    ax.axhline(d["H+ + CLIP-L"]["oracle"], color=INK, lw=0.7, ls="--")
    ax.text(0.01, d["H+ + CLIP-L"]["oracle"] - 0.0006, "ensemble, oracle thresholds", fontsize=5.5, va="top", color=INK)
    ax.axvline(0.42, color=MUTED, lw=0.6, ls=":")
    ax.set_title(title, fontsize=7, pad=2); ax.set_xlabel(r"$\tau$", fontsize=7, labelpad=0)
    ax.tick_params(labelsize=6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    lo = min(min(d[n]["curve"].values()) for n in d); ax.set_ylim(max(lo, d["H+ + CLIP-L"]["oracle"] - 0.045), d["H+ + CLIP-L"]["oracle"] + 0.004)
axes[0].set_ylabel("Challenge_Avg", fontsize=7)
axes[1].legend(fontsize=5.5, frameon=False, loc="lower left", handlelength=1.4)
fig.tight_layout(pad=0.3, w_pad=0.8)
fig.savefig(FIG / "tau_rule.pdf"); fig.savefig(FIG / "tau_rule.png", dpi=200)
print("ok")
