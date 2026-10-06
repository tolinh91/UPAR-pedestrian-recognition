"""Figures for the beginner-oriented Vietnamese Track 1 report (two kinds of error, a real threshold example).
Run from report/paper: ../../../Upar-Challenge-Track-2/.venv/bin/python make_intro_figs.py"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
FIG = HERE / "figs"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 220, "savefig.bbox": "tight"})
POS, NEG, INK, MUTED = "#c2410c", "#2563a8", "#1a1e26", "#85837c"


def bal_acc(s, y, t):
    return 0.5 * ((s[y == 1] > t).mean() + (s[y == 0] <= t).mean())


def toy():
    """Two failure modes on synthetic logits: good ranking with a shifted threshold, and poor ranking."""
    rng = np.random.default_rng(0)
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.5), sharey=True)
    cases = [("Xếp hạng tốt, ngưỡng đặt sai", rng.normal(-0.5, .8, 400), rng.normal(-3.5, .8, 2000)),
             ("Ngưỡng đặt đúng, xếp hạng kém", rng.normal(.8, 1.4, 400), rng.normal(-.8, 1.4, 2000))]
    bins = np.linspace(-7, 5, 49)
    for ax, (title, p, n) in zip(axs, cases):
        s = np.r_[p, n]; y = np.r_[np.ones(len(p)), np.zeros(len(n))]
        ax.hist(n, bins, color=NEG, alpha=.55, density=True, label="ảnh không có thuộc tính")
        ax.hist(p, bins, color=POS, alpha=.55, density=True, label="ảnh có thuộc tính")
        ax.axvline(0, color=INK, lw=1.2)
        ax.text(0.12, .78, "ngưỡng 0.5\n(logit 0)", fontsize=7.2, color=INK, transform=ax.get_xaxis_transform())
        best = max(np.linspace(-6, 4, 201), key=lambda t: bal_acc(s, y, t))
        ax.axvline(best, color=MUTED, lw=1, ls="--")
        ax.text(best - .12, .62, "ngưỡng\ntốt nhất", fontsize=7.2, color=MUTED, ha="right", transform=ax.get_xaxis_transform())
        r = np.argsort(np.argsort(s)); auc = (r[:len(p)].sum() - len(p) * (len(p) - 1) / 2 - len(p)) / (len(p) * len(n))
        ax.set_title(f"{title}\nAUC {auc:.2f}, độ chính xác cân bằng ở ngưỡng 0.5 là {bal_acc(s, y, 0):.2f}", fontsize=8, loc="left", pad=8)
        ax.set_xlabel("logit"); ax.set_yticks([])
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7.6, loc="lower center", ncol=2, bbox_to_anchor=(.5, -.06))
    fig.tight_layout(); fig.savefig(FIG / "intro_two_errors.png"); plt.close(fig)


def real():
    """A real attribute on unseen PETA: logits of positives and negatives, threshold 0.5 and the best threshold."""
    sys.path.insert(0, str(ROOT / "par"))
    from upar.data import attr_names, load_split
    A = attr_names(); va = load_split("val")
    z = np.load(ROOT / "runs" / "ft_PETA_dinov3_q" / "held_val_logits_last.npz")
    y = va[A].values[z["idx"]].astype(float)
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.5), sharey=False)
    for ax, name, vi in zip(axs, ["UpperBody-Color-Grey", "Accessory-Hat"], ["áo xám", "mũ"]):
        k = A.index(name); s = z["logits"][:, k]; t = y[:, k]
        bins = np.linspace(np.percentile(s, .5), np.percentile(s, 99.5), 50)
        ax.hist(s[t == 0], bins, color=NEG, alpha=.55, density=True, label="nhãn: không")
        ax.hist(s[t == 1], bins, color=POS, alpha=.55, density=True, label="nhãn: có")
        best = max(np.linspace(bins[0], bins[-1], 301), key=lambda th: bal_acc(s, t, th))
        ax.axvline(0, color=INK, lw=1.2); ax.axvline(best, color=MUTED, lw=1, ls="--")
        r = np.argsort(np.argsort(s)); npos = int(t.sum()); nneg = len(t) - npos
        auc = (r[t == 1].sum() - npos * (npos - 1) / 2 - npos) / (npos * nneg)
        ax.set_title(f"{vi}: AUC {auc:.2f}, độ chính xác cân bằng {bal_acc(s, t, 0):.2f} ở ngưỡng 0.5,\n"
                     f"{bal_acc(s, t, best):.2f} ở ngưỡng tốt nhất (đường đứt)", fontsize=8, loc="left")
        ax.set_xlabel("logit"); ax.set_yticks([])
        print(name, round(auc, 3), round(bal_acc(s, t, 0), 3), round(bal_acc(s, t, best), 3), round(best, 2), npos)
    axs[0].legend(frameon=False, fontsize=7.4)
    fig.tight_layout(); fig.savefig(FIG / "intro_real_threshold.png"); plt.close(fig)


toy(); real()
