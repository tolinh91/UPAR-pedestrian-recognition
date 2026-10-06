"""Figures of the English Track 1 paper. Run from report/paper_en/ with the Track 2 venv:
../../../Upar-Challenge-Track-2/.venv/bin/python make_figs.py
Data: offset_proxy_last.json of every leave-one-dataset-out run under Upar-Challenge-Track-2/runs/remote*/, and the
teacher-pairing measurements of 2026-10-04 (constants below)."""
import json
from glob import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
FIG = HERE / "figs"
RUNS = HERE.parents[2] / "Upar-Challenge-Track-2" / "runs"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e3e2db", "grid.linewidth": .5, "axes.edgecolor": "#9a988f",
                     "axes.axisbelow": True, "savefig.dpi": 300, "savefig.bbox": "tight", "axes.titlesize": 7.5,
                     "legend.fontsize": 6.5, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8})
INK, MUTED, BLUE, ORANGE, AQUA, YELLOW, RED = "#121211", "#85837c", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948"


def fig_thresholds():
    rows = []
    for f in sorted(glob(str(RUNS / "remote*" / "*" / "offset_proxy_last.json"))):
        # runs with saved held-out logits only; clipL_168_PETA is a byte-identical copy of T_clipL_PETA
        if not Path(f).with_name("held_val_logits_last.npz").exists() or Path(f).parent.name == "clipL_168_PETA":
            continue
        rows.append((json.load(open(f.replace("offset_proxy_last.json", "args.json")))["holdout"], json.load(open(f))))
    s = [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0]
    t = [0.0, 0.1, 0.2, 0.3, 0.42, 0.5]
    peta = [d for h, d in rows if h == "PETA"]
    fo = np.array([[d[f"fitted offsets x {v:.2f}"]["Challenge_Avg"] for v in s] for d in peta])
    th = np.array([[d["fitted offsets x 0.00"]["Challenge_Avg"]] + [d[f"theory tau = {v:.2f}"]["Challenge_Avg"] for v in t[1:]] for d in peta])
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 1.9), gridspec_kw={"width_ratios": [1, 1.15]})
    axs[0].plot(s, (fo - fo[:, :1]).mean(0) * 100, color=ORANGE, lw=1.4, marker="o", ms=3, label="fitted offsets $\\times s$")
    axs[0].plot(t, (th - fo[:, :1]).mean(0) * 100, color=BLUE, lw=1.4, marker="o", ms=3, label="label-shift rule $\\times\\tau$")
    axs[0].axhline(0, color=MUTED, lw=.6)
    axs[0].set_xlabel("scale $s$ or $\\tau$"); axs[0].set_ylabel("$\\Delta$Avg vs. threshold 0.5 (pts)")
    axs[0].legend(frameon=False, loc="lower right"); axs[0].set_title(f"(a) mean over {len(peta)} PETA-held-out runs", loc="left")
    cols = {"PETA": BLUE, "Market1501": AQUA, "PA100k": YELLOW}
    rng = np.random.default_rng(0)
    for y, h in enumerate(cols):
        v = np.array([d["theory tau = 0.42"]["Challenge_Avg"] - d["fitted offsets x 0.75"]["Challenge_Avg"] for hh, d in rows if hh == h]) * 100
        axs[1].scatter(v, -y + rng.uniform(-.17, .17, len(v)), s=9, color=cols[h], edgecolor="white", lw=.4, zorder=3)
        axs[1].text(1.03, -y, f"{(v > 0).sum()}/{len(v)}", va="center", fontsize=6.8, transform=axs[1].get_yaxis_transform())
    axs[1].axvline(0, color=MUTED, lw=.6)
    axs[1].set_yticks([0, -1, -2], [f"{h} held out" for h in cols]); axs[1].tick_params(axis="y", length=0)
    axs[1].set_xlabel("Avg($\\tau$=0.42) $-$ Avg(offsets $\\times$0.75) (pts)"); axs[1].grid(axis="y", visible=False)
    axs[1].set_xlim(-1.0, 1.25); axs[1].set_title("(b) per run; wins / runs on the right", loc="left")
    fig.tight_layout(w_pad=1.2); fig.savefig(FIG / "thresholds.pdf"); plt.close(fig)


ENS = [("CLIP-L", .861, .8063, .8134), ("MetaCLIP2-H", .847, .8035, .8113), ("PE-Core-L", .857, .7957, .8092),
       ("SigLIP2-So400m", .864, .7973, .8087), ("ConvNeXt-L", .876, .7967, .8067), ("MetaCLIP2-L", .842, .7900, .8059),
       ("SigLIP2-L", .864, .7917, .8054), ("AIMv2-L", .823, .7815, .8048), ("EVA02-L", .782, .7702, .8002),
       ("DINOv3-L@224", .956, .8061, .8073)]
DINO = .8026


def fig_pairing():
    """Pair gain against the partner's own accuracy, with the least-squares fit on accuracy and correlation."""
    others = [e for e in ENS if not e[0].startswith("DINO")]
    acc = np.array([e[2] for e in others]); c = np.array([e[1] for e in others]); g = np.array([e[3] - DINO for e in others]) * 100
    X = np.c_[np.ones(len(acc)), acc, c]; b = np.linalg.lstsq(X, g, rcond=None)[0]
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    xs = np.linspace(acc.min() - .002, acc.max() + .003, 50)
    ax.plot(xs, b[0] + b[1] * xs + b[2] * c.mean(), color=MUTED, lw=.8, ls="--", label="fit at mean correlation")
    off = {"MetaCLIP2-L": (-58, 3), "SigLIP2-L": (4, -7), "PE-Core-L": (-52, 3), "SigLIP2-So400m": (4, 3), "ConvNeXt-L": (-56, -9), "AIMv2-L": (-50, 2), "DINOv3-L@224": (-66, 4)}
    for name, cc, alone, ens in ENS:
        dino = name.startswith("DINO")
        ax.scatter(alone, (ens - DINO) * 100, s=22, color=RED if dino else BLUE, edgecolor="white", lw=.5, zorder=3)
        lab = {"CLIP-L": (-44, 3), "MetaCLIP2-H": (-58, 3), "EVA02-L": (4, 1), "DINOv3-L@224": (-30, -11)}
        if name in lab:
            ax.annotate(f"{name} ($\\rho$={cc:.2f})", (alone, (ens - DINO) * 100), xytext=lab[name], textcoords="offset points", fontsize=5.8)
    ax.axhline(0, color=MUTED, lw=.6)
    ax.set_xlabel("partner's own Avg (PETA held out)"); ax.set_ylabel("$\\Delta$Avg of the pair (pts)")
    ax.legend(frameon=False, loc="upper left", fontsize=6)
    fig.tight_layout(); fig.savefig(FIG / "pairing.pdf"); plt.close(fig)


if __name__ == "__main__":
    fig_thresholds(); fig_pairing()
