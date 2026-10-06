"""New figures of the Track 1 paper (track1_paper.tex). Older figures are copied from ../figs.
Data: offset_proxy_last.json of every leave-one-dataset-out run under Upar-Challenge-Track-2/runs/remote*/ and the
teacher-ensemble / distillation measurements of 2026-10-04 (logged in the session, copied here as constants).
Run from report/paper/: python make_figs.py"""
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
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e3e2db", "grid.linewidth": .6, "axes.edgecolor": "#9a988f",
                     "axes.axisbelow": True, "savefig.dpi": 220, "savefig.bbox": "tight"})
INK, MUTED, BLUE, ORANGE, AQUA, YELLOW, GRAY, RED = "#121211", "#85837c", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#b4b2a9", "#e34948"


def fig_thresholds():
    """Theory rule (tau = 0.42) vs fitted offsets x0.6 on every LODO run, and the mean curves of both families."""
    rows = []
    for f in sorted(glob(str(RUNS / "remote*" / "*" / "offset_proxy_last.json"))):
        d = json.load(open(f))
        held = json.load(open(f.replace("offset_proxy_last.json", "args.json")))["holdout"]
        rows.append((held, d))
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.15, 1]})
    s = [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0]
    t = [0.0, 0.1, 0.2, 0.3, 0.42, 0.5]
    peta = [d for h, d in rows if h == "PETA"]
    fo = np.array([[d[f"fitted offsets x {v:.2f}"]["Challenge_Avg"] for v in s] for d in peta])
    th = np.array([[d["fitted offsets x 0.00"]["Challenge_Avg"]] + [d[f"theory tau = {v:.2f}"]["Challenge_Avg"] for v in t[1:]] for d in peta])
    base = fo[:, :1]
    axs[0].plot(s, (fo - base).mean(0), color=ORANGE, lw=1.8, marker="o", ms=4, label="offset fit trên miền nguồn × s")
    axs[0].plot(t, (th - base).mean(0), color=BLUE, lw=1.8, marker="o", ms=4, label="quy tắc lý thuyết × τ")
    axs[0].axhline(0, color=MUTED, lw=.8)
    axs[0].set_xlabel("hệ số s hoặc τ"); axs[0].set_ylabel("Avg so với ngưỡng 0.5 (PETA)")
    axs[0].legend(frameon=False, fontsize=7.6, loc="lower right")
    axs[0].set_title(f"Trung bình {len(peta)} lần chạy giấu PETA", fontsize=8.6, loc="left")
    cols = {"PETA": BLUE, "Market1501": AQUA, "PA100k": YELLOW}
    diffs = {h: [] for h in cols}
    for h, d in rows:
        diffs[h].append(d["theory tau = 0.42"]["Challenge_Avg"] - d["fitted offsets x 0.60"]["Challenge_Avg"])
    y = 0
    for h, v in diffs.items():
        v = np.array(v)
        axs[1].scatter(v, y + np.random.default_rng(0).uniform(-.18, .18, len(v)), s=16, color=cols[h], edgecolor="white", lw=.6, zorder=3)
        axs[1].text(0.0105, y, f"{(v > 0).sum()}/{len(v)} thắng", va="center", fontsize=7.6, color=INK)
        y -= 1
    axs[1].axvline(0, color=MUTED, lw=.8)
    axs[1].set_yticks([0, -1, -2], [f"giấu {h}" for h in cols]); axs[1].tick_params(axis="y", length=0)
    axs[1].set_xlabel("Avg(τ = 0.42) − Avg(offset × 0.6)"); axs[1].grid(axis="y", visible=False)
    axs[1].set_xlim(-0.006, 0.0135)
    fig.tight_layout(); fig.savefig(FIG / "thresholds.png"); plt.close(fig)
    n = sum(len(v) for v in diffs.values()); w = sum((np.array(v) > 0).sum() for v in diffs.values())
    print(f"theory beats offsets in {w}/{n} runs")


# DINOv3-L + X, logits averaged; PETA held out; Avg after offsets x0.6. Correlation = mean per-attribute Pearson
# correlation of X's held-out logits with DINOv3-L's. Standalone = X alone, same protocol.
ENS = [  # name, corr, standalone, ensemble Avg@0.6, ensemble AUC
    ("CLIP-L (OpenAI)", .861, .8063, .8134, .9203), ("MetaCLIP2-H", .847, .8035, .8113, .9239),
    ("PE-Core-L", .857, .7957, .8092, .9193), ("SigLIP2-SO400M", .864, .7973, .8087, .9201),
    ("ConvNeXt-L (DINOv3)", .876, .7967, .8067, .9182), ("MetaCLIP2-L", .842, .7900, .8059, .9180),
    ("SigLIP2-L", .864, .7917, .8054, .9185), ("AIMv2-L", .823, .7815, .8048, .9176),
    ("EVA02-L", .782, .7702, .8002, .9143), ("DINOv3-L 224×112", .956, .8061, .8073, .9175)]
DINO = .8026


def fig_complementarity():
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for name, c, alone, ens, auc in ENS:
        col = BLUE if name.startswith(("CLIP", "MetaCLIP2-H")) else (RED if name.startswith("DINOv3") else INK)
        axs[0].scatter(c, ens - DINO, s=22, color=col, edgecolor="white", lw=.6, zorder=3)
        off = {"DINOv3-L 224×112": (-62, 6), "PE-Core-L": (-44, 5), "ConvNeXt-L (DINOv3)": (5, 3), "SigLIP2-L": (5, -9), "MetaCLIP2-L": (-62, 4)}.get(name, (4, 2))
        axs[0].annotate(name, (c, ens - DINO), xytext=off, textcoords="offset points", fontsize=6.6, color=INK)
        axs[1].scatter(alone, ens - DINO, s=22, color=col, edgecolor="white", lw=.6, zorder=3)
    axs[0].set_xlabel("tương quan logit với DINOv3-L"); axs[0].set_ylabel("Avg(DINOv3-L + X) − Avg(DINOv3-L)")
    axs[1].set_xlabel("Avg của X khi đứng một mình")
    axs[1].axvline(DINO, color=MUTED, lw=.8); axs[1].text(DINO, .0112, " DINOv3-L", fontsize=6.8, color=MUTED)
    for a in axs:
        a.axhline(0, color=MUTED, lw=.8)
    fig.tight_layout(); fig.savefig(FIG / "complementarity.png"); plt.close(fig)


def fig_transfer():
    """Where the teacher-ensemble gain goes: teachers, ViT-L student, ViT-B student (PETA held out, Avg@0.6)."""
    labels = ["DINOv3-L\n(thầy 1)", "CLIP-L\n(thầy 2)", "trung bình\nhai thầy", "trò ViT-L\nkhông KD", "trò ViT-L\nKD 2 thầy",
              "trò ViT-B\nkhông KD", "trò ViT-B\nKD 2 thầy", "trò ViT-B\nKD 3 thầy"]
    vals = [.8026, .8063, .8134, .8033, .8059, .7931, .7969, .7966]
    cols = [GRAY, GRAY, BLUE, MUTED, ORANGE, MUTED, ORANGE, ORANGE]
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    x = np.arange(len(vals))
    ax.bar(x, np.array(vals) - .79, bottom=.79, color=cols, width=.62, edgecolor="white", lw=2)
    for i, v in enumerate(vals):
        ax.text(i, v + .0004, f"{v:.4f}", ha="center", fontsize=7.4, color=INK)
    ax.set_xticks(x, labels, fontsize=7.3); ax.set_ylim(.79, .8165); ax.set_ylabel("Avg trên PETA (offset × 0.6)")
    ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(FIG / "transfer.png"); plt.close(fig)


# Leaderboard (hidden test) in submission order; None = run did not finish within the time limit.
LB = [("v4b", "ViT-B linear", .7775), ("v5", "ViT-B linear, offset × 1.0", .7766), ("v6", "C-RADIOv3-B", .7757),
      ("KD-B", "ViT-B học từ ViT-L", .7717), ("v8", "ViT-B query", .7750), ("v9", "ViT-L 176 query", .7857),
      ("v10", "ViT-L 176, train+val", .7838), ("v12", "ViT-L 176, KD 2 thầy", .7896), ("v13", "v12 + ngưỡng lý thuyết", .7912),
      ("v14", "soup + dự phòng thời gian", .5509), ("v17", "ViT-L kd-mix", None), ("v18", "ViT-B kd-mix", .7823),
      ("v19", "v13 tăng tốc 7%", None)]


def fig_leaderboard():
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    for i, (tag, lab, v) in enumerate(LB):
        if v is None:
            ax.scatter(i, .7705, marker="x", color=RED, s=30, zorder=3)
            ax.text(i, .7712, "quá giờ", ha="center", fontsize=6.6, color=RED)
        elif v < .77:
            ax.scatter(i, .7705, marker="v", color=RED, s=26, zorder=3)
            ax.text(i, .7712, f"{v:.4f}", ha="center", fontsize=6.6, color=RED)
        else:
            c = BLUE if v == max(x for _, _, x in LB if x) else INK
            ax.scatter(i, v, s=24, color=c, zorder=3, edgecolor="white", lw=.6)
            ax.text(i, v + .0007, f"{v:.4f}", ha="center", fontsize=6.6, color=INK)
    ax.set_xticks(range(len(LB)), [f"{t}\n{l}" for t, l, _ in LB], fontsize=6.0, rotation=40, ha="right")
    ax.set_ylim(.769, .7935); ax.set_ylabel("Challenge_Avg (tập chấm ẩn)"); ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(FIG / "leaderboard.png"); plt.close(fig)


if __name__ == "__main__":
    fig_thresholds()
    fig_complementarity()
    fig_transfer()
    fig_leaderboard()
