"""Figures of the query-head code walkthrough slides, from figs/qh_code.json (qh_code_numbers.py, real v25 H+).

../../../Upar-Challenge-Track-2/.venv/bin/python make_qh_code_figs.py
qh_dot.png  head 2 of the hat query against 4 patches: the 160 products query x key (divided by sqrt 160, as in the
            code), their sum = the score, and the softmax weight
qh_fc.png   the 1280 products feature x classifier weight of the hat attribute, sorted; positive and negative sums,
            bias, logit and probability
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, "/media/tan/F/RWS2027/Upar-Challenge-Track-2/par")
from upar.data import DATA, load_split  # noqa: E402

OUT = Path(__file__).parent / "figs"
INK, PAPER, MUTE, BLUE, ORANGE = "#14202E", "#F5F3EE", "#6B7684", "#2F7FB5", "#B85A1E"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 20, "axes.edgecolor": MUTE, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER})
d = json.load(open(OUT / "qh_code.json"))
va = load_split("val")
img = Image.open(DATA / va.image.values[30465]).convert("RGB").resize((112, 224), Image.BICUBIC)
vi = lambda x, k=1: f"{x:.{k}f}".replace(".", ",").replace("-", "−")  # noqa: E731

# ---- query x key, head 2 ----
q = np.array(d["q160"]); names = [("top", "Đỉnh đầu"), ("second", "Cạnh đỉnh đầu"), ("shoulder", "Vai"), ("shoe", "Giày")]
prods = {n: q * np.array(d["patches"][n]["k160"]) / np.sqrt(160) for n, _ in names}
lim = max(np.abs(p).max() for p in prods.values()) * 1.05
fig = plt.figure(figsize=(14, 6.4))
for i, (n, label) in enumerate(names):
    p = d["patches"][n]; y0 = 0.76 - i * 0.235
    axi = fig.add_axes([0.0, y0, 0.07, 0.2]); r, c = p["row"], p["col"]
    axi.imshow(np.asarray(img.crop((c * 16, r * 16, c * 16 + 16, r * 16 + 16)).resize((64, 64), Image.NEAREST)))
    axi.set_xticks([]); axi.set_yticks([])
    for sp in axi.spines.values():
        sp.set_visible(True); sp.set_color(ORANGE if n == "top" else MUTE); sp.set_linewidth(3 if n == "top" else 1)
    fig.text(0.085, y0 + 0.1, label, fontsize=20, color=INK, va="center")
    ax = fig.add_axes([0.275, y0 + 0.01, 0.45, 0.18])
    pr = prods[n]
    ax.bar(np.arange(160), pr, color=np.where(pr > 0, ORANGE, BLUE), width=1.0)
    ax.axhline(0, color=MUTE, lw=1); ax.set_ylim(-lim, lim); ax.set_xlim(-1, 160)
    ax.set_yticks([]); ax.set_xticks([] if i < 3 else [0, 80, 159]);
    if i == 3:
        ax.set_xticklabels(["1", "80", "160"]); ax.set_xlabel("160 tích, mỗi tích là một cặp số của query và token")
    for sp in ("left",):
        ax.spines[sp].set_visible(False)
    fig.text(0.745, y0 + 0.1, f"tổng {vi(p['score'])}", fontsize=21, color=INK, va="center", fontweight="bold" if n == "top" else "normal")
    fig.text(0.885, y0 + 0.1, f"→ {vi(p['weight'] * 100)}%", fontsize=21, color=ORANGE if n == "top" else INK, va="center",
             fontweight="bold" if n == "top" else "normal")
fig.text(0.745, 0.975, "điểm", fontsize=18, color=MUTE, va="center"); fig.text(0.885, 0.975, "tỉ lệ", fontsize=18, color=MUTE, va="center")
fig.savefig(OUT / "qh_dot.png", dpi=100); plt.close(fig)
print({n: (round(d["patches"][n]["score"], 2), round(float(prods[n].sum()), 2), round(d["patches"][n]["weight"], 4)) for n, _ in names})

# ---- feature x classifier weight, 1280 products ----
pr = np.sort(np.array(d["prod1280"]))[::-1]
fig = plt.figure(figsize=(14, 5.6))
ax = fig.add_axes([0.09, 0.2, 0.57, 0.72])
ax.bar(np.arange(1280), pr, color=np.where(pr > 0, ORANGE, BLUE), width=1.0)
ax.axhline(0, color=MUTE, lw=1); ax.set_xlim(-5, 1285)
ax.set_xticks([0, 640, 1279]); ax.set_xticklabels(["1", "640", "1280"])
ax.set_yticks([-0.05, 0, 0.05]); ax.set_yticklabels(["−0,05", "0", "0,05"])
ax.set_xlabel("1280 tích đặc trưng × hướng “có mũ”, xếp từ lớn đến nhỏ")
rows = [("phần dương", f"+{vi(d['prod_pos'], 2)}", ORANGE), ("phần âm", vi(d["prod_neg"], 2), BLUE),
        ("tổng 1280 tích", f"+{vi(d['prod_sum'], 2)}", INK), ("hệ số chệch", vi(d["bias"], 2), INK),
        ("logit", f"+{vi(d['logit'], 2)}", ORANGE), ("xác suất", f"{vi(d['prob'] * 100)}% có mũ", ORANGE)]
for i, (k, v, c) in enumerate(rows):
    y = 0.86 - i * 0.13
    fig.text(0.70, y, k, fontsize=21, color=MUTE, va="center")
    fig.text(0.985, y, v, fontsize=22, color=c, va="center", ha="right", fontweight="bold" if k in ("logit", "xác suất") else "normal")
fig.add_artist(plt.Line2D([0.70, 0.985], [0.86 - 2.5 * 0.13, 0.86 - 2.5 * 0.13], color=MUTE, lw=1))
fig.savefig(OUT / "qh_fc.png", dpi=100); plt.close(fig)
print("ok")
