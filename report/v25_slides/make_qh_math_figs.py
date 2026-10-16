"""Figures of the two query-head 'math' slides, from figs/qh_math.json (make_qh_math.py, real v25 H+ member).

../../../Upar-Challenge-Track-2/.venv/bin/python make_qh_math_figs.py
qh_score.png  left: the example image with its 14 x 7 patch grid coloured by the hat query's attention (mean of the
              8 heads), top patches labelled; right: raw scores of the most concentrated head vs the softmax weights
qh_mix.png    the hat logit with the same classifier under different mixes of patches (real attention, top-4 patches
              only, uniform average, lower body only) and on another image without a hat
qh_tile1..4   the four most attended 16 x 16 patches, enlarged
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, "/media/tan/F/RWS2027/Upar-Challenge-Track-2/par")
from upar.data import DATA, load_split  # noqa: E402

OUT = Path(__file__).parent / "figs"
INK, PAPER, MUTE, BLUE, ORANGE, LORANGE, GRID = "#14202E", "#F5F3EE", "#6B7684", "#2F7FB5", "#B85A1E", "#F0A868", "#D5DBE1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 20, "axes.edgecolor": MUTE, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER})
d = json.load(open(OUT / "qh_math.json"))
va = load_split("val")
img = Image.open(DATA / va.image.values[30465]).convert("RGB").resize((112, 224), Image.BICUBIC)
w = np.array(d["patch_weight_mean"]).reshape(14, 7); top = d["top4"]

# ---- score -> weight ----
fig = plt.figure(figsize=(14.4, 6.0))
ax0 = fig.add_axes([0.0, 0.04, 0.24, 0.92])
ax0.imshow(np.asarray(img), extent=(0, 7, 14, 0)); ax0.set_xticks([]); ax0.set_yticks([])
for s in ax0.spines.values():
    s.set_visible(False)
m = w / w.max()
for r in range(14):
    for c in range(7):
        ax0.add_patch(Rectangle((c, r), 1, 1, facecolor=ORANGE, alpha=0.75 * m[r, c] ** 0.6, edgecolor="#FFFFFF", lw=0.6))
for j in top:
    r, c = divmod(j, 7)
    ax0.add_patch(Rectangle((c, r), 1, 1, fill=False, edgecolor=INK, lw=2.5))
    ax0.text(c + 0.5, r + 0.55, f"{w[r, c] * 100:.0f}", ha="center", va="center", fontsize=13, color="#FFFFFF", fontweight="bold")
sc = np.array(d["head_score"]); hw = np.array(d["head_weight"])
order = np.argsort(-sc)
ax1 = fig.add_axes([0.31, 0.16, 0.30, 0.74])
x = np.arange(98)
ax1.bar(x, sc[order], color=[ORANGE if k == 0 else "#9DB3C6" for k in range(98)], width=0.9)
ax1.set_xticks([0, 49, 97]); ax1.set_xticklabels(["1", "50", "98"]); ax1.set_xlabel("98 mảnh, xếp theo điểm")
ax1.set_title(f"Điểm của head {d['head'] + 1} với từng mảnh", fontsize=19, color=INK, loc="left")
gap = sc[order[0]] - sc[order[1]]
ax1.annotate(f"hơn mảnh thứ hai {gap:.1f} điểm\n→ nhận {hw[order[0]] * 100:.1f}% trọng số".replace(".", ","), (1, sc[order[0]]),
             xytext=(18, sc[order[0]] - 8), fontsize=17, color=INK, arrowprops=dict(arrowstyle="->", color=INK))
ph = d["per_head"]
for h in range(8):
    axh = fig.add_axes([0.655 + (h % 4) * 0.087, 0.53 - (h // 4) * 0.47, 0.075, 0.40])
    axh.imshow(np.asarray(img), extent=(0, 7, 14, 0)); axh.set_xticks([]); axh.set_yticks([])
    for sp in axh.spines.values():
        sp.set_visible(False)
    r, c = divmod(ph[h]["patch"], 7)
    axh.add_patch(Rectangle((c, r), 1, 1, fill=False, edgecolor=ORANGE, lw=2.5))
    axh.set_title(f"head {h + 1}\n{ph[h]['weight'] * 100:.0f}%", fontsize=15, color=INK)
fig.savefig(OUT / "qh_score.png", dpi=100); plt.close(fig)
print("head", d["head"] + 1, "gap", round(float(gap), 2), "top weight", round(float(hw[order[0]]), 4), [(h + 1, x["patch"], round(x["weight"], 3)) for h, x in enumerate(ph)])
print("mean-head top4 weights %", [round(w.ravel()[j] * 100, 1) for j in top], "sum", round(float(w.ravel()[top].sum()) * 100, 1))

# ---- the same classifier on different mixes ----
rows = [("Tỉ lệ thật của query mũ", d["30465"]["logit"]),
        ("Chỉ 4 mảnh quanh đầu", d["logit_top4"]), ("Chia đều 98 mảnh", d["logit_uniform"]),
        ("Chỉ nửa dưới cơ thể", d["logit_lower"]), ("Ảnh khác, không đội mũ", d["19023"]["logit"])]
fig, ax = plt.subplots(figsize=(14, 5.6))
for i, (n, v) in enumerate(rows[::-1]):
    c = ORANGE if v > 0 else BLUE
    ax.barh(i, v, color=c, height=0.6)
    p = 1 / (1 + np.exp(-v))
    ax.text(v + (0.15 if v > 0 else -0.15), i, f"{v:+.1f}  ·  {p * 100:.0f}% có mũ".replace(".", ","), va="center",
            ha="left" if v > 0 else "right", fontsize=19, color=INK)
ax.axvline(0, color=MUTE, lw=2.5, ls="--")
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]])
ax.set_xlim(-9.5, 8.5); ax.set_xlabel("Điểm (logit) của thuộc tính mũ, âm là “không”, dương là “có”")
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
fig.tight_layout(); fig.savefig(OUT / "qh_mix.png", dpi=100); plt.close(fig)
for n, v in rows:
    print(f"{n:30s} {v:+.2f}")
for k, j in enumerate(top, 1):
    r, c = divmod(j, 7)
    img.crop((c * 16, r * 16, c * 16 + 16, r * 16 + 16)).resize((128, 128), Image.NEAREST).save(OUT / f"qh_tile{k}.png")
print("ok")
