"""Figures of the v31 paper from real model outputs (analysis/v31_min.json, exported from the submitted du.pt on
PETA val image 11178 by report/v31_arch/export.py).

../../../Upar-Challenge-Track-2/.venv/bin/python make_figs.py
figs/arch.pdf     architecture with this image: patch grids, token PCA, attention of the three heads for "hat", logits
figs/routing.pdf  the 8 attention heads of the joint query: each reads one encoder
"""
import base64
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE.parent / "figs"
D = json.load(open(HERE / "v31_min.json"))
A = D["attrs"]
HP, CL, JT, INK, MUTE, FROZ = "#2457D6", "#C9761E", "#7B3FC6", "#16202C", "#5A6676", "#9AA3AF"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "text.color": INK, "axes.edgecolor": INK})
img = np.asarray(Image.open(io.BytesIO(base64.b64decode(D["input"].split(",")[1]))).convert("RGB")) / 255.0  # 224 x 112


def overlay(ax, vals, cols, rows, color, extent, gamma=.75, alpha=.85):
    """input image under an attention map on a cols x rows patch grid"""
    ax.imshow(img, extent=extent, zorder=2)
    v = np.asarray(vals, float).reshape(rows, cols)
    v = (v / max(v.max(), 1e-9)) ** gamma
    rgba = np.zeros((rows, cols, 4)); rgba[..., :3] = matplotlib.colors.to_rgb(color); rgba[..., 3] = v * alpha
    ax.imshow(rgba, extent=extent, zorder=3, interpolation="nearest")
    ax.add_patch(Rectangle((extent[0], extent[2]), extent[1] - extent[0], extent[3] - extent[2], fill=False, lw=.5, ec=MUTE, zorder=4))


def grid(ax, cols, rows, color, extent):
    ax.imshow(img, extent=extent, zorder=2)
    x0, x1, y0, y1 = extent
    for i in range(cols + 1):
        ax.plot([x0 + i * (x1 - x0) / cols] * 2, [y0, y1], color=color, lw=.35, zorder=3)
    for j in range(rows + 1):
        ax.plot([x0, x1], [y0 + j * (y1 - y0) / rows] * 2, color=color, lw=.35, zorder=3)


def box(ax, x, y, w, h, ec, fc="white", lw=.8, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8", ec=ec, fc=fc, lw=lw, ls=ls, zorder=1))


def arrow(ax, x0, y0, x1, y1, color=MUTE, ls="-", rad=0.0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=5,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=.7, ls=ls, shrinkA=0, shrinkB=0, mutation_scale=6,
                                connectionstyle=f"arc3,rad={rad}"))


def stack(ax, x0, y0, w, h, n, nfrozen, color):
    sw = w / n
    for i in range(n):
        frozen = i < nfrozen
        ax.add_patch(Rectangle((x0 + i * sw, y0), sw * .78, h, fc="#E4E7EC" if frozen else color, ec=FROZ if frozen else color,
                               lw=.3, hatch="////" if frozen else None, alpha=1 if frozen else .85, zorder=2))


def pca(ax, rgb, cols, rows, extent, color):
    a = np.asarray(rgb, float).reshape(rows, cols, 3) / 255
    ax.imshow(a, extent=extent, interpolation="nearest", zorder=2)
    ax.add_patch(Rectangle((extent[0], extent[2]), extent[1] - extent[0], extent[3] - extent[2], fill=False, lw=.8, ec=color, zorder=4))


# ---------------------------------------------------------------- Figure 1: architecture on one image
k = A.index("Accessory-Hat")
fig = plt.figure(figsize=(7.0, 2.85))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 40.7); ax.axis("off")
ax.imshow(img, extent=(1.4, 5.4, 16, 24), zorder=2)
ax.text(3.4, 14.6, "input\n224×112", ha="center", va="top", fontsize=6.5)
grid(ax, 7, 14, HP, (7.6, 11.6, 27.5, 35.5)); grid(ax, 8, 16, CL, (7.6, 11.6, 4.5, 12.5))
ax.text(9.6, 26.9, "16 px patches:\n98 + CLS\n+ 4 registers", ha="center", va="top", fontsize=5.2, color=HP)
ax.text(9.6, 13.1, "14 px patches:\n128 + CLS", ha="center", va="bottom", fontsize=5.2, color=CL)
arrow(ax, 5.6, 22.5, 7.4, 30.5, rad=-.25); arrow(ax, 5.6, 17.5, 7.4, 9.5, rad=.25)
stack(ax, 13.6, 27.5, 15.6, 8, 32, 8, HP); stack(ax, 13.6, 4.5, 15.6, 8, 24, 8, CL)
ax.text(13.6, 36.4, "DINOv3 ViT-H+ · 32 blocks", ha="left", va="bottom", fontsize=6.2, color=HP, weight="bold")
ax.text(13.6, 3.6, "CLIP ViT-L · 24 blocks", ha="left", va="top", fontsize=6.2, color=CL, weight="bold")
ax.text(21.4, 20, "first 8 blocks frozen (hatched)\nall other blocks fine-tuned\njointly by the three losses", ha="center", va="center", fontsize=5.8, color=MUTE)
arrow(ax, 11.8, 31.5, 13.4, 31.5); arrow(ax, 11.8, 8.5, 13.4, 8.5)
pca(ax, D["pca_h"], 7, 14, (31.2, 35.2, 27.5, 35.5), HP); pca(ax, D["pca_c"], 8, 16, (31.2, 35.2, 4.5, 12.5), CL)
ax.text(33.2, 36.4, "tokens\n(PCA)", ha="center", va="bottom", fontsize=5.8)
arrow(ax, 29.4, 31.5, 31.0, 31.5); arrow(ax, 29.4, 8.5, 31.0, 8.5)
# heads
for (y, ec, fc, name, sub) in [(28.0, HP, "#EEF3FD", "H+ query head", "40 queries, 8-head cross-attn\non 99 H+ tokens (1280-d)"),
                               (14.8, JT, "#F4EEFB", "joint query head", "40 queries, 8-head cross-attn\non all 228 tokens (1024-d)"),
                               (1.6, CL, "#FCF3E8", "CLIP query head", "40 queries, 8-head cross-attn\non 129 tokens + text score")]:
    box(ax, 39.2, y, 27.6, 11.2, ec, fc)
    ax.text(51.0, y + 10.0, name, fontsize=6.6, weight="bold", color=ec, va="top")
    ax.text(51.0, y + 7.4, sub, fontsize=5.2, color=MUTE, va="top", linespacing=1.25)
overlay(ax, D["att_h"][k][1:], 7, 14, HP, (40.4, 44.4, 29.6, 37.6))
overlay(ax, D["att_j"][k][1:99], 7, 14, HP, (40.4, 44.4, 16.4, 24.4)); overlay(ax, D["att_j"][k][100:], 8, 16, CL, (45.6, 49.6, 16.4, 24.4))
overlay(ax, D["att_c"][k][1:], 8, 16, CL, (40.4, 44.4, 3.2, 11.2))
ax.text(51.0, 30.4, "attention of the\n“hat” query", fontsize=5.2, color=INK, style="italic", va="center")
ax.text(51.0, 17.2, "“hat” query: H+ part\nand CLIP part", fontsize=5.2, color=INK, style="italic", va="center")
ax.text(51.0, 4.0, "attention of the\n“hat” query", fontsize=5.2, color=INK, style="italic", va="center")
# projections into the joint head
box(ax, 36.2, 18.9, 2.6, 3.4, JT, "white", lw=.6)
ax.text(37.5, 20.6, "W", ha="center", va="center", fontsize=6, color=JT)
arrow(ax, 35.4, 28.4, 37.2, 22.4, color=HP, rad=.15); arrow(ax, 35.4, 12.0, 37.2, 18.8, color=CL, rad=-.15)
arrow(ax, 38.9, 20.6, 39.1, 20.6, color=JT)
ax.text(37.6, 25.0, "linear map to\n1024-d + source\nembedding", ha="center", va="center", fontsize=4.6, color=JT)
arrow(ax, 35.4, 31.5, 39.1, 33.6, color=HP); arrow(ax, 35.4, 8.5, 39.1, 7.2, color=CL)
# logits -> mean -> probability
zh, zj, zc = D["logits"]["h"][k], D["logits"]["j"][k], D["logits"]["c"][k]
for (y, c, n, z) in [(33.6, HP, "z_H", zh), (20.4, JT, "z_J", zj), (7.2, CL, "z_C", zc)]:
    arrow(ax, 66.9, y, 68.7, y, color=c)
    ax.add_patch(plt.Circle((70.8, y), 2.0, fc="white", ec=c, lw=.9, zorder=4))
    ax.text(70.8, y, f"{z:.1f}", ha="center", va="center", fontsize=6.2, zorder=5)
    ax.text(70.8, y + 2.7, f"${n[0]}_{n[2]}$", ha="center", va="bottom", fontsize=6, color=c)
    arrow(ax, 72.9, y, 77.6, 20.4, color=MUTE, rad=0)
ax.add_patch(plt.Circle((79.6, 20.4), 2.1, fc=INK, ec=INK, zorder=4))
ax.text(79.6, 20.4, "avg", ha="center", va="center", color="white", fontsize=6, zorder=5)
arrow(ax, 81.8, 20.4, 84.2, 20.4)
pf = D["prob"][k]
ax.text(84.6, 21.6, rf"$\sigma(\bar z)={pf:.3f}$", fontsize=6.6, va="bottom")
ax.text(84.6, 19.4, r"$> 0.5$: hat present", fontsize=6.2, va="top")
ax.text(84.6, 16.2, "same rule and threshold\nfor all 40 attributes", fontsize=5.2, va="top", color=MUTE)
box(ax, 74.4, 28.6, 25.0, 10.6, "#C2304A", "#FBEFF1", ls=(0, (3, 2)))
ax.text(75.4, 38.2, "training", fontsize=6.4, weight="bold", color="#C2304A", va="top")
ax.text(75.4, 35.6, r"$L=\sum_{k\in\{H,J,C\}}\ \ell_{\mathrm{LA}}(z_k, y)$", fontsize=6.6, va="top")
ax.text(75.4, 32.3, "logit-adjusted BCE, DeepMAR\nweights, masked unlabeled cells", fontsize=5.2, va="top", color=MUTE)
box(ax, 74.4, 1.6, 25.0, 9.0, INK, "#F3F5F8", ls=(0, (3, 2)))
ax.text(75.4, 9.6, "inference", fontsize=6.4, weight="bold", va="top")
ax.text(75.4, 7.0, "raw logits, no offset, no\nper-attribute threshold,\n2 encoder passes per image", fontsize=5.2, va="top", color=MUTE)
fig.savefig(OUT / "arch.pdf"); fig.savefig(OUT / "arch.png", dpi=220); plt.close(fig)

# ---------------------------------------------------------------- Figure 2: routing of the joint query
names = [("Accessory-Hat", "hat"), ("UpperBody-Color-White", "white top"), ("LowerBody-Color-Black", "black lower body")]
fig, axs = plt.subplots(len(names), 8, figsize=(3.35, 2.55), gridspec_kw=dict(wspace=.08, hspace=.42, left=.01, right=.99, top=.9, bottom=.02))
heads = np.asarray(D["att_j_heads"]) if "att_j_heads" in D else None
for r, (a, lab) in enumerate(names):
    i = A.index(a); nh = 0
    for h in range(8):
        w = np.asarray(heads[i][h]); ax = axs[r, h]; ax.axis("off")
        if w[:99].sum() > .5:
            overlay(ax, w[1:99], 7, 14, HP, (0, 4, 0, 8), alpha=.9); nh += 1; c = HP
        else:
            overlay(ax, w[100:], 8, 16, CL, (0, 4, 0, 8), alpha=.9); c = CL
        ax.set_xlim(-.2, 4.2); ax.set_ylim(-.2, 8.2)
        ax.set_title(f"{h + 1}", fontsize=5.5, color=c, pad=1.5)
    fig.text(.01, axs[r, 0].get_position().y1 + .045, f"“{lab}”: {nh} heads read H+ tokens (blue), {8 - nh} read CLIP tokens (orange)",
             fontsize=5.8, ha="left")
fig.savefig(OUT / "routing.pdf"); fig.savefig(OUT / "routing.png", dpi=220); plt.close(fig)
print("figures written")
