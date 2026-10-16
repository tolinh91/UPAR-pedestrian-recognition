"""Figures of the v31 paper from real model outputs (analysis/v31_min.json, exported from the submitted du.pt on
PETA val image 11178 by report/v31_arch/export.py).

../../../Upar-Challenge-Track-2/.venv/bin/python make_figs.py
figs/arch.pdf     architecture with this image: patch grids, token PCA, attention of the three heads for "hat", logits
figs/routing.pdf  the 8 attention heads of the joint query: thumbnails for 3 attributes + encoder share for all 40
figs/pick.pdf     per attribute: joint head vs the two member heads (balanced accuracy, analysis/ablations.json)
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
fig = plt.figure(figsize=(7.0, 2.95))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 40.7); ax.axis("off")
ax.imshow(img, extent=(1.4, 5.4, 16, 24), zorder=2)
ax.text(0.4, 15.2, "input\n224×112", ha="left", va="top", fontsize=6.6)
grid(ax, 7, 14, HP, (7.6, 11.6, 27.5, 35.5)); grid(ax, 8, 16, CL, (7.6, 11.6, 4.5, 12.5))
ax.text(9.6, 26.9, "16-px patches\n98 + CLS + 4 reg.", ha="center", va="top", fontsize=6.3, color=HP)
ax.text(9.6, 13.1, "14-px patches\n128 + CLS", ha="center", va="bottom", fontsize=6.3, color=CL)
arrow(ax, 5.6, 22.5, 7.4, 30.5, rad=-.25); arrow(ax, 5.6, 17.5, 7.4, 9.5, rad=.25)
stack(ax, 13.6, 27.5, 15.6, 8, 32, 8, HP); stack(ax, 13.6, 4.5, 15.6, 8, 24, 8, CL)
ax.text(13.6, 36.3, "DINOv3 ViT-H+/16", ha="left", va="bottom", fontsize=7, color=HP, weight="bold")
ax.text(13.6, 3.7, "CLIP ViT-L/14", ha="left", va="top", fontsize=7, color=CL, weight="bold")
ax.text(29.2, 27.1, "32 blocks", ha="right", va="top", fontsize=6.3, color=HP)
ax.text(29.2, 12.9, "24 blocks", ha="right", va="bottom", fontsize=6.3, color=CL)
ax.text(21.4, 20, "hatched: frozen\nothers: fine-tuned\nby all three losses", ha="center", va="center", fontsize=6.3, color=MUTE)
arrow(ax, 11.8, 31.5, 13.4, 31.5); arrow(ax, 11.8, 8.5, 13.4, 8.5)
pca(ax, D["pca_h"], 7, 14, (31.2, 35.2, 27.5, 35.5), HP); pca(ax, D["pca_c"], 8, 16, (31.2, 35.2, 4.5, 12.5), CL)
ax.text(33.2, 36.3, "tokens", ha="center", va="bottom", fontsize=6.3)
arrow(ax, 29.4, 31.5, 31.0, 31.5); arrow(ax, 29.4, 8.5, 31.0, 8.5)
# heads: box, title, one line of what it reads, attention of its "hat" query on the left of the box
for (y, ec, fc, name, sub) in [(28.0, HP, "#EEF3FD", "H+ query head", "reads 99 H+ tokens"),
                               (14.8, JT, "#F4EEFB", "joint query head", "reads all 228 tokens,\nmapped by W to 1024-d"),
                               (1.6, CL, "#FCF3E8", "CLIP query head", "reads 129 CLIP tokens\n+ zero-shot text score")]:
    box(ax, 39.2, y, 27.6, 11.2, ec, fc)
    ax.text(51.0, y + 9.6, name, fontsize=7, weight="bold", color=ec, va="top")
    ax.text(51.0, y + 6.6, sub, fontsize=6.3, color=INK, va="top", linespacing=1.3)
overlay(ax, D["att_h"][k][1:], 7, 14, HP, (40.4, 44.4, 29.6, 37.6))
overlay(ax, D["att_j"][k][1:99], 7, 14, HP, (40.4, 44.4, 16.4, 24.4)); overlay(ax, D["att_j"][k][100:], 8, 16, CL, (45.6, 49.6, 16.4, 24.4))
overlay(ax, D["att_c"][k][1:], 8, 16, CL, (40.4, 44.4, 3.2, 11.2))
# projection W into the joint head
box(ax, 36.0, 18.7, 2.8, 3.8, JT, "white", lw=.7)
ax.text(37.4, 20.6, "W", ha="center", va="center", fontsize=7, color=JT, weight="bold")
arrow(ax, 35.4, 28.4, 37.0, 22.6, color=HP, rad=.15); arrow(ax, 35.4, 12.0, 37.0, 18.6, color=CL, rad=-.15)
arrow(ax, 38.9, 20.6, 39.1, 20.6, color=JT)
arrow(ax, 35.4, 31.5, 39.1, 33.6, color=HP); arrow(ax, 35.4, 8.5, 39.1, 7.2, color=CL)
# logits -> mean -> probability
zh, zj, zc = D["logits"]["h"][k], D["logits"]["j"][k], D["logits"]["c"][k]
for (y, c, n, z) in [(33.6, HP, "z_H", zh), (20.4, JT, "z_J", zj), (7.2, CL, "z_C", zc)]:
    arrow(ax, 66.9, y, 68.6, y, color=c)
    ax.add_patch(plt.Circle((70.8, y), 2.1, fc="white", ec=c, lw=.9, zorder=4))
    ax.text(70.8, y, f"{z:.1f}", ha="center", va="center", fontsize=7, zorder=5)
    ax.text(70.8, y + 2.6, f"${n[0]}_{n[2]}$", ha="center", va="bottom", fontsize=7, color=c)
    arrow(ax, 73.0, y, 77.5, 20.4, color=MUTE, rad=0)
ax.add_patch(plt.Circle((79.6, 20.4), 2.2, fc=INK, ec=INK, zorder=4))
ax.text(79.6, 20.4, "avg", ha="center", va="center", color="white", fontsize=6.8, zorder=5)
arrow(ax, 81.9, 20.4, 84.0, 20.4)
pf = D["prob"][k]
ax.text(84.4, 21.4, rf"$\sigma(\bar z)={pf:.3f}$", fontsize=7.2, va="bottom")
ax.text(84.4, 19.4, r"$> 0.5$: hat present", fontsize=7, va="top")
ax.text(84.4, 15.9, "same rule for\nall 40 attributes", fontsize=6.3, va="top", color=MUTE)
box(ax, 74.4, 28.4, 25.0, 11.0, "#C2304A", "#FBEFF1", ls=(0, (3, 2)))
ax.text(75.6, 38.3, "training", fontsize=7, weight="bold", color="#C2304A", va="top")
ax.text(75.6, 35.4, r"$L=\sum_{k\in\{H,J,C\}}\ \ell_{\mathrm{LA}}(z_k, y)$", fontsize=7.2, va="top")
ax.text(75.6, 31.9, "logit-adjusted BCE\nwith DeepMAR weights", fontsize=6.3, va="top", color=MUTE)
box(ax, 74.4, 1.6, 25.0, 9.4, INK, "#F3F5F8", ls=(0, (3, 2)))
ax.text(75.6, 10.0, "inference", fontsize=7, weight="bold", va="top")
ax.text(75.6, 7.2, "raw logits, threshold 0.5,\nno post-hoc offset,\n2 encoder passes", fontsize=6.3, va="top", color=MUTE)
fig.savefig(OUT / "arch.pdf"); fig.savefig(OUT / "arch.png", dpi=220); plt.close(fig)


# ---------------------------------------------------------------- Figure 2: routing of the joint query
def short(a):
    for old, new in [("UpperBody-Color-", "upper "), ("LowerBody-Color-", "lower "), ("UpperBody-Length-Short", "short sleeves"),
                     ("LowerBody-Length-Short", "short lower"), ("LowerBody-Type-Trousers&Shorts", "trousers/shorts"),
                     ("LowerBody-Type-Skirt&Dress", "skirt/dress"), ("Hair-Length-", "hair "), ("Accessory-Glasses-Normal", "glasses"),
                     ("Accessory-Glasses-Sun", "sunglasses"), ("Accessory-", ""), ("Gender-", ""), ("Age-", "age ")]:
        a = a.replace(old, new)
    return a.lower()


names = [("Accessory-Hat", "hat"), ("UpperBody-Color-White", "upper white"), ("LowerBody-Color-Black", "lower black")]
share = np.asarray(D["j_head_hshare"], float)          # (8 heads, 40 attributes): attention weight on H+ tokens
heads = np.asarray(D["att_j_heads"])                   # (40, 8, 228)
fig = plt.figure(figsize=(7.0, 3.85))
gl = fig.add_gridspec(3, 8, left=.005, right=.60, top=.925, bottom=.01, wspace=.06, hspace=.30)
for r, (a, lab) in enumerate(names):
    i = A.index(a); nh = int((share[:, i] > .5).sum())
    for h in range(8):
        ax = fig.add_subplot(gl[r, h]); ax.axis("off"); w = heads[i][h]
        if share[h, i] > .5:
            overlay(ax, w[1:99], 7, 14, HP, (0, 4, 0, 8), alpha=.9); c = HP
        else:
            overlay(ax, w[100:], 8, 16, CL, (0, 4, 0, 8), alpha=.9); c = CL
        ax.set_xlim(-.15, 4.15); ax.set_ylim(-.15, 8.15)
        ax.set_title(f"head {h + 1}" if h == 0 else f"{h + 1}", fontsize=6.5, color=c, pad=1.5)
        tag = "(a) " if r == 0 else ""
        if h == 0:
            ax.text(-.15, 9.55, f"{tag}“{lab}”: {nh} heads read H+ (blue), {8 - nh} read CLIP (orange)", fontsize=7, ha="left",
                    va="bottom", transform=ax.transData, clip_on=False)
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("route", [CL, "#FFFFFF", HP])
ax = fig.add_axes([.775, .06, .095, .83])
ax.imshow(share.T, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
ax.set_yticks(range(len(A))); ax.set_yticklabels([short(a) for a in A], fontsize=5.9)
for t, a in zip(ax.get_yticklabels(), A):
    if a in dict(names):
        t.set_fontweight("bold")
ax.set_xticks(range(8)); ax.set_xticklabels([str(h + 1) for h in range(8)], fontsize=6.3)
ax.xaxis.set_ticks_position("top"); ax.tick_params(length=0, pad=1.5)
ax.set_xticks(np.arange(-.5, 8), minor=True); ax.set_yticks(np.arange(-.5, len(A)), minor=True)
ax.grid(which="minor", color="white", lw=.4); ax.tick_params(which="minor", length=0)
for s_ in ax.spines.values():
    s_.set_visible(False)
nH = (share > .5).sum(0)
for i in range(len(A)):
    ax.text(8.0, i, f"{nH[i]}", fontsize=5.9, va="center", ha="left", color=HP)
ax.text(8.0, -1.6, "H+", fontsize=6.3, ha="left", va="center", color=HP)
fig.text(.655, .99, "(b) attention share on H+ tokens", fontsize=7, ha="left", va="top")
fig.text(.775 + .0475, .035, "attention head", fontsize=6.3, ha="center", va="top")
fig.savefig(OUT / "routing.pdf"); fig.savefig(OUT / "routing.png", dpi=220); plt.close(fig)

# ---------------------------------------------------------------- Figure 3: the joint head follows the better encoder
R = json.load(open(HERE / "ablations.json"))["joint_heads"]["per_attr"]
fig, ax = plt.subplots(figsize=(3.35, 2.25)); fig.subplots_adjust(left=.15, right=.98, top=.97, bottom=.17)
for run, col, mk in [("PETA_s1", "#2F6B4F", "o"), ("PETA_s2", "#2F6B4F", "o"), ("Market_s1", "#9C3D54", "s"), ("Market_s2", "#9C3D54", "s")]:
    h, c, j = (100 * np.asarray(R[x][run], float) for x in ("h", "c", "j"))
    ok = ~np.isnan(j)
    ax.scatter((h - c)[ok], (j - (h + c) / 2)[ok], s=7, c=col, marker=mk, lw=0, alpha=.75,
               label=("PETA held out" if run == "PETA_s1" else "Market held out" if run == "Market_s1" else None))
xs = np.linspace(-9, 9, 50)
ax.plot(xs, np.abs(xs) / 2, color=INK, lw=.8, ls="--"); ax.axhline(0, color=MUTE, lw=.6)
ax.text(-8.6, 4.75, "= the better\nmember head", fontsize=6.3, va="bottom", ha="left")
ax.text(8.6, -.35, "= average of\nthe members", fontsize=6.3, va="top", ha="right", color=MUTE)
ax.set_xlim(-9, 9); ax.set_ylim(-3.5, 7)
ax.set_xlabel("H+ head minus CLIP head (balanced acc., points)", fontsize=6.8, labelpad=1.5)
ax.set_ylabel("joint head minus\nmember average (points)", fontsize=6.8, labelpad=1.5)
ax.tick_params(labelsize=6.3, length=2, pad=1.5)
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
ax.legend(fontsize=6.3, frameon=False, loc="lower left", handletextpad=.2, borderaxespad=.1)
fig.savefig(OUT / "pick.pdf"); fig.savefig(OUT / "pick.png", dpi=220); plt.close(fig)
print("figures written")
