"""Figures of the v31 paper from real model outputs (analysis/v31_min.json, exported from the submitted du.pt on
PETA val image 11178 by report/v31_arch/export.py) and saved held-out logits (analysis/ablations.json).

../../../Upar-Challenge-Track-2/.venv/bin/python make_figs.py
figs/arch.pdf     the model on this image: encoders, PCA of the tokens, three query heads, hat logits, final predictions
figs/pick.pdf     per attribute: joint head vs the two member heads (balanced accuracy)
figs/routing.pdf  (a) where the H+ / CLIP attention heads of the joint query look, (b) encoder share of every head, 40 attributes
Everything is vector except the photographs, which are resampled once from the raw crop with Lanczos at 8x.
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
HP, CL, JT, INK, MUTE = "#1F5BD8", "#D2731A", "#7A3FC8", "#18212D", "#626E7D"
HP_BG, CL_BG, JT_BG = "#EDF2FD", "#FDF3E8", "#F4EEFC"
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 7.5, "pdf.fonttype": 42, "mathtext.fontset": "stixsans",
                     "text.color": INK, "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})


def decode(s):
    return Image.open(io.BytesIO(base64.b64decode(s.split(",")[-1]))).convert("RGB")


# the crop as the model sees it (224 x 112, squashed), resampled once from the raw 63 x 184 crop at 8x for print
PHOTO = np.asarray(decode(D["raw"]).resize((112 * 8, 224 * 8), Image.LANCZOS), float) / 255.0
H_IMG, W_IMG = PHOTO.shape[:2]


def heat(vals, rows, cols, color, gamma=.55, amax=.88):
    """Photo in light grey with an attention map (rows x cols tokens) upsampled bicubically and tinted in color."""
    g = PHOTO @ np.array([.299, .587, .114])
    base = np.repeat((.42 + .55 * g)[..., None], 3, -1)
    v = np.asarray(vals, float).reshape(rows, cols); v = v / max(v.max(), 1e-9)
    up = np.asarray(Image.fromarray(v.astype(np.float32), mode="F").resize((W_IMG, H_IMG), Image.BICUBIC)).clip(0, 1)
    a = (up ** gamma * amax)[..., None]
    return base * (1 - a) + np.array(matplotlib.colors.to_rgb(color)) * a


def rbox(ax, x, y, w, h, ec, fc, lw=.9, ls="-", r=6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", ec=ec, fc=fc, lw=lw, ls=ls, zorder=1))


def arrow(ax, x0, y0, x1, y1, color=MUTE, rad=0.0, lw=.8):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=5,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0, mutation_scale=7,
                                connectionstyle=f"arc3,rad={rad}"))


def tokens(ax, rgb, cols, rows, x0, y0, cell, ec):
    """PCA colours of the patch tokens as vector cells, row 0 at the top."""
    c = np.asarray(rgb, float).reshape(rows, cols, 3) / 255
    for r in range(rows):
        for k in range(cols):
            ax.add_patch(Rectangle((x0 + k * cell, y0 + (rows - 1 - r) * cell), cell, cell, fc=c[r, k], ec="white", lw=.15, zorder=2))
    ax.add_patch(Rectangle((x0, y0), cols * cell, rows * cell, fill=False, ec=ec, lw=.9, zorder=3))


def short(a):
    for old, new in [("UpperBody-Color-", "upper "), ("LowerBody-Color-", "lower "), ("UpperBody-Length-Short", "short sleeves"),
                     ("LowerBody-Length-Short", "short lower"), ("LowerBody-Type-Trousers&Shorts", "trousers"),
                     ("LowerBody-Type-Skirt&Dress", "skirt/dress"), ("Hair-Length-", "hair "), ("Accessory-Glasses-Normal", "glasses"),
                     ("Accessory-Glasses-Sun", "sunglasses"), ("Accessory-", ""), ("Gender-", ""), ("Age-", "")]:
        a = a.replace(old, new)
    return a.lower()


# ================================================================ Figure 1: the joint model on one image (units: 1/100 inch)
k = A.index("Accessory-Hat")
fig = plt.figure(figsize=(7.0, 3.0))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 700); ax.set_ylim(0, 300); ax.axis("off")
# input
ax.imshow(PHOTO, extent=(6, 81, 72, 222), zorder=2, interpolation="none")
ax.add_patch(Rectangle((6, 72), 75, 150, fill=False, ec=MUTE, lw=.6, zorder=3))
ax.text(43.5, 64, "input", ha="center", va="top", fontsize=8)
ax.text(43.5, 52, "224 × 112", ha="center", va="top", fontsize=7.5, color=MUTE)
arrow(ax, 83, 190, 102, 234, rad=-.3); arrow(ax, 83, 104, 102, 66, rad=.3)
# encoders: a strip of transformer blocks, the first 8 frozen (hatched)
for (y0, ec, fc, name, size, n) in [(196, HP, HP_BG, "DINOv3 ViT-H+/16", "840M", 32), (28, CL, CL_BG, "CLIP ViT-L/14", "303M", 24)]:
    rbox(ax, 102, y0, 150, 76, ec, fc)
    ax.text(110, y0 + 68, name, fontsize=8.5, weight="bold", color=ec, va="top")
    ax.text(244, y0 + 68, size, fontsize=7.5, color=MUTE, va="top", ha="right")
    sw = 134 / n
    for i in range(n):
        frozen = i < 8
        ax.add_patch(Rectangle((110 + i * sw, y0 + 22), sw * .72, 26, fc="#E2E6EC" if frozen else ec, ec="#9AA4B1" if frozen else ec,
                               lw=.3, hatch="//////" if frozen else None, zorder=2))
    ax.text(110, y0 + 9, f"8 frozen · {n - 8} fine-tuned blocks", fontsize=7.5, color=MUTE, va="center")
# tokens
tokens(ax, D["pca_h"], 7, 14, 268, 195, 5.6, HP); tokens(ax, D["pca_c"], 8, 16, 268, 26, 5.0, CL)
ax.text(287.6, 279, "tokens", ha="center", va="bottom", fontsize=7.5)
arrow(ax, 252, 234, 266, 234); arrow(ax, 252, 66, 266, 66)
# W: projection of both token sets to a shared width + source embedding
ax.add_patch(plt.Circle((337, 150), 13, fc="white", ec=JT, lw=1.0, zorder=4))
ax.text(337, 150, "W", ha="center", va="center", fontsize=9, weight="bold", color=JT, zorder=5)
arrow(ax, 309, 200, 332, 163, color=HP, rad=-.15); arrow(ax, 309, 100, 332, 137, color=CL, rad=.15)
ax.text(287, 150, "to 1024-d\n+ source\nembedding", ha="center", va="center", fontsize=7, color=JT, linespacing=1.2)
# three query heads, each with its "hat" logit
heads = [(196, HP, HP_BG, "H+ query head", "reads 99 H+ tokens", "h"),
         (112, JT, JT_BG, "joint query head", "reads all 228 tokens", "j"),
         (28, CL, CL_BG, "CLIP query head", "129 tokens + text score", "c")]
for (y0, ec, fc, name, sub, key) in heads:
    rbox(ax, 366, y0, 138, 76, ec, fc)
    ax.text(374, y0 + 68, name, fontsize=8.5, weight="bold", color=ec, va="top")
    ax.text(374, y0 + 50, sub, fontsize=7.5, va="top")
    for q in range(7):                                   # attribute queries (40 in the model)
        ax.add_patch(plt.Circle((379 + q * 9.5, y0 + 15), 3.6, fc="white", ec=ec, lw=.7, zorder=3))
    ax.text(447, y0 + 15, "40 queries", fontsize=7, color=MUTE, va="center")
    z = D["logits"][key][k]
    ax.add_patch(plt.Circle((523, y0 + 38), 12.5, fc="white", ec=ec, lw=1.1, zorder=4))
    ax.text(523, y0 + 38, f"{z:.1f}", ha="center", va="center", fontsize=8, zorder=5)
    arrow(ax, 504, y0 + 38, 510, y0 + 38, color=ec)
    arrow(ax, 535.5, y0 + 38, 546, 150 + (y0 + 38 - 150) * .12, color=MUTE)
arrow(ax, 309, 234, 364, 234, color=HP); arrow(ax, 350, 150, 364, 150, color=JT); arrow(ax, 309, 66, 364, 66, color=CL)
ax.text(523, 280, "“hat” logit", ha="center", va="bottom", fontsize=7.5, color=MUTE)
ax.add_patch(plt.Circle((557, 150), 11, fc=INK, ec=INK, zorder=4))
ax.text(557, 150, "avg", ha="center", va="center", fontsize=7.5, color="white", zorder=5)
arrow(ax, 568, 150, 578, 150, color=MUTE)
# predictions for this image: sigmoid of the averaged logit, threshold 0.5 for every attribute
p = np.asarray(D["prob"]); order = list(np.argsort(-p)[:7]) + [A.index("LowerBody-Color-Purple"), A.index("Accessory-Glasses-Sun")]
x0, x1, ytop, dy = 644, 694, 255, 21
ax.text(694, 280, "prediction σ(avg)", ha="right", va="bottom", fontsize=7.5, color=MUTE)
for r, i in enumerate(order):
    y = ytop - r * dy
    ax.text(x0 - 4, y, short(A[i]), ha="right", va="center", fontsize=7.5, weight="bold" if i == k else "normal")
    ax.add_patch(Rectangle((x0, y - 6), x1 - x0, 12, fc="#EEF0F3", ec="none", zorder=1))
    ax.add_patch(Rectangle((x0, y - 6), (x1 - x0) * p[i], 12, fc=INK if p[i] > .5 else "#AEB6C1", ec="none", zorder=2))
yb = ytop - (len(order) - 1) * dy
ax.plot([x0 + (x1 - x0) * .5] * 2, [yb - 9, ytop + 9], color="#C2304A", lw=.8, ls=(0, (2, 1.5)), zorder=3)
ax.text(x0 + (x1 - x0) * .5, yb - 11, "0.5", ha="center", va="top", fontsize=7, color="#C2304A")
ax.text(374, 11, r"training: $L=\sum_{k\in\{H,J,C\}}\,\ell_{\mathrm{LA}}(z_k,y)$", fontsize=7.5, va="center", color=INK)
fig.savefig(OUT / "arch.pdf"); fig.savefig(OUT / "arch.png", dpi=300); plt.close(fig)

# ================================================================ Figure 2: the joint head follows the better member head
R = json.load(open(HERE / "ablations.json"))["joint_heads"]["per_attr"]
fig, ax = plt.subplots(figsize=(3.35, 2.45)); fig.subplots_adjust(left=.155, right=.975, top=.975, bottom=.165)
xs = np.linspace(-9, 9, 200)
ax.fill_between(xs, 0, np.abs(xs) / 2, color="#E9EDF2", lw=0, zorder=0)
ax.plot(xs, np.abs(xs) / 2, color=INK, lw=.9, ls=(0, (4, 2)), zorder=1); ax.axhline(0, color=MUTE, lw=.7, zorder=1)
for run, col, mk, lab in [("PETA_s1", "#2E7D5B", "o", "PETA held out"), ("PETA_s2", "#2E7D5B", "o", None),
                          ("Market_s1", "#B2405E", "s", "Market held out"), ("Market_s2", "#B2405E", "s", None)]:
    h, c, j = (100 * np.asarray(R[x][run], float) for x in ("h", "c", "j"))
    ok = ~np.isnan(j)
    ax.scatter((h - c)[ok], (j - (h + c) / 2)[ok], s=11, c=col, marker=mk, lw=.3, ec="white", alpha=.9, label=lab, zorder=3)
ax.text(-8.7, 4.6, "better member", fontsize=7.5, ha="left", va="bottom", rotation=-24)
ax.text(-8.7, -.3, "member average", fontsize=7.5, ha="left", va="top", color=MUTE)
ax.set_xlim(-9, 9); ax.set_ylim(-3.2, 6.8)
ax.set_xlabel("H+ head − CLIP head (balanced acc., points)", fontsize=7.5, labelpad=2)
ax.set_ylabel("joint head − member average", fontsize=7.5, labelpad=2)
ax.tick_params(labelsize=7, length=2.5, pad=2)
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
ax.spines["left"].set_linewidth(.7); ax.spines["bottom"].set_linewidth(.7)
ax.legend(fontsize=7, frameon=False, loc="upper center", handletextpad=.1, borderaxespad=.2, ncol=2, columnspacing=.8)
fig.savefig(OUT / "pick.pdf"); fig.savefig(OUT / "pick.png", dpi=300); plt.close(fig)

# ================================================================ Figure 3: routing inside the joint head
share = np.asarray(D["j_head_hshare"], float)          # (8 heads, 40 attributes): attention weight on H+ tokens
heads_att = np.asarray(D["att_j_heads"])               # (40, 8, 228): H+ CLS, 98 H+ patches, CLIP CLS, 128 CLIP patches
names = [("Accessory-Hat", "hat"), ("UpperBody-Color-White", "upper white"), ("LowerBody-Color-Black", "lower black")]
fig = plt.figure(figsize=(7.0, 3.85))
iw, ih, gap, ggap, top = .092, .36, .012, .055, .87      # image size in figure fractions (7.0 x 3.55 in)
x = (1 - (6 * iw + 3 * gap + 2 * ggap)) / 2
fig.text(.01, .985, "(a)", fontsize=8.5, weight="bold", va="top")
for (a, lab) in names:
    i = A.index(a); hsel = share[:, i] > .5
    wh = (heads_att[i][hsel][:, 1:99]).sum(0); wc = (heads_att[i][~hsel][:, 100:]).sum(0)
    for col, (img, cap, c) in enumerate([(heat(wh, 14, 7, HP), f"{hsel.sum()} H+ heads", HP), (heat(wc, 16, 8, CL), f"{(~hsel).sum()} CLIP heads", CL)]):
        axi = fig.add_axes([x + col * (iw + gap), top - ih, iw, ih]); axi.imshow(img, interpolation="none"); axi.set_xticks([]); axi.set_yticks([])
        for s_ in axi.spines.values():
            s_.set_edgecolor(c); s_.set_linewidth(1.0)
        axi.set_title(cap, fontsize=7.5, color=c, pad=2)
    fig.text(x + iw + gap / 2, top + .048, f"“{lab}”", fontsize=8.5, ha="center", va="bottom", weight="bold")
    x += 2 * iw + gap + ggap + gap
# (b) heatmap: 8 attention heads x 40 attributes, colour = share of attention on H+ tokens
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("route", [CL, "#FFFFFF", HP])
axh = fig.add_axes([.075, .16, .905, .225])
axh.imshow(share, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="none")
axh.set_yticks(range(8)); axh.set_yticklabels([f"head {h + 1}" for h in range(8)], fontsize=6.8)
axh.set_xticks(range(len(A))); axh.set_xticklabels([short(a) for a in A], fontsize=6.8, rotation=60, ha="right", rotation_mode="anchor")
for t, a in zip(axh.get_xticklabels(), A):
    if a in dict(names):
        t.set_fontweight("bold")
axh.set_xticks(np.arange(-.5, len(A)), minor=True); axh.set_yticks(np.arange(-.5, 8), minor=True)
axh.grid(which="minor", color="white", lw=.6); axh.tick_params(which="both", length=0, pad=2)
for s_ in axh.spines.values():
    s_.set_visible(False)
for b in (6.5, 19.5, 32.5, 34.5):                      # groups: age/gender/hair | upper | lower | type | accessories
    axh.axvline(b, color=INK, lw=.9)
nH = (share > .5).sum(0)
for i in range(len(A)):
    axh.text(i, -.85, str(nH[i]), fontsize=6.8, ha="center", va="bottom", color=HP)
axh.text(-.7, -.85, "H+ heads", fontsize=6.8, ha="right", va="bottom", color=HP)
fig.text(.01, .468, "(b)", fontsize=8.5, weight="bold", va="top")
fig.savefig(OUT / "routing.pdf"); fig.savefig(OUT / "routing.png", dpi=300); plt.close(fig)
print("figures written")
