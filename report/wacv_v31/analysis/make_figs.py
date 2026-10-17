"""Figures of the v31 paper from real model outputs (analysis/v31_min.json, exported from the submitted du.pt on
PETA val image 11178 by report/v31_arch/export.py) and saved held-out logits (analysis/ablations.json).

../../../Upar-Challenge-Track-2/.venv/bin/python make_figs.py
figs/arch.pdf     the model on this image: encoders, PCA of the tokens, three query heads, hat logits, final predictions
figs/pick.pdf     per attribute: gain of the joint query head over the mean of the two single-encoder query heads,
                  against their gap (all 148 points shown; bin means with 95% bootstrap intervals over attribute-fold units)
figs/routing.pdf  (a) where the DINOv3 / CLIP attention heads of the joint query head look, drawn as token cells (no smoothing),
                  (b) encoder share of every attention head, 40 attributes
Everything is vector except the photographs, which are resampled once from the raw crop with Lanczos at 8x.
Colors: DINOv3 blue / CLIP orange (distinct for common color-vision deficiencies), datasets by Okabe-Ito green / reddish
purple plus marker shape (scientific-visualization skill, K-Dense, MIT).
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


GREY = np.repeat((.42 + .55 * (PHOTO @ np.array([.299, .587, .114])))[..., None], 3, -1)


def heat(axi, vals, rows, cols, color, gamma=.6, amax=.85):
    """Photo in light grey, one tinted cell per token (no smoothing: the map has only rows x cols values)."""
    axi.imshow(GREY, extent=(0, cols, rows, 0), interpolation="none")
    v = np.asarray(vals, float).reshape(rows, cols); v = v / max(v.max(), 1e-9)
    for r in range(rows):
        for k in range(cols):
            if v[r, k] > .02:
                axi.add_patch(Rectangle((k, r), 1, 1, fc=color, ec="none", alpha=float(v[r, k] ** gamma * amax), zorder=2))
    for r in range(1, rows):
        axi.axhline(r, color="white", lw=.2, alpha=.35, zorder=3)
    for k in range(1, cols):
        axi.axvline(k, color="white", lw=.2, alpha=.35, zorder=3)
    axi.set_xlim(0, cols); axi.set_ylim(rows, 0)


def rbox(ax, x, y, w, h, ec, fc, lw=.9, ls="-", r=6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", ec=ec, fc=fc, lw=lw, ls=ls, zorder=1))


def arrow(ax, x0, y0, x1, y1, color=MUTE, rad=0.0, lw=.8):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=5,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0, mutation_scale=7,
                                connectionstyle=f"arc3,rad={rad}"))


def tokens(ax, rgb, cols, rows, x0, y0, w, h, ec):
    """PCA colors of the patch tokens as seamless vector cells (one cell = one token), row 0 at the top."""
    c = np.asarray(rgb, float).reshape(rows, cols, 3) / 255
    cw, ch = w / cols, h / rows
    for r in range(rows):
        for k in range(cols):
            ax.add_patch(Rectangle((x0 + k * cw, y0 + (rows - 1 - r) * ch), cw, ch, fc=c[r, k], ec=c[r, k], lw=.25, zorder=2))
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, ec=ec, lw=.9, zorder=3))


def short(a):
    for old, new in [("UpperBody-Color-", "upper "), ("LowerBody-Color-", "lower "), ("UpperBody-Length-Short", "short sleeves"),
                     ("LowerBody-Length-Short", "short lower"), ("LowerBody-Type-Trousers&Shorts", "trousers"),
                     ("LowerBody-Type-Skirt&Dress", "skirt/dress"), ("Hair-Length-", "hair "), ("Accessory-Glasses-Normal", "glasses"),
                     ("Accessory-Glasses-Sun", "sunglasses"), ("Accessory-", ""), ("Gender-", ""), ("Age-", "")]:
        a = a.replace(old, new)
    return a.lower()


# ================================================================ Figure 1: (a) the joint model, (b) one query head
# Block style of the DeepSeek-V3/V4 figures (user choice, v25 deck): yellow = module, grey = operation, white + lock =
# frozen, token stacks = rows of circles, residual adds, dashed = training only. Real parts: the input crop, the "hat"
# logits of the three query heads and the final probability of the submitted weights. Units: 1/100 inch.
YEL, GRY, LINE = "#F8E6BC", "#E1E6EB", "#2B3A4E"
QCOLS = ["#F0A868", "#8FC1E3", "#C9B6E4", "#9FD3B5", "#F2C4CE"]      # five of the 40 attribute queries


def blk(ax, x0, y0, x1, y1, text, sub=None, kind="main", color=INK, bold=False, fs=7.2):
    fc = {"main": YEL, "op": GRY, "frozen": "white"}[kind]
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=4", fc=fc, ec=LINE,
                                lw=.75, zorder=2))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if sub:
        ax.text(cx, cy + 4.6, text, ha="center", va="center", fontsize=fs, color=color, weight="bold" if bold else "normal", zorder=3)
        ax.text(cx, cy - 5.4, sub, ha="center", va="center", fontsize=6.5, color=MUTE, zorder=3)
    else:
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=color, weight="bold" if bold else "normal", zorder=3)


def stack(ax, cx, cy, fills, layers=3):
    """Token stack: a row of circles in a rounded frame, two frames behind it (many tokens / vectors)."""
    n = len(fills); w, h = n * 8.4 + 4, 10
    for L in range(layers - 1, -1, -1):
        x0, y0 = cx - w / 2 - 1.8 * L, cy - h / 2 + 1.8 * L
        ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=3", fc="white", ec=LINE, lw=.6, zorder=2))
        for i in range(n):
            ax.add_patch(plt.Circle((x0 + 6.2 + 8.4 * i, y0 + h / 2), 2.9, fc=fills[i] if L == 0 and fills[i] else "white",
                                    ec=LINE, lw=.55, zorder=3))
    return cx - w / 2 - 3.6, cy - h / 2, cx + w / 2, cy + h / 2 + 3.6      # visual box


def path(ax, pts, dashed=False, color=LINE, head=True, lw=.8):
    xs, ys = zip(*pts)
    ls = (0, (3, 2)) if dashed else "-"
    if len(pts) > 2:
        ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, ls=ls, solid_capstyle="butt", zorder=1)
    ax.annotate("", xy=pts[-1], xytext=pts[-2], zorder=1,
                arrowprops=dict(arrowstyle="-|>" if head else "-", color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0,
                                mutation_scale=6))


def plus(ax, cx, cy, r=5):
    ax.add_patch(plt.Circle((cx, cy), r, fc="white", ec=LINE, lw=.8, zorder=3))
    ax.plot([cx - r, cx + r], [cy, cy], color=LINE, lw=.8, zorder=4); ax.plot([cx, cx], [cy - r, cy + r], color=LINE, lw=.8, zorder=4)


def lock(ax, cx, cy, s=1.0):
    ax.add_patch(Rectangle((cx - 3.2 * s, cy - 2.6 * s), 6.4 * s, 5.0 * s, fc=MUTE, ec="none", zorder=4))
    ax.add_patch(matplotlib.patches.Arc((cx, cy + 2.4 * s), 4.2 * s, 6.0 * s, theta1=0, theta2=180, color=MUTE, lw=1.0, zorder=4))


k = A.index("Accessory-Hat")
fig = plt.figure(figsize=(7.0, 3.1))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 700); ax.set_ylim(0, 310); ax.axis("off")
SEP = 446
ax.text(3, 304, "(a) joint model", fontsize=8.5, weight="bold", va="top")
ax.text(SEP + 6, 304, "(b) one query head", fontsize=8.5, weight="bold", va="top")
ax.plot([SEP, SEP], [8, 300], color="#D5DAE0", lw=.6)
lx, ly = 118, 297                                           # legend
for i, (kind, lab) in enumerate([("main", "module"), ("op", "operation"), ("frozen", "frozen"), ("dash", "training only")]):
    x = lx + i * 70
    if kind == "dash":
        path(ax, [(x, ly), (x + 15, ly)], dashed=True)
    else:
        ax.add_patch(FancyBboxPatch((x, ly - 4.5), 15, 9, boxstyle="round,pad=0,rounding_size=2",
                                    fc={"main": YEL, "op": GRY, "frozen": "white"}[kind], ec=LINE, lw=.6, zorder=2))
        if kind == "frozen":
            lock(ax, x + 7.5, ly - .3, .7)
    ax.text(x + 19, ly, lab, fontsize=6.8, va="center", color=MUTE)

# ---------------- (a) joint model
ax.imshow(PHOTO, extent=(6, 44, 114, 190), zorder=2, interpolation="none")
ax.add_patch(Rectangle((6, 114), 38, 76, fill=False, ec=LINE, lw=.6, zorder=3))
ax.text(25, 108, "input", ha="center", va="top", fontsize=7.2)
ax.text(25, 99, "224 × 112", ha="center", va="top", fontsize=6.5, color=MUTE)
for (y0, y1, rad) in ((178, 241, -.35), (126, 59, .35)):
    ax.annotate("", xy=(56, y1), xytext=(44, y0), zorder=1, arrowprops=dict(arrowstyle="-|>", color=LINE, lw=.8,
                mutation_scale=6, shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}"))
XS = 218                                                    # token stacks and W
for (cy, name, tuned, col, dim, ntok) in [(241, "DINOv3 ViT-H+/16", "24 blocks", HP, 1280, 99), (59, "CLIP ViT-L/14", "16 blocks", CL, 1024, 129)]:
    blk(ax, 56, cy - 15, 96, cy + 15, "", kind="frozen")
    lock(ax, 76, cy + 5)
    ax.text(76, cy - 7, "8 blocks", ha="center", va="center", fontsize=6.3, color=MUTE, zorder=3)
    blk(ax, 96, cy - 15, 188, cy + 15, name, tuned, color=col, bold=True, fs=7)
    st = stack(ax, XS, cy, [col, None, None, None])
    path(ax, [(188, cy), (st[0] - .5, cy)])
    ax.text(XS, cy + 15 if cy > 150 else cy - 12, f"{ntok} × {dim}", ha="center", va="center", fontsize=6.3, color=MUTE)
blk(ax, XS - 28, 139, XS + 28, 163, "W + source", "embedding", kind="op", fs=6.6)
path(ax, [(XS, 236), (XS, 163)], color=HP); path(ax, [(XS, 67.6), (XS, 139)], color=CL)
QX0, QX1, LX = 266, 358, 377
QH = [(241, "DINOv3 query head", "99 tokens", HP, "h"), (151, "joint query head", "228 tokens", JT, "j"),
      (59, "CLIP query head", "+ text score", CL, "c")]
for (cy, name, sub, col, key) in QH:
    blk(ax, QX0, cy - 13, QX1, cy + 13, name, sub, color=col, bold=True, fs=7)
    ax.add_patch(plt.Circle((LX, cy), 9.5, fc="white", ec=col, lw=1.0, zorder=3))
    ax.text(LX, cy, f"{D['logits'][key][k]:.1f}", ha="center", va="center", fontsize=7, zorder=4)
    path(ax, [(QX1, cy), (LX - 9.5, cy)], color=col)
for cy, col in ((241, HP), (59, CL)):
    path(ax, [(XS + 18.8, cy), (QX0, cy)], color=col)
path(ax, [(XS + 28, 151), (QX0, 151)], color=JT)
ax.text(LX, 256, "“hat” logit", ha="center", va="bottom", fontsize=6.8, color=MUTE)
AX_ = 408                                                   # average of the three logits, then the probability
ax.add_patch(plt.Circle((AX_, 151), 8.5, fc=GRY, ec=LINE, lw=.75, zorder=3))
ax.text(AX_, 151, "avg", ha="center", va="center", fontsize=6.5, zorder=4)
path(ax, [(LX + 9.5, 241), (AX_, 241), (AX_, 159.5)]); path(ax, [(LX + 9.5, 151), (AX_ - 8.5, 151)])
path(ax, [(LX + 9.5, 59), (AX_, 59), (AX_, 142.5)])
path(ax, [(AX_ + 8.5, 151), (AX_ + 18, 151)])
ax.text(AX_ + 21, 151, "σ", ha="left", va="center", fontsize=7.2)
ax.text(AX_ + 2, 132, "p = %.3f" % D["prob"][k], ha="left", va="top", fontsize=6.6)
path(ax, [(LX, 49.5), (LX, 18), (334, 18)], dashed=True)
ax.text(330, 18, r"$L=\sum_{k\in\{D,J,C\}}\mathrm{BCE}(z_k+s,\,y)$", ha="right", va="center", fontsize=7.2)

# ---------------- (b) inside one query head (same for all three; the joint one reads the 228 mapped tokens)
xq, xm, xs, xt = 484, 540, 612, 674
q0 = stack(ax, xq, 34, QCOLS)
ax.text(xq, 20, "40 attribute queries", ha="center", va="center", fontsize=6.8, color=MUTE)
t0 = stack(ax, xt, 34, ["#8A96A6", None, None, None])
ax.text(697, 20, "CLS + patch tokens", ha="right", va="center", fontsize=6.8, color=MUTE)
ax.text(697, 11, "(joint: 228, both encoders)", ha="right", va="center", fontsize=6.3, color=MUTE)
blk(ax, xm - 23, 26, xm + 23, 42, "LayerNorm", kind="op", fs=6.5)
path(ax, [(q0[2] + 1, 34), (xm - 23, 34)])
path(ax, [(xm + 23, 34), (xs, 34), (xs, 80)])
ax.text(xs + 4, 58, "Q", fontsize=7.2, weight="bold", va="center")
blk(ax, xt - 23, 52, xt + 23, 68, "LayerNorm", kind="op", fs=6.5)
path(ax, [(xt, t0[3]), (xt, 52)])
path(ax, [(xt, 68), (xt, 93), (xs + 40, 93)])
ax.text(xs + 44, 97, "K, V", fontsize=7.2, weight="bold", va="bottom")
blk(ax, xs - 40, 80, xs + 40, 106, "cross-attention", "8 attention heads")
plus(ax, xq, 120)
path(ax, [(xq, q0[3]), (xq, 115)])
path(ax, [(xs, 106), (xs, 120), (xq + 5, 120)])
path(ax, [(xq, 125), (xq, 165)]); path(ax, [(xq, 141), (xm - 23, 141)])
blk(ax, xm - 23, 133, xm + 23, 149, "LayerNorm", kind="op", fs=6.5)
path(ax, [(xm + 23, 141), (xs - 40, 141)])
blk(ax, xs - 40, 131, xs + 40, 151, "MLP", "hidden width ×2")
plus(ax, xq, 170)
path(ax, [(xs, 151), (xs, 170), (xq + 5, 170)])
path(ax, [(xq, 175), (xq, 196), (xm - 23, 196)])
blk(ax, xm - 23, 188, xm + 23, 204, "LayerNorm", kind="op", fs=6.5)
path(ax, [(xm + 23, 196), (xs - 40, 196)])
blk(ax, xs - 40, 184, xs + 40, 208, "40 classifiers", "one per attribute")
z0 = stack(ax, xs, 232, ["#8FC1E3"] * 5)
path(ax, [(xs, 208), (xs, 226.5)])
ax.text(z0[0] - 3, 232, "40 logits", ha="right", va="center", fontsize=6.8, color=MUTE)
path(ax, [(z0[2] + 2, 232), (664, 232)], dashed=True)
ax.text(667, 232, "loss", ha="left", va="center", fontsize=6.8, style="italic")
ax.text(697, 278, "one layer · width 1024 (DINOv3: 1280)", ha="right", va="center", fontsize=6.5, color=MUTE)
fig.savefig(OUT / "arch.pdf"); fig.savefig(OUT / "arch.png", dpi=300); plt.close(fig)

# ================================================================ Figure 2: the joint query head moves toward the better one
# x = gap between the two single-encoder query heads, y = gain of the joint query head over their mean (balanced accuracy,
# points). y = 0: plain mean; y = x / 2: always the better of the two. All 148 points are drawn (none clipped).
R = json.load(open(HERE / "ablations.json"))["joint_heads"]["per_attr"]
PETA_C, MKT_C = "#009E73", "#CC79A7"                    # Okabe-Ito bluish green / reddish purple, plus marker shape
gap, gain, unit = [], [], []
fig, ax = plt.subplots(figsize=(3.35, 2.5)); fig.subplots_adjust(left=.14, right=.975, top=.975, bottom=.175)
for run, col, mk, lab in [("PETA_s1", PETA_C, "o", "PETA held out"), ("PETA_s2", PETA_C, "o", None),
                          ("Market_s1", MKT_C, "s", "Market held out"), ("Market_s2", MKT_C, "s", None)]:
    h, c, j = (100 * np.asarray(R[x][run], float) for x in ("h", "c", "j"))
    ok = ~np.isnan(j)
    g, y = np.abs(h - c)[ok], (j - (h + c) / 2)[ok]
    gap += list(g); gain += list(y); unit += [f"{run[:-3]}/{a}" for a in np.where(ok)[0]]   # unit = attribute in one fold
    ax.scatter(g, y, s=12, marker=mk, fc="none" if mk == "s" else col, ec=col, lw=.8 if mk == "s" else .3, alpha=.85,
               label=lab, zorder=3)
gap, gain, unit = np.array(gap), np.array(gain), np.array(unit)
xmax = np.ceil(gap.max()) + .5
xs = np.array([0, xmax])
ax.fill_between(xs, 0, xs / 2, color="#ECEFF3", lw=0, zorder=0)
ax.plot(xs, xs / 2, color=INK, lw=.9, ls=(0, (4, 2)), zorder=1); ax.axhline(0, color=MUTE, lw=.7, zorder=1)
# bin means with 95% bootstrap intervals; resampling whole attribute-fold units keeps the two seeds together
rng = np.random.default_rng(20261017); units = np.unique(unit)
edges = [0, 1, 2, 4, xmax]
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (gap >= lo) & (gap < hi)
    us = np.unique(unit[m]); bs = []
    for _ in range(4000):
        pick = np.concatenate([np.where(m & (unit == u))[0] for u in rng.choice(us, len(us))])
        bs.append(gain[pick].mean())
    lo_ci, hi_ci = np.percentile(bs, [2.5, 97.5]); xm = gap[m].mean()
    ax.errorbar(xm, gain[m].mean(), yerr=[[gain[m].mean() - lo_ci], [hi_ci - gain[m].mean()]], fmt="D", ms=4.2, color=INK,
                mfc="white", mew=1.0, elinewidth=1.0, capsize=2, zorder=5, label="bin mean, 95% CI" if lo == 0 else None)
ax.text(11.2, 11.2 / 2 - .3, "better of the two", fontsize=7.2, ha="right", va="top", rotation=np.degrees(np.arctan(.5)),
        transform_rotates_text=True, rotation_mode="anchor")
ax.text(xmax - .2, -.25, "mean of the two", fontsize=7.2, ha="right", va="top", color=MUTE)
ax.set_xlim(0, xmax); ax.set_ylim(-2.6, 9.6)
ax.set_xlabel("gap between DINOv3 and CLIP query heads (points)", fontsize=7.2, labelpad=2)
ax.set_ylabel("joint query head − mean of the two", fontsize=7.2, labelpad=2)
ax.tick_params(labelsize=7, length=2.5, pad=2)
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
ax.spines["left"].set_linewidth(.7); ax.spines["bottom"].set_linewidth(.7)
ax.legend(fontsize=7, frameon=False, loc="upper left", handletextpad=.2, borderaxespad=.2, labelspacing=.3)
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
    for col, (vals, rows, cols, cap, c) in enumerate([(wh, 14, 7, f"DINOv3 tokens\n{hsel.sum()} attn. heads", HP), (wc, 16, 8, f"CLIP tokens\n{(~hsel).sum()} attn. heads", CL)]):
        axi = fig.add_axes([x + col * (iw + gap), top - ih, iw, ih]); heat(axi, vals, rows, cols, c); axi.set_xticks([]); axi.set_yticks([])
        for s_ in axi.spines.values():
            s_.set_edgecolor(c); s_.set_linewidth(1.0)
        axi.set_title(cap, fontsize=7.2, color=c, pad=2, linespacing=1.15)
    fig.text(x + iw + gap / 2, top + .075, f"“{lab}”", fontsize=8.5, ha="center", va="bottom", weight="bold")
    x += 2 * iw + gap + ggap + gap
# (b) heatmap: 8 attention heads x 40 attributes, colour = share of attention on H+ tokens
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("route", [CL, "#FFFFFF", HP])
axh = fig.add_axes([.1, .16, .88, .225])
axh.imshow(share, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="none")
axh.set_yticks(range(8)); axh.set_yticklabels([str(h + 1) for h in range(8)], fontsize=6.8)
axh.set_ylabel("attention head", fontsize=6.8, labelpad=3)
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
axh.text(-.7, -.85, "# on DINOv3", fontsize=6.8, ha="right", va="bottom", color=HP)
fig.text(.01, .468, "(b)", fontsize=8.5, weight="bold", va="top")
cax = fig.add_axes([.84, .452, .13, .018])
cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, 1), cmap=cmap), cax=cax, orientation="horizontal")
cb.set_ticks([0, .5, 1]); cb.set_ticklabels(["0", "0.5", "1"]); cax.xaxis.set_ticks_position("top")
cb.ax.tick_params(labelsize=6.5, length=2, pad=1); cb.outline.set_linewidth(.5)
fig.text(.832, .461, "share of attention on DINOv3 tokens", fontsize=6.8, ha="right", va="center")
fig.savefig(OUT / "routing.pdf"); fig.savefig(OUT / "routing.png", dpi=300); plt.close(fig)
print("figures written")
