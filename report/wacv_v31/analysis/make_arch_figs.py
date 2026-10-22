"""Architecture figures of UniQuery from real model outputs (analysis/v31_min.json, exported from the submitted du.pt on
PETA val image 11178 by report/v31_arch/export.py). Light matplotlib only, no model is run.

../../../Upar-Challenge-Track-2/.venv/bin/python make_arch_figs.py
figs/uniquery_model.pdf  Figure 1, text width (6.875 in): input crop, the two encoders (frozen / fine-tuned parts, widths
                         proportional to the number of blocks), token PCA grids, one linear layer per encoder, concat,
                         three query heads, "hat" logits, mean, sigmoid, decision, training-only loss (dashed)
figs/query_head.pdf      Figure 2, column width (3.28 in): inside one query head (pre-norm cross-attention + MLP layer,
                         per-attribute classifiers), with the real "hat" attention of the DINOv3 query head
Visual grammar: hue = branch (DINOv3 blue, CLIP orange, joint purple; always with a text label), light tint = trained,
white + hatching + lock = frozen, grey = parameter-free operation, dashed = training only. Strokes 0.7 pt, corner radius 3.
Font: Times New Roman (regular, italic); bold text uses Liberation Serif Bold (no Times New Roman Bold installed);
math in STIX. Units of both canvases: 1/100 inch, drawn at final width, so font sizes are printed sizes.
A collision check (text vs. text, text vs. drawn lines) runs before saving and prints any overlap.
"""
import base64
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.patches import Arc, Circle, FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE.parent / "figs"
D = json.load(open(HERE / "v31_min.json"))
A = D["attrs"]
K = A.index("Accessory-Hat")

FONTS = Path.home() / ".local/share/fonts"
for f in ("TimesNewRoman-Regular.ttf", "TimesNewRoman-Italic.ttf"):
    fm.fontManager.addfont(str(FONTS / f))
BOLD_TTF = "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"   # metric-compatible with Times New Roman Bold
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 7.5, "pdf.fonttype": 42, "mathtext.fontset": "stix",
                     "text.color": "#18212D", "hatch.linewidth": .4})

HP, CL, JT = "#1F5BD8", "#D2731A", "#7A3FC8"            # DINOv3 blue / CLIP orange / joint purple
HP_T, CL_T, JT_T = "#E3EBFB", "#FBEBDA", "#EEE6F9"      # light tints = trained parts of each branch
INK, MUTE, LINE = "#18212D", "#5B6675", "#2B3A4E"
GRY, HATCH = "#E6E9ED", "#AEB6C0"
LW = .7                                                 # all strokes (pt)
RAD = 3                                                 # all corner radii (1/100 in)
FS_MIN = 6.8


def decode(s):
    return Image.open(io.BytesIO(base64.b64decode(s.split(",")[-1]))).convert("RGB")


# the crop as the model sees it (224 x 112, squashed), resampled once from the raw 63 x 184 crop at 8x for print
PHOTO = np.asarray(decode(D["raw"]).resize((112 * 8, 224 * 8), Image.LANCZOS), float) / 255.0
GREY = np.repeat((.45 + .5 * (PHOTO @ np.array([.299, .587, .114])))[..., None], 3, -1)


class Canvas:
    """Axes in 1/100 inch plus a record of every drawn segment, for the collision check."""

    def __init__(self, w_in, h_in):
        self.fig = plt.figure(figsize=(w_in, h_in))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, w_in * 100); self.ax.set_ylim(0, h_in * 100); self.ax.axis("off")
        self.segs, self.texts, self.inside = [], [], []

    def text(self, x, y, s, fs=7.5, check=True, bold=False, **kw):
        assert fs >= FS_MIN, s
        if bold:
            kw["fontproperties"] = fm.FontProperties(fname=BOLD_TTF, size=fs)
        else:
            kw["fontsize"] = fs
        t = self.ax.text(x, y, s, zorder=kw.pop("zorder", 6), **kw)
        if check:
            self.texts.append(t)
        return t

    def path(self, pts, color=LINE, dashed=False, head=True):
        ls = (0, (3, 2)) if dashed else "-"
        xs, ys = zip(*pts)
        if len(pts) > 2:
            self.ax.plot(xs[:-1], ys[:-1], color=color, lw=LW, ls=ls, solid_capstyle="butt", solid_joinstyle="miter",
                         zorder=1)
        self.ax.annotate("", xy=pts[-1], xytext=pts[-2], zorder=1,
                         arrowprops=dict(arrowstyle="-|>" if head else "-", color=color, lw=LW, ls=ls, shrinkA=0,
                                         shrinkB=0, mutation_scale=6))
        self.segs += list(zip(pts[:-1], pts[1:]))

    def dot(self, x, y, color=LINE):
        self.ax.add_patch(Circle((x, y), 1.5, fc=color, ec="none", zorder=4))

    def box(self, x0, y0, x1, y1, fc, ec=LINE, ls="-", zorder=2):
        p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={RAD}", fc=fc, ec=ec,
                           lw=LW, ls=ls, zorder=zorder)
        self.ax.add_patch(p)
        return p

    def block(self, x0, y0, x1, y1, title=None, sub=None, fc=GRY, color=INK, bold=False, fs=7.5, sfs=7):
        self.box(x0, y0, x1, y1, fc)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        ts = []
        if sub:
            ts.append(self.text(cx, cy + 5.6, title, fs, ha="center", va="center", color=color, bold=bold))
            ts.append(self.text(cx, cy - 5.6, sub, sfs, ha="center", va="center", color=MUTE))
        elif title:
            ts.append(self.text(cx, cy, title, fs, ha="center", va="center", color=color, bold=bold))
        self.inside += [(t, x0, y0, x1, y1) for t in ts]

    def circle(self, cx, cy, r, s, ec=LINE, fc="white", fs=7.5, lw=LW):
        self.ax.add_patch(Circle((cx, cy), r, fc=fc, ec=ec, lw=lw, zorder=3))
        self.text(cx, cy, s, fs, ha="center", va="center", check=False)

    def plus(self, cx, cy, r=4.5):
        self.ax.add_patch(Circle((cx, cy), r, fc=GRY, ec=LINE, lw=LW, zorder=3))
        self.ax.plot([cx - r * .55, cx + r * .55], [cy, cy], color=LINE, lw=LW, zorder=4)
        self.ax.plot([cx, cx], [cy - r * .55, cy + r * .55], color=LINE, lw=LW, zorder=4)

    def lock(self, cx, cy, s=1.0):
        self.ax.add_patch(Rectangle((cx - 2.8 * s, cy - 2.6 * s), 5.6 * s, 4.4 * s, fc=MUTE, ec="none", zorder=4))
        self.ax.add_patch(Arc((cx, cy + 1.8 * s), 3.6 * s, 5.0 * s, theta1=0, theta2=180, color=MUTE, lw=.8, zorder=4))

    def width(self, t):
        self.fig.canvas.draw()
        b = t.get_window_extent(self.fig.canvas.get_renderer())
        inv = self.ax.transData.inverted()
        return inv.transform((b.x1, 0))[0] - inv.transform((b.x0, 0))[0]

    def check(self, name):
        """Print every pair of texts closer than 1.5 units side by side, and every text crossed by a drawn line."""
        self.fig.canvas.draw()
        ren = self.fig.canvas.get_renderer(); inv = self.ax.transData.inverted()
        boxes = []
        for t in self.texts:
            if not t.get_text():
                continue
            b = t.get_window_extent(ren)
            (x0, y0), (x1, y1) = inv.transform([(b.x0, b.y0), (b.x1, b.y1)])
            boxes.append((t.get_text(), x0, y0, x1, y1))
        bad = 0
        for i, (s, x0, y0, x1, y1) in enumerate(boxes):
            for (s2, a0, b0, a1, b1) in boxes[i + 1:]:
                if min(x1, a1) - max(x0, a0) > -1.5 and min(y1, b1) - max(y0, b0) > 0:
                    print(f"  [{name}] text/text: {s!r} x {s2!r} gap {max(x0, a0) - min(x1, a1):.1f}"); bad += 1
            for (p, q) in self.segs:
                for u in np.linspace(0, 1, 200):
                    x, y = p[0] + u * (q[0] - p[0]), p[1] + u * (q[1] - p[1])
                    if x0 + .5 < x < x1 - .5 and y0 + .5 < y < y1 - .5:
                        print(f"  [{name}] text/line: {s!r} x segment {p}->{q}"); bad += 1
                        break
        for (t, bx0, by0, bx1, by1) in self.inside:       # text inside a block: at least 2 units from its sides
            b = t.get_window_extent(ren)
            (x0, y0), (x1, y1) = inv.transform([(b.x0, b.y0), (b.x1, b.y1)])
            if x0 < bx0 + 2 or x1 > bx1 - 2:
                print(f"  [{name}] text too wide for its block: {t.get_text()!r} ({x1 - x0:.1f} in {bx1 - bx0:.1f})"); bad += 1
        print(f"{name}: {len(boxes)} texts, {len(self.segs)} segments, {bad} collisions")


def pca_grid(ax, rgb, rows, cols, x0, y0, w, h, ec, soften=.35):
    """PCA colors of the patch tokens as seamless vector cells (one cell = one token), row 0 at the top. The colors are
    mixed with white (soften) so the thumbnail does not dominate the figure; the token structure is unchanged."""
    c = np.asarray(rgb, float).reshape(rows, cols, 3) / 255
    c = (1 - soften) * c + soften
    cw, ch = w / cols, h / rows
    for r in range(rows):
        for k in range(cols):
            ax.add_patch(Rectangle((x0 + k * cw, y0 + (rows - 1 - r) * ch), cw, ch, fc=c[r, k], ec=c[r, k], lw=.25, zorder=2))
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, ec=ec, lw=LW, zorder=3))


def heat(ax, vals, rows, cols, x0, y0, w, h, color, gamma=.6, amax=.85):
    """Photo in light grey, one tinted cell per token (no smoothing: the map has only rows x cols values)."""
    ax.imshow(GREY, extent=(x0, x0 + w, y0, y0 + h), interpolation="none", zorder=2)
    v = np.asarray(vals, float).reshape(rows, cols); v = v / max(v.max(), 1e-9)
    cw, ch = w / cols, h / rows
    for r in range(rows):
        for k in range(cols):
            if v[r, k] > .02:
                ax.add_patch(Rectangle((x0 + k * cw, y0 + (rows - 1 - r) * ch), cw, ch, fc=color, ec="none",
                                       alpha=float(v[r, k] ** gamma * amax), zorder=3))
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, ec=color, lw=LW, zorder=4))


def encoder(cv, x0, cy, n_frozen, n_total, tint, unit=3.4, pe=6, h=24):
    """One rounded bar per encoder, width proportional to its depth (patch embedding + blocks). The frozen part (patch
    embedding + first n_frozen blocks) is white, hatched, with a lock; the fine-tuned part has the branch tint.
    Returns the x of the frozen / fine-tuned border and the right end."""
    xb, x1 = x0 + pe + n_frozen * unit, x0 + pe + n_total * unit
    outline = cv.box(x0, cy - h / 2, x1, cy + h / 2, tint, zorder=2)
    fz = Rectangle((x0, cy - h / 2), xb - x0, h, fc="white", ec="none", hatch="////", hatchcolor=HATCH, lw=0, zorder=2.1)
    cv.ax.add_patch(fz); fz.set_clip_path(outline)
    cv.ax.add_patch(FancyBboxPatch((x0, cy - h / 2), x1 - x0, h, boxstyle=f"round,pad=0,rounding_size={RAD}", fill=False,
                                   ec=LINE, lw=LW, zorder=2.3))
    cv.ax.plot([xb, xb], [cy - h / 2, cy + h / 2], color=LINE, lw=LW, zorder=2.3)
    cv.ax.add_patch(Circle(((x0 + xb) / 2, cy), 5.2, fc="white", ec="none", zorder=3.5))
    cv.lock((x0 + xb) / 2, cy - .3, .9)
    return xb, x1


# ================================================================ Figure 1: UniQuery
W1, H1 = 6.875, 3.05
cv = Canvas(W1, H1); ax = cv.ax
YD, YJ, YC = 221, 161, 101                               # rows: DINOv3, joint, CLIP

# ---- input crop (224 x 112 as the model sees it)
PX0, PW, PH = 6, 36, 72
ax.imshow(PHOTO, extent=(PX0, PX0 + PW, YJ - PH / 2, YJ + PH / 2), zorder=2, interpolation="none")
ax.add_patch(Rectangle((PX0, YJ - PH / 2), PW, PH, fill=False, ec=LINE, lw=LW, zorder=3))
cv.text(PX0 + PW / 2, YJ - PH / 2 - 4, "input", 7.5, ha="center", va="top")
cv.text(PX0 + PW / 2, YJ - PH / 2 - 14, "224 × 112", 7, ha="center", va="top", color=MUTE)
XS, EX0 = 52, 60                                         # split of the input to the two encoders, encoder start
cv.path([(PX0 + PW, YJ), (XS, YJ), (XS, YD), (EX0, YD)])
cv.path([(XS, YJ), (XS, YC), (EX0, YC)])
cv.dot(XS, YJ)

# ---- encoders
GX0, GW, GH = 190, 28, 56                                # token grids (14 x 7 and 16 x 8 tokens, same frame)
for (cy, name, nf, nt, tint, col) in [(YD, "DINOv3 ViT-H+/16", 8, 32, HP_T, HP), (YC, "CLIP ViT-L/14", 8, 24, CL_T, CL)]:
    xb, xe = encoder(cv, EX0, cy, nf, nt, tint)
    cv.text(EX0, cy + 16, name, 8, ha="left", va="bottom", color=col, bold=True)
    cv.text(EX0, cy - 16, f"{nf} frozen, {nt - nf} fine-tuned blocks", 7, ha="left", va="top", color=MUTE)
    cv.path([(xe, cy), (GX0, cy)], color=col)

# ---- token grids (real PCA of the patch tokens), the CLS token as one separate cell
CS = 4.0
for (cy, key, rows, cols, col, n, dim, cls_up) in [(YD, "pca_h", 14, 7, HP, 99, 1280, True),
                                                   (YC, "pca_c", 16, 8, CL, 129, 1024, False)]:
    y0 = cy - GH / 2
    pca_grid(ax, D[key], rows, cols, GX0, y0, GW, GH, col)
    ycls = y0 + GH + 2 if cls_up else y0 - 2 - CS
    ax.add_patch(Rectangle((GX0, ycls), CS, CS, fc="#B9C1CC", ec=col, lw=LW, zorder=3))
    cv.text(GX0 + CS + 2, ycls + CS / 2, "CLS", 7, ha="left", va="center", color=MUTE)
    cv.text(GX0 + GW / 2, y0 - 4 if cls_up else y0 + GH + 4, f"{n} × {dim}", 7.5, ha="center",
            va="top" if cls_up else "bottom")

# ---- joint tokens: one linear layer (with bias) per encoder, then concatenation (228 x 1024). The code also adds a
#      learned source embedding per encoder, which folds into the bias of that linear layer, so it is not drawn.
GXR, BX = GX0 + GW, 232                                  # grid right edge, T-junction on each token path
WX0, WX1 = 238, 290
DW = 30                                                  # W blocks at YJ +- DW
CN0, CN1 = 296, 326                                      # concat node
CXM = (CN0 + CN1) / 2
TB0, TB1 = 336, 368                                      # joint token bar: 99 DINOv3 + 129 CLIP tokens
for (cy, yw, col, lab) in ((YD, YJ + DW, HP, r"linear $W_D$"), (YC, YJ - DW, CL, r"linear $W_C$")):
    cv.dot(BX, cy, col)
    cv.path([(BX, cy), (BX, yw), (WX0, yw)], color=col)
    cv.block(WX0, yw - 8, WX1, yw + 8, lab, fc=JT_T, fs=7.5)
    s = 1 if yw > YJ else -1
    cv.path([(WX1, yw), (CXM, yw), (CXM, YJ + s * 8)], color=JT)
cv.block(CN0, YJ - 8, CN1, YJ + 8, "concat", fs=7)
cv.path([(CN1, YJ), (TB0, YJ)], color=JT)
tb = TB0 + (TB1 - TB0) * 99 / 228
ax.add_patch(Rectangle((TB0, YJ - 4.5), tb - TB0, 9, fc=HP_T, ec=HP, lw=LW, zorder=3))
ax.add_patch(Rectangle((tb, YJ - 4.5), TB1 - tb, 9, fc=CL_T, ec=CL, lw=LW, zorder=3))
cv.text((TB0 + TB1) / 2, YJ + 8, "228 × 1024", 7.5, ha="center", va="bottom")

# ---- three query heads (same design, Figure 2)
QX0, QX1, QH = 382, 480, 13
for (cy, col) in ((YD, HP), (YC, CL)):
    cv.path([(GXR, cy), (QX0, cy)], color=col)
cv.path([(TB1, YJ), (QX0, YJ)], color=JT)
for (cy, name, sub, col, tint) in [(YD, "DINOv3 query head", "reads 99 tokens", HP, HP_T),
                                   (YJ, "joint query head", "reads 228 tokens", JT, JT_T),
                                   (YC, "CLIP query head", "reads 129 tokens", CL, CL_T)]:
    cv.block(QX0, cy - QH, QX1, cy + QH, name, sub, fc=tint, color=col, bold=True, fs=7.5)

# ---- per-attribute outputs: every query head gives 40 logits, one cell per attribute (json attribute order; "hat" =
# Accessory-Hat is the last cell, outlined). Neutral slate scale, white = low, dark = high (no clash with branch hues).
VAL = matplotlib.colors.LinearSegmentedColormap.from_list("val", ["#FFFFFF", "#2B3A4E"])
LNORM = matplotlib.colors.Normalize(-6.5, 6.5)           # logits of the three heads lie in [-5.2, 6.4]
SWD, SHT = 60, 9                                          # strip width (40 cells of 1.5) and height


def strip(x0, yc, colors, ec=LINE):
    """40 cells, one per attribute (real values); the hat cell (index K, the last one) gets an outline."""
    cw = SWD / len(colors)
    for i, c in enumerate(colors):
        ax.add_patch(Rectangle((x0 + i * cw, yc - SHT / 2), cw, SHT, fc=c, ec=c, lw=.2, zorder=3))
    ax.add_patch(Rectangle((x0, yc - SHT / 2), SWD, SHT, fill=False, ec=ec, lw=LW, zorder=4))
    ax.add_patch(Rectangle((x0 + K * cw - 1, yc - SHT / 2 - 1.5), cw + 2, SHT + 3, fill=False, ec=INK, lw=.9, zorder=5))
    return x0 + SWD


# ---- CLIP text score, 40 values (TextAnchor in heads.py, prompts in make_clip_text.py):
# image side  CLS -> frozen CLIP projection P (1024 -> 768) -> L2-normalised e
# text side   frozen CLIP text encoder, offline: prompt of each attribute (4 templates, averaged) minus "a person" -> w_k,
#             plus a learned correction 0.1 delta_k (delta starts at 0)
# score       100 <e, w_k + 0.1 delta_k> + b_k (b_k learned), added to the 40 logits of the CLIP query head
YA, YB, YS = 62, 18, 40                                   # image-side row, text-side row, score node row


def fblock(x0, y0, x1, y1, title, sub=None, fs=7.5):
    """Frozen block: white, light hatching, lock at the left; text on a white backing so the hatching never cuts it."""
    b = cv.box(x0, y0, x1, y1, "white")
    b.set_hatch("////"); b.set_hatchcolor(HATCH)
    cv.lock(x0 + 6.5, (y0 + y1) / 2 - .3, .8)
    cx, cy = (x0 + 12 + x1) / 2, (y0 + y1) / 2
    bb = dict(boxstyle="square,pad=0.12", fc="white", ec="none")
    ts = [cv.text(cx, cy + (5.6 if sub else 0), title, fs, ha="center", va="center", bbox=bb)]
    if sub:
        ts.append(cv.text(cx, cy - 5.6, sub, 7, ha="center", va="center", color=MUTE, bbox=bb))
    cv.inside += [(t, x0 + 10, y0, x1, y1) for t in ts]


# image side
PX_0, PX_1 = 226, 270
cv.path([(GX0 + CS / 2, YC - GH / 2 - 2 - CS), (GX0 + CS / 2, YA), (PX_0, YA)], color=CL)    # from the CLIP CLS cell
fblock(PX_0, YA - 8, PX_1, YA + 8, "proj. P")
NX0, NX1 = 296, 372                                       # score node
NXC = 334
cv.path([(PX_1, YA), (NXC, YA), (NXC, YS + 9)], color=CL)
cv.text((PX_1 + NXC) / 2, YA + 2.5, "e", 7.5, ha="center", va="bottom", style="italic")
# text side
TE0, TE1 = 160, 242
cv.text(TE0 - 12, YB + 5, "“a person wearing a hat or a cap”", 7, ha="right", va="center", style="italic")
cv.text(TE0 - 12, YB - 5, "minus “a person”, 4 templates", 7, ha="right", va="center", color=MUTE)
cv.path([(TE0 - 10, YB), (TE0, YB)], color=CL)
fblock(TE0, YB - QH, TE1, YB + QH, "CLIP text", "encoder, offline")
DL0, DL1 = 262, 300
cv.path([(TE1, YB), (DL0, YB)], color=CL)
cv.text((TE1 + DL0) / 2, YB + 2.5, r"$w_k$", 7.5, ha="center", va="bottom")
cv.block(DL0, YB - 8, DL1, YB + 8, r"$+\,0.1\,\delta_k$", fc=CL_T, fs=7.5)
cv.path([(DL1, YB), (NXC, YB), (NXC, YS - 9)], color=CL)
# score node, 40-cell text-score strip, + into the CLIP path
cv.block(NX0, YS - 9, NX1, YS + 9, r"$100\,\langle e, \cdot\rangle + b_k$", fc=CL_T, fs=7.5)
TS0 = QX1 - SWD
cv.path([(NX1, YS), (TS0, YS)], color=CL)
ct = np.asarray(D["logits"]["c_text"], float)
TS1 = strip(TS0, YS, VAL(LNORM(ct)), ec=CL)
cv.text(TS1 + 1, YS + SHT / 2 + 2.5, "hat +%.1f" % ct[K], 7, ha="right", va="bottom", color=MUTE)
cv.text((TS0 + TS1) / 2, YS - SHT / 2 - 3, "40 text scores", 7, ha="center", va="top", color=MUTE)
PLX, ZX0 = QX1 + 13, QX1 + 26                             # + of the text score, left end of the logit strips
cv.path([(QX1, YC), (PLX - 4.5, YC)], color=CL)
cv.plus(PLX, YC)
cv.path([(TS1, YS), (PLX, YS), (PLX, YC - 4.5)], color=CL)

cv.path([(QX1, YD), (ZX0, YD)], color=HP)
cv.path([(QX1, YJ), (ZX0, YJ)], color=JT)
cv.path([(PLX + 4.5, YC), (ZX0, YC)], color=CL)
for cy, key, col in ((YD, "h", HP), (YJ, "j", JT), (YC, "c", CL)):
    v = np.asarray(D["logits"][key], float)
    ZX1 = strip(ZX0, cy, VAL(LNORM(v)), ec=col)
    cv.text(ZX1 + 1, cy + SHT / 2 + 2.5, "hat %.1f" % v[K], 7, ha="right", va="bottom", color=MUTE)

cv.text((ZX0 + ZX1) / 2, YD + SHT / 2 + 20.5, r"40 logits $z_k$ per head", 7, ha="center", va="bottom", color=MUTE)

# ---- training only: one loss per query head on its own 40 logits, with the logit-adjustment shift s (never the mean).
# Each node sits right under its strip, so no dashed line crosses another path; the sum is written in the box below.
BX0, BX1 = ZX0 - 6, ZX1 + 6                               # loss nodes
for cy, k in ((YD, "D"), (YJ, "J"), (YC, "C")):
    cv.path([(ZX0 + 16, cy - SHT / 2 - 1.5), (ZX0 + 16, cy - 19)], dashed=True)
    cv.box(BX0, cy - 33, BX1, cy - 19, "white", ls=(0, (3, 2)))
    t = cv.text((BX0 + BX1) / 2, cy - 26, r"$\mathrm{BCE}(z_%s+s,\,y)$" % k, 7, ha="center", va="center")
    cv.inside.append((t, BX0, cy - 33, BX1, cy - 19))
TR0, TR1, TRY0, TRY1 = BX0, 684, 5, 39                    # training box: the total loss
cv.box(TR0, TRY0, TR1, TRY1, "white", ls=(0, (3, 2)))
cv.text(TR0 + 5, TRY1 - 4, "training only", 7, ha="left", va="top", color=MUTE, style="italic")
cv.text((TR0 + TR1) / 2, TRY0 + 10, r"$L=\Sigma_{k\in\{D,J,C\}}\,\mathrm{BCE}(z_k+s,\;y)$", 7.5, ha="center", va="center")

# ---- inference only: per attribute, mean of the three heads, sigmoid -> 40 probabilities, > 0.5 -> 40 predictions
IR, IRE = "#EEF1F4", "#C9D0D8"                         # inference region fill and border
IX0, IX1, IY0, IY1 = BX1 + 6, 686, YC - 28, YD + 20
cv.box(IX0, IY0, IX1, IY1, IR, ec=IRE, zorder=0.5)
cv.text(IX1 - 4, IY1 - 3, "inference only", 7, ha="right", va="top", color=MUTE, style="italic")
MXC = IX0 + 18                                            # mean node; the DINOv3 / CLIP strips join it vertically
cv.block(MXC - 13, YJ - 8, MXC + 13, YJ + 8, "mean", fs=7.5)
cv.path([(ZX1, YD), (MXC, YD), (MXC, YJ + 8)])
cv.path([(ZX1, YJ), (MXC - 13, YJ)])
cv.path([(ZX1, YC), (MXC, YC), (MXC, YJ - 8)])
OX0 = 682 - SWD                                           # probability / prediction strips, right-aligned in the region
SXC, SR = OX0 + 14, 7                                     # sigma, above the left end of the strips
cv.path([(MXC + 13, YJ), (SXC - SR, YJ)])
cv.circle(SXC, YJ, SR, r"$\sigma$", fc=GRY, fs=8)
YP, YQ = YJ - 32, YJ - 64                                 # probability strip, prediction strip
p = np.asarray(D["prob"], float)
cv.path([(SXC, YJ - SR), (SXC, YP + SHT / 2 + 1.5)])
strip(OX0, YP, VAL(p))
cv.text(OX0 + SWD + 1, YP + SHT / 2 + 2.5, "hat %.3f" % p[K], 7, ha="right", va="bottom", color=MUTE)
cv.path([(SXC, YP - SHT / 2 - 1.5), (SXC, YQ + SHT / 2 + 1.5)])
cv.text(SXC + 3, (YP + YQ) / 2, "> 0.5", 7, ha="left", va="center", color=MUTE)
strip(OX0, YQ, ["#2B3A4E" if x > .5 else "white" for x in p])
cv.text(OX0 + SWD + 1, YQ - SHT / 2 - 3, "hat: present", 7.5, ha="right", va="top", bold=True)


# ---- legend: branch hues and value scale (left group), block kinds (right group), one row on top
LY = 0
SW, GAP = 14, 12


def legend(items, x, align, ly):
    global LY
    LY = ly
    parts = []
    for kind, lab in items:
        t = cv.text(0, LY, lab, 7, ha="left", va="center", color=MUTE)
        parts.append((kind, t, cv.width(t)))
    extra = {"frozen": 9}
    total = sum(SW + 3 + w + extra.get(k, 0) for k, _, w in parts) + GAP * (len(parts) - 1)
    lx = x if align == "left" else x - total
    for kind, t, w in parts:
        if kind == "frozen":
            b = cv.box(lx, LY - 5, lx + SW, LY + 5, "white")
            b.set_hatch("////"); b.set_hatchcolor(HATCH)
            cv.lock(lx + SW + 5, LY - .3, .75)
            lx += 9
        elif kind == "tuned":
            cv.box(lx, LY - 5, lx + SW, LY + 5, JT_T)
        elif kind == "op":
            cv.box(lx, LY - 5, lx + SW, LY + 5, GRY)
        elif kind == "dash":
            cv.box(lx, LY - 5, lx + SW, LY + 5, "white", ls=(0, (3, 2)))
        elif kind == "infer":
            cv.box(lx, LY - 5, lx + SW, LY + 5, IR, ec=IRE)
        elif kind == "scale":
            for i in range(14):                           # vector steps, no raster
                ax.add_patch(Rectangle((lx + i * SW / 14, LY - 4.5), SW / 14, 9, fc=VAL(i / 13), ec=VAL(i / 13), lw=.2,
                                       zorder=2))
            ax.add_patch(Rectangle((lx, LY - 4.5), SW, 9, fill=False, ec=LINE, lw=LW, zorder=3))
        else:                                             # branch hue
            cv.path([(lx, LY), (lx + SW, LY)], color=kind)
        t.set_x(lx + SW + 3)
        lx += SW + 3 + w + GAP


legend([(HP, "DINOv3"), (CL, "CLIP"), (JT, "joint (both encoders)"), ("scale", "low → high value")],
       EX0, "left", 296)
legend([("frozen", "frozen"), ("tuned", "trained"), ("op", "parameter-free"), ("dash", "training only"), ("infer", "inference only")],
       EX0, "left", 281)


cv.check("uniquery_model")
cv.fig.savefig(OUT / "uniquery_model.pdf"); cv.fig.savefig(OUT / "uniquery_model.png", dpi=300); plt.close(cv.fig)


# ================================================================ Figure 2: one query head
W2, H2 = 3.28, 2.95
cv = Canvas(W2, H2); ax = cv.ax
NEUT = "#C9CFD7"                                         # vector circles


def stack(cx, cy, n=4, layers=3):
    """A set of vectors: a row of circles in a rounded frame, two frames behind it."""
    w, h = n * 8.4 + 4, 10
    for L in range(layers - 1, -1, -1):
        x0, y0 = cx - w / 2 - 1.8 * L, cy - h / 2 + 1.8 * L
        ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle=f"round,pad=0,rounding_size={RAD}", fc="white", ec=LINE,
                                    lw=LW, zorder=2))
        for i in range(n):
            ax.add_patch(Circle((x0 + 6.2 + 8.4 * i, y0 + h / 2), 2.9, fc=NEUT if L == 0 else "white", ec=LINE, lw=.5,
                                zorder=3))
    return cx - w / 2 - 3.6, cy - h / 2, cx + w / 2, cy + h / 2 + 3.6


S, XQ, XT = 46, 122, 250                                 # residual stream of the queries, Q branch, token (K, V) branch
LNW, LNH = 28, 6.5                                       # half width / half height of a LayerNorm block
MH = 13                                                  # half height of a module block
Y0 = 34                                                  # inputs
q0 = stack(S, Y0)
cv.text(S, Y0 - 9, "40 attribute queries", 7.5, ha="center", va="top")
cv.text(S, Y0 - 19, "learned, 40 × d", 7, ha="center", va="top", color=MUTE)
t0 = stack(XT, Y0)
cv.text(XT, Y0 - 9, "CLS + patch tokens", 7.5, ha="center", va="top")
cv.text(XT, Y0 - 19, "N × d,  N = 99, 129 or 228", 7, ha="center", va="top", color=MUTE)

# cross-attention sub-layer: q = q + Attn(Q = LN(q), K = V = LN(tokens))
YB1, YLN1, YATT, YP1 = 54, 72, 106, 134
cv.path([(S, q0[3]), (S, YP1 - 4.5)])
cv.dot(S, YB1)
cv.path([(S, YB1), (XQ, YB1), (XQ, YLN1 - LNH)])
cv.path([(XT, t0[3]), (XT, YLN1 - LNH)])
cv.block(XQ - LNW, YLN1 - LNH, XQ + LNW, YLN1 + LNH, "LayerNorm", fs=7)
cv.block(XT - LNW, YLN1 - LNH, XT + LNW, YLN1 + LNH, "LayerNorm", fs=7)
cv.block(XQ - LNW, YATT - MH, XT + LNW, YATT + MH, "cross-attention", "8 attention heads", fc="white", fs=8)
cv.path([(XQ, YLN1 + LNH), (XQ, YATT - MH)])
cv.path([(XT, YLN1 + LNH), (XT, YATT - MH)])
cv.text(XQ + 3, (YLN1 + LNH + YATT - MH) / 2, "Q", 7.5, ha="left", va="center", bold=True)
cv.text(XT + 3, (YLN1 + LNH + YATT - MH) / 2, "K, V", 7.5, ha="left", va="center", bold=True)
cv.plus(S, YP1)
cv.path([(XQ, YATT + MH), (XQ, YP1), (S + 4.5, YP1)])

# MLP sub-layer: q = q + MLP(LN(q))
YB2, YLN2, YMLP, YP2 = 150, 168, 202, 230
cv.path([(S, YP1 + 4.5), (S, YP2 - 4.5)])
cv.dot(S, YB2)
cv.path([(S, YB2), (XQ, YB2), (XQ, YLN2 - LNH)])
cv.block(XQ - LNW, YLN2 - LNH, XQ + LNW, YLN2 + LNH, "LayerNorm", fs=7)
cv.block(XQ - 44, YMLP - MH, XQ + 44, YMLP + MH, "MLP", "d → 2d → d, GELU", fc="white", fs=8)
cv.path([(XQ, YLN2 + LNH), (XQ, YMLP - MH)])
cv.plus(S, YP2)
cv.path([(XQ, YMLP + MH), (XQ, YP2), (S + 4.5, YP2)])
cv.text(S - 4, (YP1 + YP2) / 2, "40 × d", 7, ha="right", va="center", color=MUTE, rotation=90)

# output: LayerNorm, one linear classifier per attribute, 40 logits
YO = 264
cv.path([(S, YP2 + 4.5), (S, YO), (XQ - LNW, YO)])
cv.block(XQ - LNW, YO - LNH, XQ + LNW, YO + LNH, "LayerNorm", fs=7)
CL0, CL1 = 162, 236
cv.path([(XQ + LNW, YO), (CL0, YO)])
cv.block(CL0, YO - MH, CL1, YO + MH, "40 classifiers", "one per attribute", fc="white", fs=8)
z = stack(XT + 23, YO - 3, layers=1)
cv.path([(CL1, YO), (z[0] + 3.6, YO)])
cv.text(XT + 23, YO + 6, "40 logits", 7.5, ha="center", va="bottom")

# real attention of the "hat" query (DINOv3 query head, mean over its 8 attention heads) on its 14 x 7 patch tokens
HW_, HH_ = 26, 52
HX0, HY0 = XT - HW_ / 2, YMLP - HH_ / 2 - 4
heat(ax, np.asarray(D["att_h"][K], float)[1:], 14, 7, HX0, HY0, HW_, HH_, HP)
cv.text(HX0 + HW_ / 2, HY0 + HH_ + 3, "“hat” query attention", 7, ha="center", va="bottom", color=HP)
cv.text(HX0 + HW_ / 2, HY0 - 3, "(DINOv3 head)", 7, ha="center", va="top", color=MUTE)

cv.text(3, H2 * 100 - 4, "one layer,  d = 1024  (DINOv3 head: 1280)", 7, ha="left", va="top", color=MUTE)

cv.check("query_head")
cv.fig.savefig(OUT / "query_head.pdf"); cv.fig.savefig(OUT / "query_head.png", dpi=300); plt.close(cv.fig)
print("written", OUT / "uniquery_model.pdf", OUT / "query_head.pdf")
