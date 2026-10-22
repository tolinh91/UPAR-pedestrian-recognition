"""Figure 4 v2: how the joint query head splits its attention heads between the two encoders.

../../../Upar-Challenge-Track-2/.venv/bin/python make_routing_fig.py
figs/routing_v2.pdf (+ .png, 300 dpi), figure* at text width (6.875 in).
(a) eight people, one target attribute each (submitted v31 model; these val images were seen in training):
    the crop in grey, the attention of that attribute's joint query, summed over the attention heads that read mostly
    DINOv3 (share > 0.5, blue cells on the 14 x 7 DINOv3 patch grid) and over the heads that read CLIP (orange cells on the
    16 x 8 CLIP patch grid). One cell per token, no smoothing; each map is scaled to its own maximum; CLS tokens not drawn.
    Data: remote_results/na/fig4/fig4_multi.npz (export_multi.py).
(b) histogram of the DINOv3 share s of every (image, attention head, attribute) triple over the 6,963 held-out PETA
    images (PETA-fold joint model, PETA never seen in training). routing_stats.py dropped the triples whose float32 share
    came out slightly above 1 (np.histogram range (0, 1)); they are added back to the last bin (see check below).
(c) per attribute: number of the 8 attention heads that read DINOv3 (s > 0.5), mean over the same 6,963 images, with the
    10-90% range over images. Data (b, c): remote_results/duf2/logits/route_DU32_PETA_s1.npz (routing_stats.py).
Style as make_arch_figs.py: Times New Roman, STIX math, Liberation Serif Bold for bold, 0.7 pt strokes, units 1/100 in,
collision check (text vs. text, text vs. drawn lines) before saving.
"""
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE.parent / "figs"
ROUTE = Path("/media/tan/F/RWS2027/remote_results/duf2/logits/route_DU32_PETA_s1.npz")
MULTI = Path("/media/tan/F/RWS2027/remote_results/na/fig4/fig4_multi.npz")

FONTS = Path.home() / ".local/share/fonts"
for f in ("TimesNewRoman-Regular.ttf", "TimesNewRoman-Italic.ttf"):
    fm.fontManager.addfont(str(FONTS / f))
BOLD_TTF = "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 7.5, "pdf.fonttype": 42, "mathtext.fontset": "stix",
                     "text.color": "#18212D", "axes.edgecolor": "#2B3A4E", "xtick.color": "#2B3A4E",
                     "ytick.color": "#2B3A4E", "axes.linewidth": .7})
HP, CL = "#1F5BD8", "#D2731A"
HP_T, CL_T = "#9DB8EE", "#EDB98A"                       # bar fills (lighter, still clearly blue / orange)
INK, MUTE, LINE = "#18212D", "#5B6675", "#2B3A4E"
LW, RAD, FS_MIN = .7, 3, 6.8

ATTRS = ["Age-Young", "Age-Adult", "Age-Old", "Gender-Female", "Hair-Length-Short", "Hair-Length-Long", "Hair-Length-Bald",
         "UpperBody-Length-Short", "UpperBody-Color-Black", "UpperBody-Color-Blue", "UpperBody-Color-Brown",
         "UpperBody-Color-Green", "UpperBody-Color-Grey", "UpperBody-Color-Orange", "UpperBody-Color-Pink",
         "UpperBody-Color-Purple", "UpperBody-Color-Red", "UpperBody-Color-White", "UpperBody-Color-Yellow",
         "UpperBody-Color-Other", "LowerBody-Length-Short", "LowerBody-Color-Black", "LowerBody-Color-Blue",
         "LowerBody-Color-Brown", "LowerBody-Color-Green", "LowerBody-Color-Grey", "LowerBody-Color-Orange",
         "LowerBody-Color-Pink", "LowerBody-Color-Purple", "LowerBody-Color-Red", "LowerBody-Color-White",
         "LowerBody-Color-Yellow", "LowerBody-Color-Other", "LowerBody-Type-Trousers&Shorts", "LowerBody-Type-Skirt&Dress",
         "Accessory-Backpack", "Accessory-Bag", "Accessory-Glasses-Normal", "Accessory-Glasses-Sun", "Accessory-Hat"]


def short(a):
    for old, new in [("UpperBody-Length-Short", "short sleeves"), ("LowerBody-Length-Short", "short lower body"),
                     ("LowerBody-Type-Trousers&Shorts", "trousers/shorts"), ("LowerBody-Type-Skirt&Dress", "skirt/dress"),
                     ("UpperBody-Color-", "upper "), ("LowerBody-Color-", "lower "), ("Hair-Length-", "hair "),
                     ("Accessory-Glasses-Normal", "glasses"), ("Accessory-Glasses-Sun", "sunglasses"), ("Accessory-", ""),
                     ("Gender-", ""), ("Age-", "")]:
        a = a.replace(old, new)
    return a.lower()


class Canvas:
    """Axes in 1/100 inch plus a record of every drawn segment, for the collision check."""

    def __init__(self, w_in, h_in):
        self.fig = plt.figure(figsize=(w_in, h_in))
        self.W, self.H = w_in * 100, h_in * 100
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, self.W); self.ax.set_ylim(0, self.H); self.ax.axis("off")
        self.segs, self.texts = [], []

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

    def line(self, pts, color=LINE, lw=LW, ls="-", zorder=2):
        xs, ys = zip(*pts)
        self.ax.plot(xs, ys, color=color, lw=lw, ls=ls, solid_capstyle="butt", zorder=zorder)
        self.segs += list(zip(pts[:-1], pts[1:]))

    def check(self, name, extra_axes=()):
        self.fig.canvas.draw()
        ren = self.fig.canvas.get_renderer(); inv = self.ax.transData.inverted()
        texts = list(self.texts)
        for a in extra_axes:                             # tick labels and axis labels of data axes
            texts += [t for t in a.get_xticklabels() + a.get_yticklabels() if t.get_visible() and t.get_text()]
            texts += [a.xaxis.label, a.yaxis.label]
        boxes = []
        for t in texts:
            if not t.get_text():
                continue
            b = t.get_window_extent(ren)
            (x0, y0), (x1, y1) = inv.transform([(b.x0, b.y0), (b.x1, b.y1)])
            boxes.append((t.get_text(), x0, y0, x1, y1))
        bad = 0
        for i, (s, x0, y0, x1, y1) in enumerate(boxes):
            if x0 < 0 or y0 < 0 or x1 > self.W or y1 > self.H:
                print(f"  [{name}] text outside the canvas: {s!r}"); bad += 1
            for (s2, a0, b0, a1, b1) in boxes[i + 1:]:
                if min(x1, a1) - max(x0, a0) > -1.5 and min(y1, b1) - max(y0, b0) > 0:
                    print(f"  [{name}] text/text: {s!r} x {s2!r}"); bad += 1
            for (p, q) in self.segs:
                for u in np.linspace(0, 1, 200):
                    x, y = p[0] + u * (q[0] - p[0]), p[1] + u * (q[1] - p[1])
                    if x0 + .5 < x < x1 - .5 and y0 + .5 < y < y1 - .5:
                        print(f"  [{name}] text/line: {s!r} x segment {p}->{q}"); bad += 1
                        break
        print(f"{name}: {len(boxes)} texts, {len(self.segs)} segments, {bad} collisions")


R = np.load(ROUTE)
N_IMG = int(R["n_images"]); assert str(R["holdout"]) == "PETA" and N_IMG == 6963
HIST = R["hist"].astype(float)
TOTAL = N_IMG * 8 * 40
MISSING = TOTAL - HIST.sum()
# the dropped triples are shares that came out just above 1 in float32: adding them at s = 1 reproduces the saved
# frac_dino and n_dino exactly (all of them have s > 0.5) and the saved mean_share to 0.001
assert abs((HIST[25:].sum() + MISSING) / TOTAL - R["frac_dino"].mean()) < 1e-6
HIST[-1] += MISSING
ND = R["n_dino"].astype(float)                           # (6963, 40) heads with s > 0.5

W, H = 6.875, 3.62
cv = Canvas(W, H); ax = cv.ax
TOP = cv.H

# ================================================================ (a) eight people, one attribute each
GREYF = lambda rgb: np.repeat((.45 + .5 * (rgb @ np.array([.299, .587, .114])))[..., None], 3, -1)  # noqa: E731


def heat(vals, rows, cols, x0, y0, w, h, grey, color, gamma=.6, amax=.85):
    """Photo in light grey, one tinted cell per token (no smoothing: the map has only rows x cols values)."""
    cv.ax.imshow(grey, extent=(x0, x0 + w, y0, y0 + h), interpolation="none", zorder=2)
    v = np.asarray(vals, float).reshape(rows, cols)
    v = v / v.max() if v.max() > 0 else v
    cw, ch = w / cols, h / rows
    for r in range(rows):
        for k in range(cols):
            if v[r, k] > .02:
                cv.ax.add_patch(Rectangle((x0 + k * cw, y0 + (rows - 1 - r) * ch), cw, ch, fc=color, ec="none",
                                          alpha=float(v[r, k] ** gamma * amax), zorder=3))
    cv.ax.add_patch(Rectangle((x0, y0), w, h, fill=False, ec=color, lw=LW, zorder=4))


cv.text(4, TOP - 3, "(a)", 8, bold=True, ha="left", va="top")
if MULTI.exists():
    M = np.load(MULTI, allow_pickle=False)
    AT = [str(a) for a in M["attrs"]]; assert AT == ATTRS
    att = M["att_j"].astype(np.float32)                    # (8 images, 40 attrs, 8 heads, 228 tokens)
    n_ex = len(M["rows"])
    IW, IH, IG = 28, 56, 2                                 # one copy; pair = 2 copies + gap
    PAIR = 2 * IW + IG
    PIT = (cv.W - 8) / n_ex
    YI0 = TOP - 35 - IH                                    # bottom of the images
    A_EX = []
    for e in range(n_ex):
        t = str(M["targets"][e]); ti = ATTRS.index(t)
        a = att[e, ti]                                     # (8 heads, 228)
        s = a[:, :99].sum(1) / a.sum(1)                    # DINOv3 share of each head
        on_d = s > .5; k = int(on_d.sum())
        md = a[on_d][:, 1:99].sum(0) if k else np.zeros(98)
        mc = a[~on_d][:, 100:228].sum(0) if k < 8 else np.zeros(128)
        rgb = np.asarray(Image.open(io.BytesIO(M[f"raw{e}"].tobytes())).convert("RGB")
                         .resize((112 * 8, 224 * 8), Image.LANCZOS), float) / 255
        grey = GREYF(rgb)
        x0 = 4 + e * PIT + (PIT - PAIR) / 2
        heat(md, 14, 7, x0, YI0, IW, IH, grey, HP)
        heat(mc, 16, 8, x0 + IW + IG, YI0, IW, IH, grey, CL)
        cx = x0 + PAIR / 2
        p, g = float(M["prob"][e, ti]), int(M["gt"][e, ti])
        cv.text(cx, YI0 + IH + 11.5, short(t), 7.5, bold=True, ha="center", va="bottom")
        cv.text(cx - 2.5, YI0 + IH + 1.5, f"{k} DINOv3", 7, ha="right", va="bottom", color=HP)
        cv.text(cx + 2.5, YI0 + IH + 1.5, f"{8 - k} CLIP", 7, ha="left", va="bottom", color=CL)
        cv.text(cx, YI0 - 3, f"p = {p:.3f} · label {g}", 7, ha="center", va="top")
        cv.text(cx, YI0 - 12.5, str(M["ds"][e]).replace("Market1501", "Market-1501").replace("PA100k", "PA-100K"), 7,
                ha="center", va="top", color=MUTE)
        A_EX.append((short(t), str(M["ds"][e]), int(M["rows"][e]), round(p, 3), g, k, np.round(s, 3).tolist()))
    for r in A_EX:
        print("(a)", r)
else:
    print("(a) MISSING", MULTI, "- panel (a) left empty (draft only)")
# legend of (a), top line right
LYA, lx = TOP - 9, 150
for col, lab in [(HP, "k DINOv3: the k heads that read DINOv3, on its 14 × 7 tokens"),
                 (CL, "8 \u2212 k CLIP: the other heads, on CLIP\u2019s 16 × 8 tokens")]:
    cv.ax.add_patch(Rectangle((lx, LYA - 4), 10, 8, fc=col, alpha=.6, ec=col, lw=LW, zorder=3))
    t = cv.text(lx + 13, LYA, lab, 7, ha="left", va="center")
    cv.fig.canvas.draw()
    b = t.get_window_extent(cv.fig.canvas.get_renderer())
    lx = cv.ax.transData.inverted().transform((b.x1, 0))[0] + 14


# ================================================================ (b) one bar: where a single attention head looks
f_lo, f_hi = HIST[0] / TOTAL, HIST[-1] / TOTAL           # s < 0.02 (CLIP only), s >= 0.98 (DINOv3 only)
f_mid = 1 - f_lo - f_hi
YB = TOP - 120                                            # top of panel (b)
cv.text(4, YB, "(b)", 8, bold=True, ha="left", va="top")
cv.text(24, YB, "Where one attention head looks when it reads one image for one attribute "
        "(all 8 heads × 40 attributes × 6,963 held-out PETA images, PETA-fold model)", 7.5, ha="left", va="top")
BX0, BX1, BY = 24, cv.W - 6, YB - 24                      # bar extent and centre line
BBH = 10
x = BX0
for frac, col, lab in [(f_lo, CL_T, f"almost only CLIP tokens (< 2% on DINOv3): {100 * f_lo:.0f}%"),
                       (f_mid, "#D5DAE0", f"mixed: {100 * f_mid:.0f}%"),
                       (f_hi, HP_T, f"almost only DINOv3 tokens (> 98% on DINOv3): {100 * f_hi:.0f}%")]:
    w = frac * (BX1 - BX0)
    cv.ax.add_patch(Rectangle((x, BY - BBH / 2), w, BBH, fc=col, ec="white", lw=.6, zorder=2))
    cv.text(x + w / 2, BY - BBH / 2 - 2.5, lab, 7, ha="center", va="top",
            color=CL if col == CL_T else HP if col == HP_T else MUTE)
    x += w

# ================================================================ (c) heads on DINOv3 per attribute
YC = BY - 30                                              # top of panel (c)
cv.text(4, YC, "(c)", 8, bold=True, ha="left", va="top")
cv.text(24, YC, "Number of the 8 attention heads that read DINOv3 (blue, value printed) and that read CLIP (orange), "
        "per attribute, mean over the same images", 7.5, ha="left", va="top")
mean = ND.mean(0)
GROUPS = [("age, gender, hair", range(0, 7)), ("accessories", range(35, 40)), ("upper body", range(7, 20)),
          ("lower body", range(20, 35))]
COLS = [GROUPS[:2], GROUPS[2:3], GROUPS[3:]]
PITCH, BH = 9.6, 6.8
LABW, BARW = 79, 112
U = BARW / 8
YAX = YC - 27
for c, groups in enumerate(COLS):
    x_lab = 92 + c * 229
    xb0 = x_lab + 3
    cv.line([(xb0, YAX), (xb0 + BARW, YAX)], color=LINE)
    for v in range(0, 9, 2):
        cv.line([(xb0 + v * U, YAX), (xb0 + v * U, YAX + 2.5)], color=LINE)
        cv.text(xb0 + v * U, YAX + 3.5, str(v), 7, ha="center", va="bottom")
    y = YAX - 6
    for gname, idx in groups:
        idx = sorted(idx, key=lambda i: -mean[i])
        y_top = y + PITCH / 2
        for i in idx:
            cv.ax.add_patch(Rectangle((xb0, y - BH / 2), mean[i] * U, BH, fc=HP_T, ec="none", zorder=2))
            cv.ax.add_patch(Rectangle((xb0 + mean[i] * U, y - BH / 2), (8 - mean[i]) * U, BH, fc=CL_T, ec="none", zorder=2))
            cv.text(xb0 + mean[i] * U - 1.5, y, f"{mean[i]:.1f}", 6.8, ha="right", va="center", zorder=6)
            cv.text(x_lab - 2, y, short(ATTRS[i]), 7, ha="right", va="center")
            y -= PITCH
        y_bot = y + PITCH / 2
        xg = x_lab - LABW + 2
        cv.line([(xg + 3, y_top - 1), (xg, y_top - 1), (xg, y_bot + 1), (xg + 3, y_bot + 1)], color=MUTE, lw=.6)
        cv.text(xg - 2, (y_top + y_bot) / 2, gname, 7, ha="right", va="center", rotation=90, color=MUTE)
        y -= 5
    print(f"(c) column {c} bottom at y = {y:.1f}")

print("(c) mean heads on DINOv3:", {short(ATTRS[i]): round(float(mean[i]), 2) for i in range(40)})
print(f"(b) only CLIP (s < 0.02) {100 * f_lo:.1f}%, mixed {100 * f_mid:.1f}%, only DINOv3 (s >= 0.98) {100 * f_hi:.1f}%, "
      f"pure99 {100 * float(R['pure99']):.2f}%")

cv.check("routing_v2")
cv.fig.savefig(OUT / "routing_v2.pdf"); cv.fig.savefig(OUT / "routing_v2.png", dpi=300); plt.close(cv.fig)
