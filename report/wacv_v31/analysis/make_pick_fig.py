"""Figure pick (v2): where the joint query head lands next to the two single-encoder query heads, one row per attribute.

../../../../Upar-Challenge-Track-2/.venv/bin/python make_pick_fig.py      -> figs/pick.pdf (+ pick_preview.png here)
Data: remote_results/fusion/logits16b/DTB_<fold>_s<seed>.npz, the UniQuery variant of Tab. main (two seeds, both folds);
keys h_ / c_ / j_w1.00 = logits of the DINOv3 / CLIP / joint query head on the held-out validation images.
Balanced accuracy per attribute (decision logit > 0), mean of the two seeds.
Rows: every attribute whose two single-encoder query heads differ by at least 2 points (two-seed mean) and that has at
least 50 positive held-out images, in two blocks (clothing colors / all other attributes), each sorted by DINOv3 - CLIP.
Each row: DINOv3 query head (blue circle), CLIP query head (orange square), joint query head (purple diamond) on one
shared balanced-accuracy axis; a grey bar joins the two single-encoder heads. Raw values, no smoothing, nothing clipped.
Colors as Fig. 1 (DINOv3 blue / CLIP orange / joint purple) plus distinct marker shapes.
Also prints the statistics quoted in the text, over every attribute and all four runs.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, "/media/tan/F/RWS2027/Upar-Challenge-Track-2/par")
from upar.data import attr_names, load_split  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE.parent / "figs"
LOG = Path("/media/tan/F/RWS2027/remote_results/fusion/logits16b")
for f in ("TimesNewRoman-Regular.ttf", "TimesNewRoman-Italic.ttf"):
    fm.fontManager.addfont(str(Path.home() / ".local/share/fonts" / f))
HP, CL, JT = "#1F5BD8", "#D2731A", "#7A3FC8"
INK, MUTE, BAR = "#18212D", "#5B6675", "#C9D0D8"
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 7.5, "pdf.fonttype": 42, "mathtext.fontset": "stix",
                     "text.color": INK, "axes.edgecolor": "#2B3A4E", "xtick.color": "#2B3A4E", "ytick.color": INK,
                     "axes.linewidth": .6, "xtick.major.width": .6, "xtick.major.size": 2.5, "xtick.labelsize": 7})
A = attr_names(); va = load_split("val"); GT = va[A].values; DS = va.ds.values
COLORS = set(range(8, 20)) | set(range(21, 33))          # upper-body and lower-body color attributes
FOLDS = {"PETA": "PETA", "Market": "Market1501"}
MIN_GAP, MIN_POS = 2.0, 50


def short(a):
    for old, new in [("UpperBody-Length-Short", "short sleeves"), ("LowerBody-Length-Short", "short lower body"),
                     ("LowerBody-Type-Trousers&Shorts", "trousers/shorts"), ("LowerBody-Type-Skirt&Dress", "skirt/dress"),
                     ("UpperBody-Color-", "upper "), ("LowerBody-Color-", "lower "), ("Hair-Length-", "hair "),
                     ("Accessory-Glasses-Normal", "glasses"), ("Accessory-Glasses-Sun", "sunglasses"), ("Accessory-", ""),
                     ("Gender-", ""), ("Age-", "")]:
        a = a.replace(old, new)
    return a.lower()


def bacc(y, z):
    v = y >= 0; y, d = y[v], z[v] > 0
    return 100 * .5 * (d[y == 1].mean() + (~d[y == 0]).mean()) if 0 < y.sum() < len(y) else np.nan


# per fold and seed: balanced accuracy of the three query heads for every attribute
acc, npos = {}, {}
for fold, hold in FOLDS.items():
    for s in (1, 2):
        d = np.load(LOG / f"DTB_{fold}_s{s}.npz"); m = DS[d["idx"]] == hold; y = GT[d["idx"][m]]
        acc[fold, s] = {h: np.array([bacc(y[:, k], d[f"{h}_w1.00"][m][:, k]) for k in range(40)]) for h in "hcj"}
        npos[fold] = (y == 1).sum(0)

# statistics quoted in the text: every (attribute, run) with labels, both folds, both seeds
h = np.concatenate([acc[k]["h"] for k in acc]); c = np.concatenate([acc[k]["c"] for k in acc])
j = np.concatenate([acc[k]["j"] for k in acc]); kk = np.tile(np.arange(40), len(acc))
ok = np.isfinite(j); h, c, j, kk = h[ok], c[ok], j[ok], kk[ok]
better, mean = np.maximum(h, c), (h + c) / 2
col = np.array([k in COLORS for k in kk])
print(f"all {len(j)} (attribute, run) pairs: joint > mean of the two {100 * np.mean(j > mean):.1f}%, "
      f">= better {100 * np.mean(j >= better):.1f}%")
for name, sel in (("clothing colors", col), ("other attributes", ~col)):
    print(f"  {name:16s} n={sel.sum():3d}: > mean {100 * np.mean((j > mean)[sel]):.1f}%, "
          f">= better {100 * np.mean((j >= better)[sel]):.1f}%")

# rows of the figure: two-seed means, large gap, enough positives
rows = []
for fold in FOLDS:
    H, C, J = (np.nanmean([acc[fold, s][x] for s in (1, 2)], 0) for x in "hcj")
    for k in range(40):
        if np.isfinite(J[k]) and abs(H[k] - C[k]) >= MIN_GAP and npos[fold][k] >= MIN_POS:
            rows.append(dict(k=k, fold=fold, h=H[k], c=C[k], j=J[k], color=k in COLORS))
blocks = [("other attributes", [r for r in rows if not r["color"]]), ("clothing colors", [r for r in rows if r["color"]])]
for name, b in blocks:
    b.sort(key=lambda r: r["h"] - r["c"])
    near = sum(r["j"] >= max(r["h"], r["c"]) - .25 for r in b)
    print(f"figure block {name}: {len(b)} rows, joint within 0.25 pt of or above the better head on {near}")

# ---------------------------------------------------------------- figure
n_rows = sum(len(b) for _, b in blocks)
PITCH, HEAD = .106, .15                                    # inches per row, per block title
W = 3.35
Hgt = .30 + n_rows * PITCH + len(blocks) * HEAD + .30
fig = plt.figure(figsize=(W, Hgt))
L, R = 1.14, .08                                           # label column and right margin (inches)
ax = fig.add_axes([L / W, .30 / Hgt, (W - L - R) / W, (Hgt - .30 - .30) / Hgt])
y, ticks, labels, titles = 0.0, [], [], []
for name, b in blocks:
    titles.append((y - .05, name)); y += HEAD / PITCH
    for r in b:
        lo, hi = sorted((r["h"], r["c"]))
        ax.plot([lo, hi], [y, y], color=BAR, lw=2.2, solid_capstyle="round", zorder=1)
        ax.scatter([r["h"]], [y], s=15, marker="o", color=HP, lw=0, zorder=3)
        ax.scatter([r["c"]], [y], s=14, marker="s", color=CL, lw=0, zorder=3)
        ax.scatter([r["j"]], [y], s=30, marker="D", fc="none", ec=JT, lw=1.1, zorder=4)   # open: a head under it shows
        ticks.append(y); labels.append(f"{short(A[r['k']])} ({'PETA' if r['fold'] == 'PETA' else 'Market'})")
        y += 1
ax.set_ylim(y - .4, -.6)
ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=7)
ax.tick_params(axis="y", length=0, pad=3)
vals = np.array([[r["h"], r["c"], r["j"]] for _, b in blocks for r in b])
x0, x1 = np.floor(vals.min() / 5) * 5, np.ceil(vals.max() / 5) * 5
ax.set_xlim(x0 - .5, x1 + .5); ax.set_xticks(np.arange(x0, x1 + .1, 5))
ax.grid(axis="x", color="#E3E7EC", lw=.5, zorder=0); ax.set_axisbelow(True)
for s_ in ("top", "right", "left"):
    ax.spines[s_].set_visible(False)
ax.set_xlabel("balanced accuracy on the held-out dataset (%)", fontsize=7.2, labelpad=2)
for yt, name in titles:
    ax.text(-(L - .03) / (W - L - R), yt + .35, name, transform=ax.get_yaxis_transform(), ha="left", va="center",
            fontsize=7.2, fontstyle="italic", color=MUTE)
fig.legend(handles=[Line2D([], [], ls="", marker="o", ms=4.0, mfc=HP, mec=HP, label="DINOv3 query head"),
                    Line2D([], [], ls="", marker="s", ms=3.6, mfc=CL, mec=CL, label="CLIP query head"),
                    Line2D([], [], ls="", marker="D", ms=4.6, mfc="none", mec=JT, mew=1.1, label="joint query head")],
           loc="upper center", bbox_to_anchor=(.5, 1.0), ncol=3, fontsize=7, frameon=False, handletextpad=.15,
           columnspacing=1.0, borderaxespad=.2)
OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "pick.pdf"); fig.savefig(HERE / "pick_preview.png", dpi=240)
print("saved", OUT / "pick.pdf", f"{W:.2f} x {Hgt:.2f} in, {n_rows} rows")
