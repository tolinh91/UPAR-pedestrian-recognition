"""Figures text_idea and text_auc of the v31 paper: the CLIP text score before training, from real zero-shot scores.

python make_zs_fig.py      (reads analysis/zs_auc.json and remote_results/zs_text/zs_scores.npz from zs_viz.py,
                            and two PETA val crops; writes figs/text_idea.pdf for the Method, figs/text_auc.pdf for
                            the Experiments)
text_idea: the idea on the hat attribute, PETA val: every image is a point, x = 100 <e, t(a person)>, y = 100 <e, t(hat sentence)>;
    the text score is y - x, the vertical distance to the dashed line y = x (where CLIP is undecided between the two
    sentences). Two real crops: a no-hat image with a higher raw similarity to the hat sentence than a hat image, but
    below the line, and that hat image above it.
text_auc: zero-shot AUC of every attribute that has positives, attribute sentence alone (x) vs minus 'a person' (y), PETA
    (circles) and Market-1501 (triangles); all points shown, points more than 0.05 below the line are named.
Colors: hat CLIP orange triangles / no hat grey dots; datasets Okabe-Ito green / reddish purple plus marker shape.
Photos are resampled once with Lanczos; everything else is vector except the 6963-point scatter (rasterized).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox, OffsetImage  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).parent
ROOT = HERE.parents[3]
OUT = HERE.parent / "figs"
DATA = ROOT / "Upar-Challenge-Track-2" / "UPAR-Challenge-2027" / "data"
for f in ("TimesNewRoman-Regular.ttf", "TimesNewRoman-Italic.ttf"):
    fm.fontManager.addfont(str(Path.home() / ".local/share/fonts" / f))
INK, MUTE, CL, PETA_C, MKT_C = "#18212D", "#626E7D", "#D2731A", "#009E73", "#CC79A7"
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 7.5, "pdf.fonttype": 42, "mathtext.fontset": "stix",
                     "text.color": INK, "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "axes.linewidth": .6, "xtick.major.width": .6, "ytick.major.width": .6,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.labelsize": 7, "ytick.labelsize": 7})
J = json.load(open(HERE / "zs_auc.json"))
Z = np.load(ROOT / "remote_results" / "zs_text" / "zs_scores.npz")
A = J["attributes"]
SHORT = {"Age-Adult": "adult", "Hair-Length-Short": "short hair"}
NO_HAT, HAT = 4268, 2972            # PETA val rows: raw similarity to the hat sentence 27.7 vs 24.3, score -1.0 vs +2.8

OUT.mkdir(exist_ok=True)
# ---------------- figure 1: the idea on the hat attribute (Method)
fig = plt.figure(figsize=(2.2, 2.15))
k = A.index("Accessory-Hat")
x, y, lab, im = Z["PETA_s_neu"], Z["PETA_s_raw"][:, k], Z["PETA_y"][:, k], Z["PETA_image"]
ax = fig.add_axes([.165, .155, .74 * 2.15 / 2.2, .74])
lo, hi = 13.0, 33.0
ax.plot([lo, hi], [lo, hi], ls=(0, (3, 2)), lw=.6, color=MUTE, zorder=1)
ax.scatter(x[lab == 0], y[lab == 0], s=1.2, marker="o", color=MUTE, alpha=.35, lw=0, rasterized=True, zorder=2)
ax.scatter(x[lab == 1], y[lab == 1], s=3.0, marker="^", color=CL, alpha=.8, lw=0, rasterized=True, zorder=3)
for i, xy_box, col in ((NO_HAT, (31.0, 17.2), MUTE), (HAT, (15.6, 29.0), CL)):
    img = Image.open(DATA / str(im[i])).convert("RGB").resize((48 * 3, 128 * 3), Image.LANCZOS)
    ab = AnnotationBbox(OffsetImage(np.asarray(img), zoom=.075), (x[i], y[i]), xybox=xy_box, xycoords="data",
                        boxcoords="data", pad=.12, frameon=True, zorder=5,
                        bboxprops=dict(edgecolor=col, lw=.9),
                        arrowprops=dict(arrowstyle="-", color=col, lw=.6, shrinkA=0, shrinkB=1.5))
    ax.add_artist(ab)
    ax.scatter([x[i]], [y[i]], s=14, marker="^" if lab[i] else "o", facecolor="none", edgecolor=col, lw=.9, zorder=6)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xticks([15, 20, 25, 30]); ax.set_yticks([15, 20, 25, 30])
ax.set_xlabel(r"similarity to $\mathtt{a\ person}$")
ax.set_ylabel("similarity to the hat sentence")
fig.legend(handles=[Line2D([], [], ls="", marker="^", ms=3.4, mfc=CL, mec=CL, label="with a hat"),
                    Line2D([], [], ls="", marker="o", ms=2.6, mfc=MUTE, mec=MUTE, alpha=.6, label="without a hat")],
           loc="upper center", bbox_to_anchor=(.165 + .37 * 2.15 / 2.2, 1.0), ncol=2, fontsize=6.5, frameon=False,
           handletextpad=.1, columnspacing=1.0, borderaxespad=.1)
fig.savefig(OUT / "text_idea.pdf", dpi=600)
fig.savefig(HERE / "text_idea_preview.png", dpi=240)

# ---------------- figure 2: evidence on every attribute (Experiments)
fig = plt.figure(figsize=(2.4, 2.05))
bx = fig.add_axes([.19, .17, .78 * 2.05 / 2.4, .78])
pts = []
for name, c, m in (("PETA", PETA_C, "o"), ("Market1501", MKT_C, "^")):
    at = J[name]["attrs"]
    u = np.array([v["auc_raw"] for v in at.values()]); w = np.array([v["auc_sub"] for v in at.values()])
    pts += [u, w]
    bx.scatter(u, w, s=9, marker=m, facecolor=c, edgecolor="white", linewidth=.3, zorder=3)
    for a, v in at.items():
        if v["auc_sub"] - v["auc_raw"] < -.05:
            bx.annotate(SHORT.get(a, a), (v["auc_raw"], v["auc_sub"]), xytext=(3.5, -1), textcoords="offset points",
                        fontsize=6.3, color=MUTE, va="center")
v = np.concatenate(pts)
blo = np.floor((v.min() - .01) * 10) / 10
bx.plot([blo, 1], [blo, 1], ls=(0, (3, 2)), lw=.6, color=MUTE, zorder=1)
bx.set_xlim(blo, 1); bx.set_ylim(blo, 1)
bx.set_xticks(np.arange(np.ceil(blo * 5) / 5, 1.01, .2)); bx.set_yticks(np.arange(np.ceil(blo * 5) / 5, 1.01, .2))
bx.set_xlabel("AUC, attribute sentence alone")
bx.set_ylabel(r"AUC, minus $\mathtt{a\ person}$")
bx.legend(handles=[Line2D([], [], ls="", marker="o", ms=3.3, mfc=PETA_C, mec="white", mew=.3, label="PETA"),
                   Line2D([], [], ls="", marker="^", ms=3.6, mfc=MKT_C, mec="white", mew=.3, label="Market-1501")],
          loc="upper left", fontsize=6.5, frameon=False, handletextpad=.1, borderaxespad=.15, labelspacing=.2)
fig.savefig(OUT / "text_auc.pdf")
fig.savefig(HERE / "text_auc_preview.png", dpi=240)
print("saved", OUT / "text_idea.pdf", OUT / "text_auc.pdf")
