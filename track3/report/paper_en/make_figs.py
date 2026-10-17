"""Figures of the English Track 3 paper. Run from report/paper_en/:
../../../Upar-Challenge-Track-2/.venv/bin/python make_figs.py
Reuses data helpers of ../paper/make_figs.py (leave-one-dataset-out features in runs/lodo)."""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).parent
import fig_helpers as V  # noqa: E402  (data helpers and logged constants)

FIG = HERE / "figs"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e3e2db", "grid.linewidth": .5, "axes.edgecolor": "#9a988f",
                     "axes.axisbelow": True, "savefig.dpi": 300, "savefig.bbox": "tight", "axes.titlesize": 7.5,
                     "legend.fontsize": 6.5, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8})
INK, MUTED, BLUE, ORANGE, AQUA, GRAY = "#121211", "#85837c", "#2a78d6", "#eb6834", "#1baf7a", "#b4b2a9"
LAB = {"peta": "PETA", "pa100k": "PA100k", "market": "Market1501"}
JOINTS = ["nose", "head bottom", "head top", "l. ear", "r. ear", "l. shoulder", "r. shoulder", "l. elbow", "r. elbow",
          "l. wrist", "r. wrist", "l. hip", "r. hip", "l. knee", "r. knee", "l. ankle", "r. ankle"]


def _draw(ax, img, g, p):
    ax.imshow(img); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for kp, c, lw in ((g, AQUA, 1.4), (p, ORANGE, 1.0)):
        for u, v in V.EDGES:
            if kp is g and (g[u, 2] == 0 or g[v, 2] == 0):
                continue
            ax.plot([kp[u, 0], kp[v, 0]], [kp[u, 1], kp[v, 1]], color=c, lw=lw, alpha=.9)
    ax.set_xlim(0, img.width); ax.set_ylim(img.height, 0)


def fig_examples():
    z = V.load("peta"); d = json.load(open(V.ANN / "train_peta.json"))
    ims = {im["id"]: im for im in d["images"]}; anns = {a["id"]: a for a in d["annotations"]}
    order = np.argsort(-z["oks"]); n = len(order)
    picks = [order[int(n * q)] for q in (0.02, 0.2, 0.5, 0.8, 0.95, 0.985, 0.995, 0.999)]
    fig, axs = plt.subplots(1, 8, figsize=(6.9, 2.0))
    for ax, i in zip(axs, picks):
        a = anns[int(z["ann_id"][i])]; img = Image.open(V.PETA_IMG / Path(ims[a["image_id"]]["file_name"]).name).convert("RGB")
        _draw(ax, img, np.array(a["keypoints"], float).reshape(17, 3), z["keypoints"][i].reshape(17, 3))
        ax.set_title(f"OKS {z['oks'][i]:.2f}\nscore {z['X'][i, 70]:.2f}", fontsize=6.5)
    fig.savefig(FIG / "examples.pdf"); plt.close(fig)


def fig_residuals():
    data, resc = V.lodo_scores(); z = data["peta"]; s = resc["peta"]
    d = json.load(open(V.ANN / "train_peta.json")); ims = {im["id"]: im for im in d["images"]}; anns = {a["id"]: a for a in d["annotations"]}
    rank = s.argsort().argsort() / len(s); cand = np.nonzero((z["oks"] < .5) & (rank > .5))[0]
    pick = cand[np.argsort(-rank[cand])][:6]
    fig, axs = plt.subplots(1, 6, figsize=(6.9, 2.1))
    for ax, i in zip(axs, pick):
        a = anns[int(z["ann_id"][i])]; img = Image.open(V.PETA_IMG / Path(ims[a["image_id"]]["file_name"]).name).convert("RGB")
        _draw(ax, img, np.array(a["keypoints"], float).reshape(17, 3), z["keypoints"][i].reshape(17, 3))
        ax.set_title(f"OKS {z['oks'][i]:.2f}, top {100 * (1 - rank[i]):.0f}%", fontsize=6.5)
    fig.savefig(FIG / "residuals.pdf"); plt.close(fig)


def fig_noise():
    from scipy.stats import spearmanr
    data, resc = V.lodo_scores(); rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(3.35, 2.0))
    for n, c in zip(V.NAMES, (BLUE, ORANGE, AQUA)):
        z = data[n]; o = z["oks"]; xs, ys = [], []
        for sd in (0, .01, .02, .04, .06, .1, .2, .5, 5):
            s = o + rng.normal(0, sd, len(o)); xs.append(spearmanr(s, o).correlation); ys.append(V.ap(n, z, s))
        ax.plot(xs, ys, color=c, lw=1.2, marker=".", ms=3, label=f"{LAB[n]} held out")
        for s, mk in ((z["X"][:, 70], "o"), (resc[n], "s")):
            ax.scatter(spearmanr(s, o).correlation, V.ap(n, z, s), color=c, marker=mk, s=18, edgecolor=INK, lw=.5, zorder=4)
    ax.scatter([], [], marker="o", color="white", edgecolor=INK, label="default score")
    ax.scatter([], [], marker="s", color="white", edgecolor=INK, label="learned score")
    ax.set_xlabel("Spearman correlation of score with OKS"); ax.set_ylabel("OKS AP"); ax.legend(frameon=False, fontsize=5.8)
    fig.tight_layout(); fig.savefig(FIG / "noise.pdf"); plt.close(fig)


def fig_thresholds():
    """AP gain over the default score per OKS threshold, one panel per held-out dataset: learned score, the default
    score without the two never-annotated ear slots, and the oracle (clipped, value printed)."""
    data, resc = V.lodo_scores(); thr = np.arange(.5, .96, .05); keep = [k for k in range(17) if k not in (3, 4)]
    fig, axs = plt.subplots(1, 3, figsize=(6.9, 1.75), sharey=True); top = 2.4
    for ax, n in zip(axs, V.NAMES):
        z = data[n]; base = V._ap_per_threshold(n, z, z["X"][:, 70])
        g = {k: (V._ap_per_threshold(n, z, sc) - base) * 100 for k, sc in
             (("oracle", z["oks"]), ("no_ears", z["X"][:, :17][:, keep].mean(1)), ("learned", resc[n]))}
        ax.bar(thr, np.minimum(g["oracle"], top), width=.04, color=GRAY, label="oracle (true OKS)")
        for t, v in zip(thr, g["oracle"]):
            if v > top:
                ax.text(t, top + .05, f"{v:.1f}", ha="center", fontsize=5.8, color=MUTED)
        ax.bar(thr - .009, g["no_ears"], width=.017, color=ORANGE, label="default without ears")
        ax.bar(thr + .009, g["learned"], width=.017, color=BLUE, label="learned score")
        ax.axhline(0, color=MUTED, lw=.5); ax.set_ylim(-.3, top + .4)
        ax.set_xticks(thr[::2], [f"{t:.2f}" for t in thr[::2]]); ax.set_title(f"{LAB[n]} held out", loc="left")
        ax.set_xlabel("OKS threshold"); ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("$\\Delta$AP vs. default (pts)"); axs[0].legend(frameon=False, loc="upper left", fontsize=5.8)
    fig.tight_layout(w_pad=.5); fig.savefig(FIG / "per_threshold.pdf"); plt.close(fig)


def fig_joints():
    keep = [k for k in range(17) if k not in (3, 4)]
    fig, ax = plt.subplots(figsize=(3.35, 1.9)); w = .27
    for r, (n, c) in enumerate(zip(V.NAMES, (BLUE, ORANGE, AQUA))):
        z = V.load(n); anns = V._gt(n); ks = np.zeros(17); cnt = np.zeros(17)
        for aid, kp in zip(z["ann_id"], z["keypoints"]):
            a = anns[int(aid)]; g = np.array(a["keypoints"], float).reshape(17, 3); p = kp.reshape(17, 3); v = g[:, 2] > 0
            e = ((p[:, 0] - g[:, 0]) ** 2 + (p[:, 1] - g[:, 1]) ** 2) / (2 * V.SIGMAS) ** 2 / (a["area"] + 1e-9) / 2
            ks[v] += np.exp(-e[v]); cnt[v] += 1
        sim = ks / np.maximum(cnt, 1)
        ax.bar(np.arange(len(keep)) + (r - 1) * w, [1 - sim[k] for k in keep], w, color=c, label=f"{LAB[n]} held out")
    ax.set_xticks(range(len(keep)), [JOINTS[k] for k in keep], rotation=50, ha="right", fontsize=6)
    ax.set_ylabel("1 $-$ keypoint similarity"); ax.legend(frameon=False); ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(FIG / "joints.pdf"); plt.close(fig)


def fig_importance():
    from scipy.stats import spearmanr
    data = {n: V.load(n) for n in V.NAMES}
    fams = {"heatmap maxima": range(0, 17), "flip disagreement": range(17, 34), "mass / peak": range(34, 51),
            "spatial spread": range(51, 68), "box size": range(68, 70)}
    drops = {f: [] for f in fams}; rng = np.random.default_rng(0)
    for held in V.NAMES:
        tr = [data[n] for n in V.NAMES if n != held]
        m = V.gbm().fit(V.features(np.concatenate([t["X"] for t in tr])), np.concatenate([t["oks"] for t in tr]))
        F = V.features(data[held]["X"]); y = data[held]["oks"]; base = spearmanr(m.predict(F), y).correlation
        for f, ix in fams.items():
            G = F.copy(); perm = rng.permutation(len(G)); G[:, list(ix)] = G[perm][:, list(ix)]
            drops[f].append(base - spearmanr(m.predict(G), y).correlation)
    keys = sorted(fams, key=lambda f: -np.mean(drops[f]))
    fig, ax = plt.subplots(figsize=(3.35, 1.5))
    ax.barh(range(5)[::-1], [np.mean(drops[k]) for k in keys], color=BLUE, height=.6,
            xerr=[np.std(drops[k]) for k in keys], error_kw={"elinewidth": .6, "ecolor": MUTED})
    ax.set_yticks(range(5)[::-1], keys); ax.set_xlabel("drop in Spearman $\\rho$ when permuted"); ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(FIG / "importance.pdf"); plt.close(fig)


if __name__ == "__main__":
    for f in sys.argv[1:] or ["fig_examples", "fig_residuals", "fig_noise", "fig_thresholds", "fig_joints", "fig_importance"]:
        globals()[f]()
