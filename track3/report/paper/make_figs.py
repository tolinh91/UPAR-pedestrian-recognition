"""Figures of the Track 3 paper (track3_paper.tex). Run from report/paper/ with the Track 2 venv (numpy, sklearn,
pycocotools, matplotlib): ../../../Upar-Challenge-Track-2/.venv/bin/python make_figs.py [fig_name ...]
Data: leave-one-dataset-out features of the ViTPose+ huge models (runs/lodo/feats_{peta,pa100k,market}.npz, written by
t3/rescore_feats.py), the PETA images of the Track 1 data folder, and logged AP values (constants below)."""
import contextlib
import copy
import io
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "t3"))
FIG = HERE / "figs"
ANN = ROOT / "starter/data/annotations/specialization"
LODO = ROOT / "runs/lodo"
PETA_IMG = ROOT.parent / "Upar-Challenge-Track-1/UPAR-Challenge-2027/data/PETA/images"
SIGMAS = np.array([.026, .025, .025, .035, .035, .079, .079, .072, .072, .062, .062, .107, .107, .087, .087, .089, .089])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e3e2db", "grid.linewidth": .6, "axes.edgecolor": "#9a988f",
                     "axes.axisbelow": True, "savefig.dpi": 220, "savefig.bbox": "tight"})
INK, MUTED, BLUE, ORANGE, AQUA, YELLOW, GRAY, RED = "#121211", "#85837c", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#b4b2a9", "#e34948"
NAMES = ("peta", "pa100k", "market")
LABEL = {"peta": "PETA", "pa100k": "PA100k", "market": "Market1501"}
EDGES = [(2, 1), (1, 5), (1, 6), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
         (12, 14), (14, 16), (0, 1)]


def load(name):
    z = dict(np.load(LODO / f"feats_{name}.npz"))
    z["oks"] = np.nan_to_num(z["oks"]).astype(np.float32)
    return z


def ap(name, z, score):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(ANN / f"train_{name}.json"))
        preds = [{"image_id": int(i), "category_id": 1, "score": float(s), "keypoints": k.tolist()}
                 for i, s, k in zip(z["image_id"], score, z["keypoints"])]
        ev = COCOeval(gt, gt.loadRes(copy.deepcopy(preds)), "keypoints")
        ev.params.kpt_oks_sigmas = SIGMAS
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return ev.stats[0]


def features(X):
    F = X[:, :70].astype(np.float64).copy()
    F[:, 17:34] = np.log1p(F[:, 17:34]); F[:, 51:68] = np.log1p(F[:, 51:68])
    return F


def gbm():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_leaf_nodes=7, min_samples_leaf=100,
                                         early_stopping=False, random_state=0)


def lodo_scores():
    """Rescored instance scores for every dataset, from a model fit on the other two."""
    data = {n: load(n) for n in NAMES}
    out = {}
    for held in NAMES:
        tr = [data[n] for n in NAMES if n != held]
        m = gbm().fit(features(np.concatenate([t["X"] for t in tr])), np.concatenate([t["oks"] for t in tr]))
        out[held] = m.predict(features(data[held]["X"]))
    return data, out


def fig_examples():
    """PETA crops: ground truth (green) and leave-PETA-out huge prediction (orange), from high to low OKS."""
    z = load("peta")
    d = json.load(open(ANN / "train_peta.json"))
    ims = {im["id"]: im for im in d["images"]}
    anns = {a["id"]: a for a in d["annotations"]}
    order = np.argsort(-z["oks"])
    n = len(order)
    picks = [order[int(n * q)] for q in (0.02, 0.2, 0.5, 0.8, 0.95, 0.985, 0.995, 0.999)]
    fig, axs = plt.subplots(1, 8, figsize=(7.4, 2.9))
    for ax, i in zip(axs, picks):
        a = anns[int(z["ann_id"][i])]; im = ims[a["image_id"]]
        img = Image.open(PETA_IMG / Path(im["file_name"]).name).convert("RGB")
        ax.imshow(img); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        g = np.array(a["keypoints"], float).reshape(17, 3)
        p = z["keypoints"][i].reshape(17, 3)
        for kp, c, lw in ((g, AQUA, 1.6), (p, ORANGE, 1.2)):
            for u, v in EDGES:
                if kp is g and (g[u, 2] == 0 or g[v, 2] == 0):
                    continue
                ax.plot([kp[u, 0], kp[v, 0]], [kp[u, 1], kp[v, 1]], color=c, lw=lw, alpha=.9)
        ax.set_title(f"OKS {z['oks'][i]:.2f}\nđiểm cũ {z['X'][i, 70]:.2f}", fontsize=7)
        ax.set_xlim(0, img.width); ax.set_ylim(img.height, 0)
    fig.savefig(FIG / "examples.png"); plt.close(fig)


def fig_oks_hist():
    fig, ax = plt.subplots(figsize=(4.8, 2.4))
    bins = np.linspace(0.5, 1, 26)
    for n, c in zip(NAMES, (BLUE, ORANGE, AQUA)):
        o = load(n)["oks"]
        ax.hist(np.clip(o, 0.5, 1), bins=bins, histtype="step", lw=1.6, color=c, density=True,
                label=f"{LABEL[n]} (trung vị {np.median(o):.3f}, OKS<0.75: {np.mean(o < .75):.1%})")
    ax.set_xlabel("OKS của từng người (mô hình huge giấu bộ dữ liệu đó)"); ax.set_ylabel("mật độ")
    ax.legend(frameon=False, fontsize=6.8, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "oks_hist.png"); plt.close(fig)


def fig_score_choices():
    """AP on each held-out dataset with five instance-score rules (same keypoints)."""
    data, resc = lodo_scores()
    rng = np.random.default_rng(0)
    rules = [("hằng số", lambda z, n: np.ones(len(z["oks"]))), ("ngẫu nhiên", lambda z, n: rng.random(len(z["oks"]))),
             ("trung bình 17 cực đại", lambda z, n: z["X"][:, 70]), ("bộ chấm học được", lambda z, n: resc[n]),
             ("OKS thật (oracle)", lambda z, n: z["oks"])]
    cols = [GRAY, MUTED, ORANGE, BLUE, AQUA]
    res = {r: [ap(n, data[n], f(data[n], n)) for n in NAMES] for r, f in rules}
    for r, v in res.items():
        print(f"{r:24s}", " ".join(f"{x:.4f}" for x in v), f"mean {np.mean(v):.4f}")
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    w = .16
    for k, ((r, _), c) in enumerate(zip(rules, cols)):
        v = res[r]
        ax.bar(np.arange(3) + (k - 2) * w, np.array(v) - .89, bottom=.89, width=w, color=c, label=r, edgecolor="white", lw=.8)
    ax.set_xticks(range(3), [f"giấu {LABEL[n]}" for n in NAMES]); ax.set_ylim(.89, .958)
    ax.set_ylabel("OKS AP"); ax.legend(frameon=False, fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(.5, 1.18))
    ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(FIG / "score_choices.png"); plt.close(fig)
    json.dump(res, open(HERE / "score_choices.json", "w"), indent=1)


def fig_scatter():
    from scipy.stats import spearmanr
    data, resc = lodo_scores()
    z = data["peta"]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, s, lab in ((axs[0], z["X"][:, 70], "trung bình 17 cực đại heatmap"), (axs[1], resc["peta"], "bộ chấm học được (không thấy PETA)")):
        ax.scatter(s, z["oks"], s=2, alpha=.25, color=BLUE, lw=0)
        rho = spearmanr(s, z["oks"]).correlation
        ax.set_title(f"Spearman ρ = {rho:.3f}", fontsize=8.4, loc="left"); ax.set_xlabel(lab)
        print(lab, "spearman", round(float(rho), 3))
    axs[0].set_ylabel("OKS thật"); axs[0].set_ylim(0.3, 1.01)
    fig.tight_layout(); fig.savefig(FIG / "scatter.png"); plt.close(fig)


def fig_importance():
    """Permutation importance per feature family: drop of Spearman(score, OKS) on the held-out dataset, LODO mean."""
    from scipy.stats import spearmanr
    data = {n: load(n) for n in NAMES}
    fams = {"cực đại heatmap": range(0, 17), "lệch thường/lật": range(17, 34), "khối/đỉnh": range(34, 51),
            "độ loe heatmap": range(51, 68), "kích thước khung": range(68, 70)}
    drops = {f: [] for f in fams}
    rng = np.random.default_rng(0)
    for held in NAMES:
        tr = [data[n] for n in NAMES if n != held]
        m = gbm().fit(features(np.concatenate([t["X"] for t in tr])), np.concatenate([t["oks"] for t in tr]))
        F = features(data[held]["X"]); y = data[held]["oks"]
        base = spearmanr(m.predict(F), y).correlation
        for f, ix in fams.items():
            G = F.copy(); perm = rng.permutation(len(G)); G[:, list(ix)] = G[perm][:, list(ix)]
            drops[f].append(base - spearmanr(m.predict(G), y).correlation)
    fig, ax = plt.subplots(figsize=(4.8, 2.3))
    keys = sorted(fams, key=lambda f: -np.mean(drops[f]))
    ax.barh(range(len(keys))[::-1], [np.mean(drops[k]) for k in keys], color=BLUE, height=.6,
            xerr=[np.std(drops[k]) for k in keys], error_kw={"elinewidth": .8, "ecolor": MUTED})
    ax.set_yticks(range(len(keys))[::-1], keys); ax.set_xlabel("giảm Spearman ρ khi hoán vị nhóm đặc trưng")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(FIG / "importance.png"); plt.close(fig)
    print({k: [round(float(x), 4) for x in drops[k]] for k in keys})


LB = [("mẫu", "ViTPose+ S\nkhông tinh chỉnh", .6238), ("v1", "ViTPose+ B\ntinh chỉnh", .8813), ("v2", "ViTPose+ L", .8939),
      ("v4", "ViTPose+ H", .8958), ("v3", "L + H\ngộp heatmap", .8977), ("v6", "L + H\n+ bộ chấm điểm", .8992)]


def fig_leaderboard():
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1, 2.2]})
    axs[0].bar(range(2), [LB[0][2], LB[1][2]], color=[GRAY, BLUE], width=.55)
    for i in range(2):
        axs[0].text(i, LB[i][2] + .01, f"{LB[i][2]:.4f}", ha="center", fontsize=7.2)
    axs[0].set_xticks(range(2), [f"{t}\n{l}" for t, l, _ in LB[:2]], fontsize=6.8); axs[0].set_ylim(0, 1)
    axs[0].set_ylabel("OKS AP (tập chấm ẩn)"); axs[0].grid(axis="x", visible=False)
    v = [x[2] for x in LB[1:]]
    axs[1].plot(range(len(v)), v, color=BLUE, lw=1.4, marker="o", ms=5)
    for i, x in enumerate(v):
        axs[1].text(i, x + .0008, f"{x:.4f}", ha="center", fontsize=7.2)
    axs[1].set_xticks(range(len(v)), [f"{t}\n{l}" for t, l, _ in LB[1:]], fontsize=6.8)
    axs[1].set_ylim(.879, .902); axs[1].grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(FIG / "leaderboard.png"); plt.close(fig)


def fig_lodo():
    """Held-out AP: zero-shot huge, fine-tuned large / huge / large+huge, and the ensemble with the rescorer."""
    vals = {"huge không tinh chỉnh": [.7025, .7018, .6995], "large tinh chỉnh": [.9314, .9244, .9392],
            "huge tinh chỉnh": [.9304, .9281, .9396], "large + huge": [.9352, .9291, .9416],
            "large + huge + bộ chấm": [.9377, .9297, .9431]}
    cols = [GRAY, MUTED, ORANGE, AQUA, BLUE]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1, 2.4]})
    axs[0].bar(range(2), [np.mean(vals["huge không tinh chỉnh"]), np.mean(vals["huge tinh chỉnh"])], color=[GRAY, ORANGE], width=.55)
    axs[0].set_xticks(range(2), ["huge\nkhông tinh chỉnh", "huge\ntinh chỉnh"], fontsize=7); axs[0].set_ylim(0, 1)
    axs[0].set_ylabel("OKS AP, trung bình 3 miền giấu"); axs[0].grid(axis="x", visible=False)
    for i, x in enumerate([np.mean(vals["huge không tinh chỉnh"]), np.mean(vals["huge tinh chỉnh"])]):
        axs[0].text(i, x + .015, f"{x:.3f}", ha="center", fontsize=7.2)
    w = .19
    for k, (lab, c) in enumerate(zip(list(vals)[1:], cols[1:])):
        v = np.array(vals[lab])
        axs[1].bar(np.arange(3) + (k - 1.5) * w, v - .92, bottom=.92, width=w, color=c, label=lab, edgecolor="white", lw=.8)
    axs[1].set_xticks(range(3), [f"giấu {LABEL[n]}" for n in NAMES]); axs[1].set_ylim(.92, .946)
    axs[1].legend(frameon=False, fontsize=6.8, ncol=2, loc="upper left"); axs[1].grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(FIG / "lodo.png"); plt.close(fig)



JOINTS = ["mũi", "đáy đầu", "đỉnh đầu", "tai trái", "tai phải", "vai trái", "vai phải", "khuỷu trái", "khuỷu phải",
          "cổ tay trái", "cổ tay phải", "hông trái", "hông phải", "gối trái", "gối phải", "cổ chân trái", "cổ chân phải"]


def _gt(name):
    d = json.load(open(ANN / f"train_{name}.json"))
    return {a["id"]: a for a in d["annotations"]}


def fig_joints():
    """Per-joint keypoint similarity exp(-d^2 / 2 s^2 k^2) on held-out data (huge), mean over labelled joints."""
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    w = .26
    keep = [k for k in range(17) if k not in (3, 4)]
    for r, (n, c) in enumerate(zip(NAMES, (BLUE, ORANGE, AQUA))):
        z = load(n); anns = _gt(n)
        ks = np.zeros(17); cnt = np.zeros(17)
        for aid, kp in zip(z["ann_id"], z["keypoints"]):
            a = anns[int(aid)]; g = np.array(a["keypoints"], float).reshape(17, 3); p = kp.reshape(17, 3)
            v = g[:, 2] > 0
            e = ((p[:, 0] - g[:, 0]) ** 2 + (p[:, 1] - g[:, 1]) ** 2) / (2 * SIGMAS) ** 2 / (a["area"] + 1e-9) / 2
            ks[v] += np.exp(-e[v]); cnt[v] += 1
        sim = ks / np.maximum(cnt, 1)
        ax.bar(np.arange(len(keep)) + (r - 1) * w, [1 - sim[k] for k in keep], w, color=c, label=f"giấu {LABEL[n]}")
        print(n, {JOINTS[k]: round(float(1 - sim[k]), 3) for k in keep})
    ax.set_xticks(range(len(keep)), [JOINTS[k] for k in keep], rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("1 − độ tương đồng khớp"); ax.legend(frameon=False, fontsize=7.2); ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(FIG / "joints.png"); plt.close(fig)


def fig_size():
    """OKS by person size (box area quintiles) and the share of the bottom-OKS tail per quintile."""
    fig, ax = plt.subplots(figsize=(4.8, 2.5))
    for n, c in zip(NAMES[:2], (BLUE, ORANGE)):  # Market1501 boxes are almost all the full 128x64 image
        z = load(n); anns = _gt(n)
        area = np.array([anns[int(a)]["area"] for a in z["ann_id"]])
        q = np.quantile(area, [0, .2, .4, .6, .8, 1])
        b = np.clip(np.digitize(area, q[1:-1]), 0, 4)
        ax.plot(range(5), [z["oks"][b == i].mean() for i in range(5)], color=c, lw=1.6, marker="o", ms=4, label=f"giấu {LABEL[n]}")
        print(n, "area quintile edges", np.round(q).astype(int), "mean oks", [round(float(z['oks'][b == i].mean()), 3) for i in range(5)])
    ax.set_xticks(range(5), ["nhỏ nhất", "2", "3", "4", "lớn nhất"]); ax.set_xlabel("nhóm 20% theo diện tích người")
    ax.set_ylabel("OKS trung bình"); ax.legend(frameon=False, fontsize=7.2)
    fig.tight_layout(); fig.savefig(FIG / "size.png"); plt.close(fig)


def _ap_per_threshold(name, z, score):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(ANN / f"train_{name}.json"))
        preds = [{"image_id": int(i), "category_id": 1, "score": float(s), "keypoints": k.tolist()}
                 for i, s, k in zip(z["image_id"], score, z["keypoints"])]
        ev = COCOeval(gt, gt.loadRes(copy.deepcopy(preds)), "keypoints")
        ev.params.kpt_oks_sigmas = SIGMAS
        ev.evaluate(); ev.accumulate()
    pr = ev.eval["precision"][:, :, 0, 0, -1]          # (T thresholds, R recall points)
    return np.array([p[p > -1].mean() for p in pr])


def fig_thresholds():
    """Mức tăng AP so với điểm mặc định theo ngưỡng OKS, mỗi miền giấu một ô: bộ chấm học được, điểm mặc định bỏ hai
    kênh tai (không bao giờ có nhãn), và OKS thật (cắt ở trục, giá trị ghi trên cột)."""
    data, resc = lodo_scores()
    thr = np.arange(.5, .96, .05); keep = [k for k in range(17) if k not in (3, 4)]; top = 2.4
    lab = {"peta": "giấu PETA", "pa100k": "giấu PA100k", "market": "giấu Market1501"}
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.5), sharey=True)
    for ax, n in zip(axs, NAMES):
        z = data[n]; base = _ap_per_threshold(n, z, z["X"][:, 70])
        g = {k: (_ap_per_threshold(n, z, sc) - base) * 100 for k, sc in
             (("ora", z["oks"]), ("ears", z["X"][:, :17][:, keep].mean(1)), ("lrn", resc[n]))}
        ax.bar(thr, np.minimum(g["ora"], top), width=.04, color=GRAY, label="OKS thật (lý tưởng)")
        for t, v in zip(thr, g["ora"]):
            if v > top:
                ax.text(t, top + .05, f"{v:.1f}", ha="center", fontsize=6.4, color=MUTED)
        ax.bar(thr - .009, g["ears"], width=.017, color=ORANGE, label="mặc định bỏ kênh tai")
        ax.bar(thr + .009, g["lrn"], width=.017, color=BLUE, label="bộ chấm học được")
        ax.axhline(0, color=MUTED, lw=.6); ax.set_ylim(-.3, top + .4)
        ax.set_xticks(thr[::2], [f"{t:.2f}" for t in thr[::2]], fontsize=7); ax.set_title(lab[n], fontsize=8.4, loc="left")
        ax.set_xlabel("ngưỡng OKS", fontsize=8); ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("ΔAP so với mặc định (điểm %)", fontsize=8); axs[0].legend(frameon=False, fontsize=6.6, loc="upper left")
    fig.tight_layout(w_pad=.5); fig.savefig(FIG / "per_threshold.png"); plt.close(fig)


def fig_noise_curve():
    """AP as a function of score quality: score = OKS + Gaussian noise; mark default and learned scores (Spearman)."""
    from scipy.stats import spearmanr
    data, resc = lodo_scores()
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(4.8, 2.7))
    for n, c in zip(NAMES, (BLUE, ORANGE, AQUA)):
        z = data[n]; o = z["oks"]
        xs, ys = [], []
        for sd in (0, .01, .02, .04, .06, .1, .2, .5, 5):
            s = o + rng.normal(0, sd, len(o))
            xs.append(spearmanr(s, o).correlation); ys.append(ap(n, z, s))
        ax.plot(xs, ys, color=c, lw=1.4, marker=".", ms=5, label=f"giấu {LABEL[n]}")
        for s, mk in ((z["X"][:, 70], "o"), (resc[n], "s")):
            ax.scatter(spearmanr(s, o).correlation, ap(n, z, s), color=c, marker=mk, s=34, edgecolor=INK, lw=.6, zorder=4)
    ax.scatter([], [], marker="o", color="white", edgecolor=INK, label="điểm mặc định"); ax.scatter([], [], marker="s", color="white", edgecolor=INK, label="bộ chấm học được")
    ax.set_xlabel("tương quan Spearman giữa điểm tin cậy và OKS"); ax.set_ylabel("OKS AP"); ax.legend(frameon=False, fontsize=6.6)
    fig.tight_layout(); fig.savefig(FIG / "noise_curve.png"); plt.close(fig)


def fig_residuals():
    """PETA instances that the learned score still ranks high although their OKS is low."""
    data, resc = lodo_scores()
    z = data["peta"]; s = resc["peta"]
    d = json.load(open(ANN / "train_peta.json")); ims = {im["id"]: im for im in d["images"]}; anns = {a["id"]: a for a in d["annotations"]}
    rank = s.argsort().argsort() / len(s)
    cand = np.nonzero((z["oks"] < .5) & (rank > .5))[0]
    print("low-OKS (<0.5) instances", int((z['oks'] < .5).sum()), "of which ranked in the top half by the learned score", len(cand))
    pick = cand[np.argsort(-rank[cand])][:6]
    fig, axs = plt.subplots(1, 6, figsize=(7.2, 2.6))
    for ax, i in zip(axs, pick):
        a = anns[int(z["ann_id"][i])]; im = ims[a["image_id"]]
        img = Image.open(PETA_IMG / Path(im["file_name"]).name).convert("RGB")
        ax.imshow(img); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        g = np.array(a["keypoints"], float).reshape(17, 3); p = z["keypoints"][i].reshape(17, 3)
        for kp, c, lw in ((g, AQUA, 1.6), (p, ORANGE, 1.2)):
            for u, v in EDGES:
                if kp is g and (g[u, 2] == 0 or g[v, 2] == 0):
                    continue
                ax.plot([kp[u, 0], kp[v, 0]], [kp[u, 1], kp[v, 1]], color=c, lw=lw, alpha=.9)
        ax.set_title(f"OKS {z['oks'][i]:.2f}\nhạng top {100 * (1 - rank[i]):.0f}%", fontsize=7)
        ax.set_xlim(0, img.width); ax.set_ylim(img.height, 0)
    fig.savefig(FIG / "residuals.png"); plt.close(fig)


if __name__ == "__main__":
    todo = sys.argv[1:] or ["fig_examples", "fig_oks_hist", "fig_score_choices", "fig_scatter", "fig_importance",
                            "fig_leaderboard", "fig_lodo", "fig_joints", "fig_size",
                            "fig_thresholds", "fig_noise_curve", "fig_residuals"]
    for f in todo:
        globals()[f]()

