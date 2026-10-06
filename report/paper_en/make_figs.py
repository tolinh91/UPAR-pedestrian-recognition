"""Figures of the English Track 2 paper. Run from report/paper_en/: ../../.venv/bin/python make_figs.py
Reuses the data helpers and logged constants of ../paper/make_figs.py."""
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
INK, MUTED, BLUE, ORANGE, AQUA, YELLOW, RED, GRAY = "#121211", "#85837c", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948", "#b4b2a9"
EN_ATTR = {"Age-Young": "young", "Age-Adult": "adult", "Age-Old": "old", "Gender-Female": "female",
           "Hair-Length-Short": "short hair", "Hair-Length-Long": "long hair", "Hair-Length-Bald": "bald",
           "UpperBody-Length-Short": "short sleeves", "LowerBody-Length-Short": "shorts",
           "LowerBody-Type-Trousers&Shorts": "trousers/shorts", "LowerBody-Type-Skirt&Dress": "skirt/dress",
           "Accessory-Backpack": "backpack", "Accessory-Bag": "bag", "Accessory-Glasses-Normal": "glasses",
           "Accessory-Glasses-Sun": "sunglasses", "Accessory-Hat": "hat"}


def en(a):
    if "-Color-" in a:
        return ("upper " if a.startswith("Upper") else "lower ") + a.split("-")[-1].lower()
    return EN_ATTR[a]


def fig_example():
    from upar.data import DATA, attr_names, load_split
    from upar.retrieval import score_loglik
    from eval_retrieval import load_task2
    va, A = load_split("val"), attr_names()
    z = np.load(V.ROOT / "runs/remote2/kd2_a08_e6_PETA/held_val_logits_last.npz")
    gt = va[A].values[z["idx"]].astype(np.float32)
    ids, queries = load_task2("val"); qid_of = dict(zip(ids.image, ids.qid))
    paths = va.image.values[z["idx"]]; gq = np.array([qid_of[p] for p in paths])
    p = 1 / (1 + np.exp(-z["logits"]))
    q = 156
    s = score_loglik(queries[q][None], p)[0]; order = np.argsort(-s)[:8]; hit = gq[order] == q
    qv = queries[q]
    pos = [en(A[k]) for k in np.nonzero(qv)[0]]
    if not qv[A.index("Gender-Female")]:
        pos.insert(1, "male")
    fig, axs = plt.subplots(1, 8, figsize=(6.9, 1.85))
    for r, (ax, i) in enumerate(zip(axs, order)):
        ax.imshow(Image.open(DATA / paths[i]).convert("RGB")); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        wrong = int((gt[i] != qv).sum())
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(AQUA if hit[r] else RED); sp.set_linewidth(2.2)
        ax.set_title(f"#{r + 1}  " + ("match" if hit[r] else f"{wrong} off"), fontsize=6.8)
    fig.suptitle("query: " + ", ".join(pos), fontsize=7, y=1.02)
    fig.savefig(FIG / "example.pdf"); plt.close(fig)


def _family(name):
    if name.startswith(("b_", "na_", "clean_", "cnxB", "eva02B")):
        return "ViT-B (and B-size)"
    if name.startswith(("L176", "T_dino", "Lk18", "kd")):
        return "DINOv3 ViT-L"
    return "other L-size encoders"


def fig_nll():
    """mADM against AUC, NLL and the listwise criterion over the PETA-held-out runs of t12_runs.json."""
    import json
    from scipy.stats import spearmanr
    d = json.load(open(HERE.parents[2] / "report_review/analysis/t12_runs.json"))
    runs = {k: v for k, v in d.items() if v["held"] == "PETA" and k != "clipL_168_PETA"}
    cols = {"ViT-B (and B-size)": ORANGE, "DINOv3 ViT-L": BLUE, "other L-size encoders": GRAY}
    fig, axs = plt.subplots(1, 3, figsize=(6.9, 1.85), sharey=True)
    for ax, key, lab in ((axs[0], "auc", "mean AUC"), (axs[1], "nll", "NLL per image (nats)"),
                         (axs[2], "listwise", "listwise log-likelihood of the match")):
        for fam, c in cols.items():
            r = [v for k, v in runs.items() if _family(k) == fam]
            ax.scatter([v[key] for v in r], [v["madm"] for v in r], s=10, color=c, edgecolor="white", lw=.4, zorder=3,
                       label=fam if key == "auc" else None)
        x = np.array([v[key] for v in runs.values()]); y = np.array([v["madm"] for v in runs.values()])
        b = np.array([_family(k).startswith("ViT-B") for k in runs])
        ax.set_title(f"$\\rho$ = {spearmanr(x, y).correlation:+.2f} (ViT-B {spearmanr(x[b], y[b]).correlation:+.2f})", loc="left")
        ax.set_xlabel(lab)
    axs[1].invert_xaxis(); axs[0].set_ylabel("mADM"); axs[0].legend(frameon=False, loc="lower right", fontsize=5.8)
    fig.tight_layout(w_pad=.6); fig.savefig(FIG / "nll.pdf"); plt.close(fig)


def fig_temperature():
    cols = {"PETA": BLUE, "Market1501": AQUA, "PA100k": ORANGE}
    fig, ax = plt.subplots(figsize=(3.35, 1.8))
    seen = set()
    for (held, _), d in V.TEMP.items():
        T = sorted(d)
        ax.plot(T, [(d[t] - d[1.0]) * 100 for t in T], color=cols[held], lw=1.3, marker="o", ms=2.5,
                label=f"{held} held out" if held not in seen else None)
        seen.add(held)
    ax.axhline(0, color=MUTED, lw=.6); ax.set_xlabel("temperature $T$"); ax.set_ylabel("$\\Delta$mADM vs. $T{=}1$ (pts)")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(FIG / "temperature.pdf"); plt.close(fig)


def fig_reliability():
    from upar.data import attr_names, load_split
    va, A = load_split("val"), attr_names()
    edges = np.linspace(0, 1, 11)
    fig, ax = plt.subplots(figsize=(3.35, 2.0))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=.6, ls="--")
    for run, lab, c in (("remote/b_q_PETA_s1", "ViT-B", ORANGE), ("remote2/kd2_a08_e6_PETA", "ViT-L distilled", BLUE)):
        for f, ls, tag in (("held_val_logits_last.npz", "-", "PETA"), ("seen_val_logits_last.npz", ":", "source")):
            z = np.load(V.ROOT / f"runs/{run}/{f}")
            y = va[A].values[z["idx"]].astype(float); p = 1 / (1 + np.exp(-z["logits"]))
            ok = ~np.isnan(y); pp, yy = p[ok], y[ok]; b = np.clip(np.digitize(pp, edges) - 1, 0, 9)
            ece = sum((b == i).mean() * abs(pp[b == i].mean() - yy[b == i].mean()) for i in range(10) if (b == i).any())
            ax.plot([pp[b == i].mean() for i in range(10)], [yy[b == i].mean() for i in range(10)], color=c, ls=ls, lw=1.2,
                    marker="o" if ls == "-" else None, ms=2.5, label=f"{lab}, {tag} (ECE {ece:.3f})")
    ax.set_xlabel("predicted probability"); ax.set_ylabel("observed frequency"); ax.legend(frameon=False, loc="upper left", fontsize=5.8)
    fig.tight_layout(); fig.savefig(FIG / "reliability.pdf"); plt.close(fig)


def fig_difficulty():
    from upar.retrieval import score_loglik
    A, gt, lg, gq, queries = V._load("remote2/kd2_a08_e6_PETA")
    p = 1 / (1 + np.exp(-lg)); q_ids = np.unique(gq); S = score_loglik(queries[q_ids], p)
    adm, R, first = [], [], []
    for a, q in enumerate(q_ids):
        dom = (gt == queries[q]).mean(1); mean, mx = dom.mean(), dom.max()
        nd = np.clip((dom - mean) / max(mx - mean, 1e-9), 0, None)
        o = np.argsort(-S[a], kind="stable"); ranks = np.nonzero(gq[o] == q)[0] + 1; cs = np.cumsum(nd[o])
        adm.append(np.mean(cs[ranks - 1] / ranks)); R.append(len(ranks)); first.append(ranks[0])
    adm, R, first = map(np.array, (adm, R, first))
    cats = [(1, 1, "1"), (2, 5, "2-5"), (6, 20, "6-20"), (21, 100, "21-100"), (101, 10 ** 7, ">100")]
    fig, ax = plt.subplots(figsize=(3.35, 1.75))
    share = [((first >= a) & (first <= b)).mean() for a, b, _ in cats]
    vals = [adm[(first >= a) & (first <= b)].mean() for a, b, _ in cats]
    ax.bar(range(5), share, color=GRAY, width=.6, label="share of queries")
    ax.plot(range(5), vals, color=BLUE, marker="o", ms=3, lw=1.3, label="mean ADM")
    for i, (s, v) in enumerate(zip(share, vals)):
        ax.text(i, v + .03, f"{v:.2f}", ha="center", fontsize=6.2, color=BLUE)
    ax.set_xticks(range(5), [c[2] for c in cats]); ax.set_xlabel("rank of the first exact match"); ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper right"); ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(FIG / "difficulty.pdf"); plt.close(fig)


def fig_nll_groups():
    runs = [("remote/b_q_PETA_s2", "ViT-B query", GRAY), ("remote/b_q_gce_PETA_s2", "+ group CE", ORANGE),
            ("remote2/kd2_a08_e6_PETA", "ViT-L distilled", BLUE)]
    groups = ["age", "gender", "hair", "sleeves", "upper color", "shorts", "lower color", "lower type", "accessory"]
    fig, ax = plt.subplots(figsize=(3.35, 1.9))
    w = .27
    for r, (run, lab, c) in enumerate(runs):
        A, gt, lg, _, _ = V._load(run)
        p = np.clip(1 / (1 + np.exp(-lg)), 1e-4, 1 - 1e-4); nll = -(gt * np.log(p) + (1 - gt) * np.log(1 - p)).mean(0)
        ax.bar(np.arange(9) + (r - 1) * w, [nll[list(ix)].sum() for _, ix in V.GROUPS], w, color=c, label=f"{lab} ({nll.sum():.2f})")
    ax.set_xticks(range(9), groups, rotation=35, ha="right"); ax.set_ylabel("NLL per image (nats)")
    ax.legend(frameon=False); ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(FIG / "nll_groups.pdf"); plt.close(fig)


def fig_confusers():
    from upar.retrieval import score_loglik
    A, gt, lg, gq, queries = V._load("remote2/kd2_a08_e6_PETA")
    p = 1 / (1 + np.exp(-lg)); q_ids = np.unique(gq); S = score_loglik(queries[q_ids], p)
    cnt = np.zeros(len(A)); n = 0; nw = []
    for a, q in enumerate(q_ids):
        o = np.argsort(-S[a], kind="stable"); first = np.nonzero(gq[o] == q)[0][0]; top = o[:min(first, 10)]
        if len(top):
            d = gt[top] != queries[q][None]; cnt += d.sum(0); n += len(top); nw += list(d.sum(1))
    share = cnt / n; k = np.argsort(-share)[:12]; nw = np.array(nw)
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 1.9), gridspec_kw={"width_ratios": [1.5, 1]})
    axs[0].barh(range(12)[::-1], share[k] * 100, color=BLUE, height=.6)
    axs[0].set_yticks(range(12)[::-1], [en(A[i]) for i in k]); axs[0].set_xlabel("% of confusers that differ in this attribute")
    axs[0].grid(axis="y", visible=False)
    vals = [np.mean(nw == j) * 100 for j in range(1, 6)] + [np.mean(nw >= 6) * 100]
    axs[1].bar(range(6), vals, color=AQUA, width=.6); axs[1].set_xticks(range(6), ["1", "2", "3", "4", "5", "$\\geq$6"])
    axs[1].set_xlabel("attributes that differ"); axs[1].set_ylabel("% of confusers"); axs[1].grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(FIG / "confusers.pdf"); plt.close(fig)


if __name__ == "__main__":
    for f in sys.argv[1:] or ["fig_example", "fig_nll", "fig_temperature", "fig_reliability", "fig_difficulty", "fig_nll_groups", "fig_confusers"]:
        globals()[f]()
