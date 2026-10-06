"""Figures of the Track 2 paper (track2_paper.tex). Run from report/paper/ with the Track 2 venv:
../../.venv/bin/python make_figs.py
Data: held-out PETA logits of the leave-one-dataset-out runs (runs/remote*/), the val queries (eval_retrieval.load_task2)
and the scoring-function / temperature measurements of 2026-10-02 and 2026-10-04 (constants below)."""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "par"))
FIG = HERE / "figs"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e3e2db", "grid.linewidth": .6, "axes.edgecolor": "#9a988f",
                     "axes.axisbelow": True, "savefig.dpi": 220, "savefig.bbox": "tight"})
INK, MUTED, BLUE, ORANGE, AQUA, YELLOW, GRAY, RED = "#121211", "#85837c", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#b4b2a9", "#e34948"


VI_ATTR = {"Age-Young": "tuổi trẻ", "Age-Adult": "trưởng thành", "Age-Old": "già", "Gender-Female": "nữ",
           "Hair-Length-Short": "tóc ngắn", "Hair-Length-Long": "tóc dài", "Hair-Length-Bald": "hói",
           "UpperBody-Length-Short": "áo ngắn tay", "LowerBody-Length-Short": "quần ngắn",
           "LowerBody-Type-Trousers&Shorts": "quần (không phải váy)", "LowerBody-Type-Skirt&Dress": "váy",
           "Accessory-Backpack": "ba lô", "Accessory-Bag": "túi", "Accessory-Glasses-Normal": "kính",
           "Accessory-Glasses-Sun": "kính râm", "Accessory-Hat": "mũ"}
VI_COL = {"Black": "đen", "Blue": "xanh dương", "Brown": "nâu", "Green": "xanh lá", "Grey": "xám", "Orange": "cam",
          "Pink": "hồng", "Purple": "tím", "Red": "đỏ", "White": "trắng", "Yellow": "vàng", "Other": "màu khác"}


def vi_attr(a):
    if "-Color-" in a:
        return ("áo " if a.startswith("Upper") else "quần ") + VI_COL[a.split("-")[-1]]
    return VI_ATTR[a]


def fig_ndom():
    """Credit that mADM gives a gallery image with j wrong attributes, at the typical PETA mean dom of 0.82."""
    j = np.arange(0, 9)
    nd = np.clip((1 - j / 40 - 0.82) / (1 - 0.82), 0, None)
    fig, ax = plt.subplots(figsize=(4.2, 2.3))
    ax.bar(j, nd, color=[BLUE] + [ORANGE] * 8, width=.62, edgecolor="white", lw=2)
    for a, b in zip(j, nd):
        ax.text(a, b + .02, f"{b:.2f}", ha="center", fontsize=7.4)
    ax.set_xlabel("số thuộc tính lệch so với truy vấn"); ax.set_ylabel("ndom (điểm cộng)")
    ax.set_ylim(0, 1.12); ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(FIG / "ndom.png"); plt.close(fig)


# PETA held out, ViT-B / ViT-L variants (AUC, NLL per image, mADM with log-likelihood ranking), 2026-10-02 notes.
VAR = [("ViT-L 256 tuyến tính", .9126, 6.569, .6697), ("ViT-B truy vấn + group CE", .9048, 6.771, .6424),
       ("ViT-B truy vấn, 3 epoch", .9041, 7.102, .6350), ("ViT-L 160 tuyến tính", .8960, 6.823, .6335),
       ("ViT-B tuyến tính, KD logit", .9028, 6.861, .6334), ("ViT-B tuyến tính", .9076, 6.993, .6313),
       ("ViT-B truy vấn + cân bằng trắng", .9001, 6.985, .6294), ("PE-Core tuyến tính", .8950, 6.916, .6276),
       ("ViT-B truy vấn 2 tầng", .9020, 7.013, .6270), ("ViT-B truy vấn (lần 1)", .9007, 6.976, .6261),
       ("ViT-B truy vấn (lần 2)", .9040, 6.998, .6254), ("ViT-B head kép", .9041, 7.037, .6260),
       ("ViT-B truy vấn, 4 tầng token", .8988, 6.997, .6242), ("ViT-B truy vấn, cân bằng bộ dữ liệu", .9049, 7.008, .6214)]


def fig_auc_nll():
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    a = np.array([[v[1], v[2], v[3]] for v in VAR])
    for ax, x, lab in ((axs[0], a[:, 0], "AUC trung bình (PETA giấu)"), (axs[1], a[:, 1], "NLL trên một ảnh (thấp là tốt)")):
        ax.scatter(x, a[:, 2], s=22, color=BLUE, edgecolor="white", lw=.6, zorder=3)
        r = np.corrcoef(x, a[:, 2])[0, 1]
        ax.set_xlabel(lab); ax.set_title(f"hệ số tương quan Pearson r = {r:+.2f}", fontsize=8.4, loc="left")
    axs[1].invert_xaxis()
    axs[0].set_ylabel("mADM (xếp hạng log-likelihood)")
    for name, auc, nll, m in VAR:
        if name in ("ViT-L 256 tuyến tính", "ViT-B truy vấn + group CE", "ViT-B truy vấn, 3 epoch", "ViT-L 160 tuyến tính"):
            axs[1].annotate(name, (nll, m), xytext=(4, -3), textcoords="offset points", fontsize=6.4)
    fig.tight_layout(); fig.savefig(FIG / "auc_nll.png"); plt.close(fig)


# mADM (log-likelihood ranking) as a function of the temperature, one line per run; 2026-10-04 measurements.
TEMP = {("PETA", "ViT-L KD 2 thầy"): {1.0: .6635, 1.1: .6658, 1.2: .6670, 1.3: .6678, 1.4: .6655},
        ("PETA", "ViT-L KD DINO+MetaCLIP"): {1.0: .6697, 1.1: .6720, 1.2: .6718, 1.3: .6702, 1.4: .6692},
        ("Market1501", "ViT-B truy vấn + group CE"): {1.0: .7136, 1.1: .7139, 1.2: .7142, 1.3: .7142, 1.4: .7131},
        ("Market1501", "ViT-B, không tăng cường"): {1.0: .7126, 1.1: .7135, 1.2: .7138, 1.3: .7140, 1.4: .7136},
        ("PA100k", "ViT-B truy vấn"): {1.0: .5871, 1.1: .5829, 1.2: .5787, 1.3: .5737, 1.4: .5685}}


def fig_temperature():
    cols = {"PETA": BLUE, "Market1501": AQUA, "PA100k": ORANGE}
    fig, ax = plt.subplots(figsize=(4.8, 2.7))
    seen = set()
    for (held, name), d in TEMP.items():
        T = sorted(d); y = [d[t] - d[1.0] for t in T]
        ax.plot(T, y, color=cols[held], lw=1.6, marker="o", ms=3.5, label=f"giấu {held}" if held not in seen else None)
        seen.add(held)
    ax.axhline(0, color=MUTED, lw=.8)
    ax.set_xlabel("nhiệt độ T (logit / T)"); ax.set_ylabel("mADM so với T = 1")
    ax.legend(frameon=False, fontsize=7.6, loc="lower left")
    fig.tight_layout(); fig.savefig(FIG / "temperature.png"); plt.close(fig)


def fig_kd():
    labels = ["ViT-B\nkhông KD", "ViT-B\nKD 2 thầy", "ViT-B\nKD 3 thầy", "ViT-L\nkhông KD", "ViT-L\nKD 2 thầy\n6 epoch",
              "ViT-L\nKD 2 thầy\n10 epoch", "ViT-L\nKD kd-mix\n20 epoch"]
    vals = [.6406, .6577, .6582, .6496, .6635, .6681, .6751]
    cols = [MUTED, ORANGE, ORANGE, MUTED, ORANGE, ORANGE, BLUE]
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    x = np.arange(len(vals))
    ax.bar(x, np.array(vals) - .62, bottom=.62, color=cols, width=.6, edgecolor="white", lw=2)
    for i, v in enumerate(vals):
        ax.text(i, v + .0012, f"{v:.4f}", ha="center", fontsize=7.4)
    ax.set_xticks(x, labels, fontsize=7.2); ax.set_ylim(.62, .682); ax.set_ylabel("mADM, PETA giấu")
    ax.grid(axis="x", visible=False); ax.axvline(2.5, color=GRAY, lw=.8)
    fig.tight_layout(); fig.savefig(FIG / "kd_madm.png"); plt.close(fig)


LB = [("v3b", "ViT-B truy vấn + group CE", .6247), ("v4", "ViT-L KD, dự phòng\nthời gian", .2375),
      ("v5", "ViT-B KD 2 thầy\n+ group CE", .6315), ("v6", "ViT-B kd-mix\n+ group CE", .6365)]


def fig_leaderboard():
    fig, ax = plt.subplots(figsize=(4.8, 2.4))
    for i, (t, l, v) in enumerate(LB):
        if v < .5:
            ax.scatter(i, .615, marker="v", color=RED, s=28); ax.text(i, .6165, f"{v:.4f}", ha="center", fontsize=7, color=RED)
        else:
            ax.scatter(i, v, s=26, color=BLUE if v == max(x[2] for x in LB) else INK, zorder=3)
            ax.text(i, v + .0012, f"{v:.4f}", ha="center", fontsize=7.2)
    ax.set_xticks(range(len(LB)), [f"{t}\n{l}" for t, l, _ in LB], fontsize=6.6)
    ax.set_ylim(.613, .641); ax.set_ylabel("mADM (tập chấm ẩn)"); ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(FIG / "leaderboard.png"); plt.close(fig)


def fig_example():
    """One real val query on the PETA gallery, ranked by log-likelihood with the held-out ViT-L KD model."""
    from eval_retrieval import load_task2
    from upar.data import DATA, attr_names, load_split
    from upar.retrieval import score_loglik
    va, A = load_split("val"), attr_names()
    z = np.load(ROOT / "runs/remote2/kd2_a08_e6_PETA/held_val_logits_last.npz")
    gt = va[A].values[z["idx"]].astype(np.float32)
    ids, queries = load_task2("val")
    qid_of = dict(zip(ids.image, ids.qid))
    paths = va.image.values[z["idx"]]
    gq = np.array([qid_of[p] for p in paths])
    q_ids, counts = np.unique(gq, return_counts=True)
    p = 1 / (1 + np.exp(-z["logits"]))
    rng = np.random.default_rng(3)
    cand = [q for q, c in zip(q_ids, counts) if c == 3]
    best = None
    for q in rng.permutation(cand)[:400]:  # a query whose 3 matches are found among the first 8, with near misses between
        s = score_loglik(queries[q][None], p)[0]
        order = np.argsort(-s)[:8]
        hit = gq[order] == q
        if 2 <= hit.sum() <= 3 and hit[:2].any() and (~hit).sum() >= 4:
            best = (q, order, hit); break
    q, order, hit = best
    qv = queries[q]
    VI = {"Age-Young": "trẻ", "Age-Adult": "trưởng thành", "Age-Old": "già", "Gender-Female": "nữ",
          "Hair-Length-Short": "tóc ngắn", "Hair-Length-Long": "tóc dài", "Hair-Length-Bald": "hói",
          "UpperBody-Length-Short": "áo ngắn tay", "LowerBody-Length-Short": "quần ngắn",
          "LowerBody-Type-Trousers&Shorts": "quần (không phải váy)", "LowerBody-Type-Skirt&Dress": "váy",
          "Accessory-Backpack": "ba lô", "Accessory-Bag": "túi", "Accessory-Glasses-Normal": "kính",
          "Accessory-Glasses-Sun": "kính râm", "Accessory-Hat": "mũ"}
    COL = {"Black": "đen", "Blue": "xanh dương", "Brown": "nâu", "Green": "xanh lá", "Grey": "xám", "Orange": "cam",
           "Pink": "hồng", "Purple": "tím", "Red": "đỏ", "White": "trắng", "Yellow": "vàng", "Other": "màu khác"}
    def vi(a):
        if "-Color-" in a:
            return ("áo " if a.startswith("Upper") else "quần ") + COL[a.split("-")[-1]]
        return VI[a]
    pos = [vi(A[k]) for k in np.nonzero(qv)[0]]
    if not qv[A.index("Gender-Female")]:
        pos.insert(1, "nam")
    fig, axs = plt.subplots(1, 8, figsize=(7.4, 2.6))
    for r, (ax, i) in enumerate(zip(axs, order)):
        im = Image.open(DATA / paths[i]).convert("RGB")
        ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
        wrong = int((gt[i] != qv).sum())
        c = AQUA if hit[r] else (YELLOW if wrong <= 2 else RED)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(c); sp.set_linewidth(3)
        ax.set_title(f"#{r + 1}\n{'khớp' if hit[r] else f'lệch {wrong}'}", fontsize=7.4, color=INK)
        ax.grid(False)
    fig.suptitle("Truy vấn: " + ", ".join(pos), fontsize=7.6, y=1.04)
    fig.savefig(FIG / "example.png"); plt.close(fig)
    print("query", q, "positives", pos, "hits", hit.astype(int))




def _load(run):
    from eval_retrieval import load_task2
    from upar.data import attr_names, load_split
    va, A = load_split("val"), attr_names()
    z = np.load(ROOT / f"runs/{run}/held_val_logits_last.npz")
    gt = va[A].values[z["idx"]].astype(np.float32)
    ids, queries = load_task2("val")
    qid_of = dict(zip(ids.image, ids.qid))
    gq = np.array([qid_of[p] for p in va.image.values[z["idx"]]])
    return A, gt, z["logits"], gq, queries


GROUPS = [("Tuổi", range(0, 3)), ("Giới tính", [3]), ("Tóc", range(4, 7)), ("Áo ngắn tay", [7]), ("Màu áo", range(8, 20)),
          ("Quần ngắn", [20]), ("Màu quần", range(21, 33)), ("Kiểu quần", range(33, 35)), ("Phụ kiện", range(35, 40))]


def fig_nll_groups():
    """Per-group share of the full-vector NLL on held-out PETA, ViT-B query vs ViT-L distilled (seen-domain NLL too)."""
    runs = [("remote/b_q_PETA_s1", "ViT-B truy vấn", GRAY), ("remote/b_q_gce_PETA_s2", "ViT-B truy vấn + group CE", ORANGE),
            ("remote2/kd2_a08_e6_PETA", "ViT-L KD 2 thầy", BLUE)]
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    w = .26
    for r, (run, lab, c) in enumerate(runs):
        A, gt, lg, _, _ = _load(run)
        p = np.clip(1 / (1 + np.exp(-lg)), 1e-4, 1 - 1e-4)
        nll = -(gt * np.log(p) + (1 - gt) * np.log(1 - p)).mean(0)
        vals = [nll[list(ix)].sum() for _, ix in GROUPS]
        ax.bar(np.arange(len(GROUPS)) + (r - 1) * w, vals, w, color=c, label=f"{lab} (tổng {nll.sum():.2f})", edgecolor="white", lw=1)
        print(lab, "total NLL", round(float(nll.sum()), 3), [f"{g}:{v:.2f}" for (g, _), v in zip(GROUPS, vals)])
    ax.set_xticks(range(len(GROUPS)), [g for g, _ in GROUPS], fontsize=7.6)
    ax.set_ylabel("NLL trung bình trên ảnh (nats)"); ax.legend(frameon=False, fontsize=7.4); ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(FIG / "nll_groups.png"); plt.close(fig)


def fig_query_difficulty():
    """ADM per query vs number of exact matches R and vs the log-probability rank of the first match (ViT-L KD, PETA)."""
    from upar.retrieval import score_loglik
    A, gt, lg, gq, queries = _load("remote2/kd2_a08_e6_PETA")
    p = 1 / (1 + np.exp(-lg))
    q_ids = np.unique(gq)
    S = score_loglik(queries[q_ids], p)
    K = gt.shape[1]
    adm, R, first = [], [], []
    for a, q in enumerate(q_ids):
        qq = queries[q]
        dom = (gt == qq).mean(1)
        mean, mx = dom.mean(), dom.max()
        nd = np.clip((dom - mean) / max(mx - mean, 1e-9), 0, None)
        o = np.argsort(-S[a], kind="stable")
        ranks = np.nonzero(gq[o] == q)[0] + 1
        cs = np.cumsum(nd[o])
        adm.append(float(np.mean(cs[ranks - 1] / ranks))); R.append(len(ranks)); first.append(ranks[0])
    adm, R, first = map(np.array, (adm, R, first))
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))
    bins = [(1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"), (5, 9, "5-9"), (10, 10 ** 6, "≥10")]
    vals = [adm[(R >= a) & (R <= b)].mean() for a, b, _ in bins]
    cnt = [((R >= a) & (R <= b)).sum() for a, b, _ in bins]
    axs[0].bar(range(len(bins)), vals, color=BLUE, width=.6, edgecolor="white", lw=2)
    for i, (v, n) in enumerate(zip(vals, cnt)):
        axs[0].text(i, v + .01, f"{v:.2f}\n(n={n})", ha="center", fontsize=6.8)
    axs[0].set_xticks(range(len(bins)), [b[2] for b in bins]); axs[0].set_xlabel("số ảnh khớp chính xác R")
    axs[0].set_ylabel("ADM trung bình"); axs[0].set_ylim(0, 1.05); axs[0].grid(axis="x", visible=False)
    cats = [(1, 1, "1"), (2, 5, "2-5"), (6, 20, "6-20"), (21, 100, "21-100"), (101, 10 ** 6, ">100")]
    share = [((first >= a) & (first <= b)).mean() for a, b, _ in cats]
    axs[1].bar(range(len(cats)), share, color=AQUA, width=.6, edgecolor="white", lw=2)
    for i, v in enumerate(share):
        axs[1].text(i, v + .01, f"{v:.0%}", ha="center", fontsize=7)
    axs[1].set_xticks(range(len(cats)), [c[2] for c in cats]); axs[1].set_xlabel("thứ hạng của ảnh khớp đầu tiên")
    axs[1].set_ylabel("tỉ lệ truy vấn"); axs[1].grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(FIG / "query_difficulty.png"); plt.close(fig)
    print("queries", len(adm), "mADM", adm.mean().round(4), "first@1", (first == 1).mean().round(3), "share R=1", (R == 1).mean().round(3))


def table_metrics():
    from upar.retrieval import retrieval_metrics, score_loglik, score_l1
    for run, lab in (("remote/b_q_PETA_s1", "ViT-B truy vấn"), ("remote/b_q_gce_PETA_s2", "ViT-B + group CE"),
                     ("remote2/kd2_a08_e6_PETA", "ViT-L KD 2 thầy"), ("remote2/kdB_mc_e10_PETA", "ViT-L KD DINO+MetaCLIP")):
        A, gt, lg, gq, queries = _load(run)
        p = 1 / (1 + np.exp(-lg)); q_ids = np.unique(gq)
        for name, fn in (("loglik", score_loglik), ("L1", score_l1)):
            m = retrieval_metrics(fn(queries[q_ids], p), queries[q_ids], gt, gq, q_ids)
            print(f"{lab:24s} {name:6s} " + " ".join(f"{k} {m[k]:.4f}" for k in ("mADM", "mAP", "mINP", "Rank-1", "Rank-5", "Rank-10")))


def fig_reliability():
    """Reliability diagram on held-out PETA (all 40 attributes pooled) and on the seen domains' val, ViT-B vs ViT-L KD."""
    from upar.data import attr_names, load_split
    va, A = load_split("val"), attr_names()
    runs = [("remote/b_q_PETA_s1", "ViT-B truy vấn", ORANGE), ("remote2/kd2_a08_e6_PETA", "ViT-L KD 2 thầy", BLUE)]
    edges = np.linspace(0, 1, 11)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, f, title in ((axs[0], "seen_val_logits_last.npz", "Val miền nguồn (Market1501 + PA100k)"),
                         (axs[1], "held_val_logits_last.npz", "PETA giấu")):
        ax.plot([0, 1], [0, 1], color=MUTED, lw=.8, ls="--")
        for run, lab, c in runs:
            z = np.load(ROOT / f"runs/{run}/{f}")
            y = va[A].values[z["idx"]].astype(float); p = 1 / (1 + np.exp(-z["logits"]))
            ok = ~np.isnan(y)
            pp, yy = p[ok], y[ok]
            b = np.clip(np.digitize(pp, edges) - 1, 0, 9)
            xs = [pp[b == i].mean() for i in range(10) if (b == i).sum() > 50]
            ys = [yy[b == i].mean() for i in range(10) if (b == i).sum() > 50]
            ece = sum((b == i).mean() * abs(pp[b == i].mean() - yy[b == i].mean()) for i in range(10) if (b == i).any())
            ax.plot(xs, ys, color=c, lw=1.6, marker="o", ms=3.5, label=f"{lab} (ECE {ece:.3f})")
            print(title, lab, "ECE", round(float(ece), 4), "mean p", round(float(pp.mean()), 4), "mean y", round(float(yy.mean()), 4))
        ax.set_title(title, fontsize=8.4, loc="left"); ax.set_xlabel("xác suất dự đoán"); ax.legend(frameon=False, fontsize=7.2, loc="upper left")
    axs[0].set_ylabel("tần suất dương thực tế")
    fig.tight_layout(); fig.savefig(FIG / "reliability.png"); plt.close(fig)


def fig_confusers():
    """Which attributes make non-matching images rank above the first exact match (ViT-L KD, PETA held out)."""
    from upar.retrieval import score_loglik
    A, gt, lg, gq, queries = _load("remote2/kd2_a08_e6_PETA")
    p = 1 / (1 + np.exp(-lg)); q_ids = np.unique(gq)
    S = score_loglik(queries[q_ids], p)
    cnt = np.zeros(len(A)); n = 0
    nwrong = []
    for a, q in enumerate(q_ids):
        o = np.argsort(-S[a], kind="stable")
        first = np.nonzero(gq[o] == q)[0][0]
        top = o[:min(first, 10)]                         # non-matching images ranked above the first match (at most 10)
        if len(top) == 0:
            continue
        diff = gt[top] != queries[q][None]
        cnt += diff.sum(0); n += len(top); nwrong += list(diff.sum(1))
    share = cnt / n
    names = [vi_attr(a) for a in A]
    k = np.argsort(-share)[:15]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.6, 1]})
    axs[0].barh(range(len(k))[::-1], share[k], color=BLUE, height=.62)
    axs[0].set_yticks(range(len(k))[::-1], [names[i] for i in k], fontsize=7)
    axs[0].set_xlabel("tỉ lệ ảnh sai ở đầu danh sách bị lệch thuộc tính này"); axs[0].grid(axis="y", visible=False)
    nw = np.array(nwrong)
    vals = [np.mean(nw == j) for j in range(1, 6)] + [np.mean(nw >= 6)]
    axs[1].bar(range(6), vals, color=AQUA, width=.6)
    axs[1].set_xticks(range(6), ["1", "2", "3", "4", "5", "≥6"]); axs[1].set_xlabel("số thuộc tính lệch")
    axs[1].set_ylabel("tỉ lệ ảnh"); axs[1].grid(axis="x", visible=False)
    for i, v in enumerate(vals):
        axs[1].text(i, v + .005, f"{v:.0%}", ha="center", fontsize=6.8)
    fig.tight_layout(); fig.savefig(FIG / "confusers.png"); plt.close(fig)
    print("images", n, "mean wrong", nw.mean().round(2), "top", [(names[i], round(float(share[i]), 3)) for i in k[:8]])


if __name__ == "__main__":
    import sys as _s
    todo = _s.argv[1:] or ["fig_ndom", "fig_auc_nll", "fig_temperature", "fig_kd", "fig_leaderboard", "fig_example", "fig_nll_groups", "fig_query_difficulty", "table_metrics"]
    for f in todo:
        globals()[f]()
