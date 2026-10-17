"""Figures of the T2 v12 deck, from the held-out fold logits (PETA held out = PETA val is the gallery) and logged numbers.

../../.venv/bin/python make_t2_figs.py [--pick] [--q QID] [--knn FEATS_NPZ]
io_t2.png       the top 8 of one PETA query, ranked by the log-likelihood of the pair H+ + CLIP (WiSE 0.7)
adm_t2.png      ADM of that ranking: degree of match per rank, exact matches marked, precision at each match
l1ll_t2.png     one confident error vs several small doubts: per-attribute penalty under L1 and log-likelihood
llgain_t2.png   mADM gain of log-likelihood over L1 on every run (logged in the T2 paper, Table 2)
members_t2.png  mADM fold mean (PETA, Market) of single members and pairs (remote_results/rare/t2_compare.py logic)
kd_t2.png       distillation and training length for the ViT-L 176 student (T2 paper, Table 5)
lb_t2.png       leaderboard history of the T2 submissions
knn_t2.png      (--knn) one PETA gallery image and its 5 nearest neighbours by the H+ embedding of the package
knngain_t2.png  (--knn) mADM with and without kNN smoothing, PETA and Market folds, H+ embedding, k 50, alpha 0.2
--pick          prints candidate queries
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

PAR = Path("/media/tan/F/RWS2027/Upar-Challenge-Track-2/par")
sys.path.insert(0, str(PAR))
from eval_retrieval import load_task2  # noqa: E402
from upar.data import DATA, attr_names, load_split  # noqa: E402
from upar.retrieval import retrieval_metrics, score_l1, score_loglik  # noqa: E402

OUT = Path(__file__).parent / "figs"; OUT.mkdir(exist_ok=True)
R = Path("/media/tan/F/RWS2027/remote_results/rare/runs")
INK, PAPER, MUTE, BLUE, ORANGE, LORANGE, GRID = "#14202E", "#F5F3EE", "#6B7684", "#2F7FB5", "#B85A1E", "#F0A868", "#D5DBE1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 20, "axes.edgecolor": MUTE, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER})
A = attr_names(); va = load_split("val")
ids, Q = load_task2("val"); qid_of = dict(zip(ids.image, ids.qid))
VI = {"Accessory-Glasses-Sun": "kính râm", "Accessory-Hat": "mũ", "Hair-Length-Bald": "hói", "Age-Old": "người già",
      "Age-Adult": "trưởng thành", "Age-Young": "trẻ", "Gender-Female": "nữ", "Hair-Length-Long": "tóc dài",
      "Hair-Length-Short": "tóc ngắn", "UpperBody-Length-Short": "áo ngắn tay", "LowerBody-Length-Short": "quần ngắn",
      "LowerBody-Type-Trousers&Shorts": "quần dài", "LowerBody-Type-Skirt&Dress": "váy", "Accessory-Backpack": "balo",
      "Accessory-Bag": "túi xách", "Accessory-Glasses-Normal": "kính thường"}
COL = {"Black": "đen", "Blue": "xanh lam", "Brown": "nâu", "Green": "xanh lá", "Grey": "xám", "Orange": "cam",
       "Pink": "hồng", "Purple": "tím", "Red": "đỏ", "White": "trắng", "Yellow": "vàng", "Other": "màu khác"}


def vi(a):
    if "-Color-" in a:
        return ("áo " if a.startswith("Upper") else "quần ") + COL[a.split("-")[-1]]
    return VI[a]


def load(name):
    run, _, wise = name.partition("@")
    d = np.load(R / run / (f"held_val_logits_{wise}.npz" if wise else "held_val_logits_last.npz"))
    return dict(zip(d["idx"], d["logits"].astype(np.float64)))


def fold(names_h, names_c, pre=""):
    """Gallery of one held-out fold: rows, labels, query ids, and the logit mean of H+ (one run) and CLIP (seed mean)."""
    zc = [load(pre + n) for n in names_c]; idx = np.array(sorted(zc[0]))
    z = np.mean([np.stack([c[i] for i in idx]) for c in zc], 0)
    if names_h:
        zh = load(pre + names_h); z = (np.stack([zh[i] for i in idx]) + z) / 2
    gt = va[A].values[idx].astype(np.float32); gq = np.array([qid_of[x] for x in va.image.values[idx]])
    return idx, gt, gq, z


def madm(z, gt, gq, score=score_loglik):
    qids = np.unique(gq); p = 1 / (1 + np.exp(-z))
    return retrieval_metrics(score(Q[qids], p), Q[qids], gt, gq, qids)["mADM"]


def img(row):
    return Image.open(DATA / va.image.values[row]).convert("RGB")


idx, gt, gq, z = fold("HB_s1", ["T_s1@wise0.70", "T_s2@wise0.70"])
p = 1 / (1 + np.exp(-z)); qids = np.unique(gq)
S_ll, S_l1 = score_loglik(Q[qids], p), score_l1(Q[qids], p)
eq = Q[qids] @ gt.T + (1 - Q[qids]) @ (1 - gt).T; dom = eq / 40
ndom = np.clip((dom - dom.mean(1, keepdims=True)) / np.maximum(dom.max(1, keepdims=True) - dom.mean(1, keepdims=True), 1e-9), 0, None)


def ranking(i, S):
    o = np.argsort(-S[i], kind="stable"); m = gq[o] == qids[i]; ranks = np.nonzero(m)[0] + 1
    cs = np.cumsum(ndom[i][o]); return o, m, ranks, float(np.mean(cs[ranks - 1] / ranks))


if "--pick" in sys.argv:
    for i in range(len(qids)):
        o, m, ranks, adm = ranking(i, S_ll); _, _, r1, adm1 = ranking(i, S_l1)
        if 3 <= len(ranks) <= 8 and ranks[0] <= 2 and m[:8].sum() >= 2 and (~m[:8]).sum() >= 2 and r1[0] > 3:
            print(qids[i], "R", len(ranks), "ranks LL", ranks[:5], "ADM %.2f" % adm, "| L1 first", r1[0], "ADM %.2f" % adm1,
                  "| pos", [vi(A[k]) for k in np.where(Q[qids[i]] == 1)[0]])
    sys.exit()

QID = int(sys.argv[sys.argv.index("--q") + 1])
i = int(np.where(qids == QID)[0][0])
o, m, ranks, adm = ranking(i, S_ll)
print("query", QID, [vi(A[k]) for k in np.where(Q[QID] == 1)[0]], "R", len(ranks), "ranks", ranks[:8], "ADM", round(adm, 4))

# ---- io: top 8 ----
fig, axs = plt.subplots(1, 8, figsize=(16, 5.4))
for r, ax in enumerate(axs):
    g = o[r]; ax.imshow(img(idx[g]).resize((96, 192))); ax.set_xticks([]); ax.set_yticks([])
    wrong = [vi(A[k]) for k in np.where(gt[g] != Q[QID])[0]]
    good = len(wrong) == 0
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(ORANGE if good else GRID); s.set_linewidth(5 if good else 2)
    ax.set_title(f"#{r + 1}", fontsize=22, color=INK, pad=8)
    ax.set_xlabel("khớp đúng" if good else f"sai {len(wrong)} thuộc tính", fontsize=18,
                  color=ORANGE if good else MUTE, labelpad=8)
    print(r + 1, va.image.values[idx[g]], "exact" if good else wrong)
fig.tight_layout(); fig.savefig(OUT / "io_t2.png", dpi=100); plt.close(fig)

# ---- ADM on this ranking ----
N = 30
fig, ax = plt.subplots(figsize=(14, 5.4))
nd = ndom[i][o][:N]; cs = np.cumsum(ndom[i][o])
ax.bar(np.arange(1, N + 1), nd, color=[ORANGE if mm else "#9DB3C6" for mm in m[:N]], width=0.75)
for r in ranks[ranks <= N]:
    ax.plot(r, cs[r - 1] / r, "D", ms=13, color=INK, zorder=4)
    if cs[r - 1] / r < 0.999 or r == 2:
        ax.annotate(f"{cs[r - 1] / r:.2f}", (r, cs[r - 1] / r), xytext=(0, 14), textcoords="offset points", ha="center",
                    fontsize=17, color=INK)
ax.set_xlim(0.3, N + 0.7); ax.set_ylim(0, 1.25); ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
ax.set_xlabel("Thứ hạng trong danh sách trả về"); ax.set_ylabel("Mức khớp")
ax.text(N + 0.3, 1.16, f"ADM của truy vấn = {adm:.2f}", ha="right", fontsize=20, color=INK)
fig.tight_layout(); fig.savefig(OUT / "adm_t2.png", dpi=100); plt.close(fig)
print("ADM points", [(int(r), round(float(cs[r - 1] / r), 3)) for r in ranks[:6]])

# ---- L1 vs log-likelihood: a query where L1 puts a confidently wrong image above every exact match ----
best = None
for qi in range(len(qids)):
    ol1 = np.argsort(-S_l1[qi], kind="stable"); oll = np.argsort(-S_ll[qi], kind="stable")
    ex = np.where(gq == qids[qi])[0]
    if len(ex) == 0 or gq[oll[0]] != qids[qi] or gq[ol1[0]] == qids[qi]:
        continue
    w = ol1[0]; e = oll[0]
    pen_w = np.abs(Q[qids[qi]] - p[w]); pen_e = np.abs(Q[qids[qi]] - p[e])
    if pen_w.max() > 0.9 and pen_e.max() < 0.6 and (gt[w] != Q[qids[qi]]).sum() == 1:
        gap = S_l1[qi, w] - S_l1[qi, e]
        if best is None or gap > best[0]:
            best = (gap, qi, w, e)
_, qi, w, e = best
q = Q[qids[qi]]
eps = 1e-4
l1w, l1e = np.abs(q - p[w]), np.abs(q - p[e])
llw = -(q * np.log(np.clip(p[w], eps, 1)) + (1 - q) * np.log(np.clip(1 - p[w], eps, 1)))
lle = -(q * np.log(np.clip(p[e], eps, 1)) + (1 - q) * np.log(np.clip(1 - p[e], eps, 1)))
top = np.argsort(-(llw + lle))[:6]
fig, axs = plt.subplots(1, 2, figsize=(14.4, 6.0), sharey=True)
for ax, (a, b, title) in zip(axs, ((l1w, l1e, "L1: tổng |nhãn − xác suất|"), (llw, lle, "Log-likelihood: tổng −log xác suất"))):
    y = np.arange(len(top)); h = 0.38
    ax.barh(y - h / 2, a[top], h, color=MUTE, label="ảnh A, sai 1 thuộc tính")
    ax.barh(y + h / 2, b[top], h, color=ORANGE, label="ảnh B, khớp đúng")
    ax.set_yticks(y); ax.set_yticklabels([vi(A[k]) for k in top]); ax.invert_yaxis()
    ax.set_title(f"{title}\ntổng của A = {a.sum():.2f}, của B = {b.sum():.2f}", fontsize=20, color=INK, loc="left")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
h0, l0 = axs[0].get_legend_handles_labels(); fig.legend(h0, l0, frameon=False, fontsize=18, ncol=2, loc="lower center")
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(OUT / "l1ll_t2.png", dpi=100); plt.close(fig)
print("L1/LL example query", qids[qi], "A", va.image.values[idx[w]], "wrong:", [vi(A[k]) for k in np.where(gt[w] != q)[0]],
      "p_wrong", [round(float(p[w][k]), 3) for k in np.where(gt[w] != q)[0]],
      "| B", va.image.values[idx[e]], "L1 A %.2f B %.2f | LL A %.2f B %.2f" % (l1w.sum(), l1e.sum(), llw.sum(), lle.sum()),
      "| rank of B under L1", int(np.where(np.argsort(-S_l1[qi], kind="stable") == e)[0][0]) + 1)
for tag, row in (("A", idx[w]), ("B", idx[e])):
    img(row).save(OUT / f"l1ll_{tag}.png")

# ---- LL gain over L1 on every run (T2 paper Table 2) ----
fig, ax = plt.subplots(figsize=(12, 5))
rows = [("PETA", 5.23, 3.74, 6.40, 41), ("Market1501", 4.94, 4.56, 5.35, 5), ("PA100k", 8.63, 8.16, 9.10, 2)]
for j, (n, mval, lo, hi, k) in enumerate(rows[::-1]):
    ax.plot([lo, hi], [j, j], color=LORANGE, lw=10, solid_capstyle="round"); ax.plot(mval, j, "o", ms=20, color=ORANGE)
    ax.text(hi + 0.25, j, f"+{mval:.2f} điểm, {k} lần chạy", va="center", fontsize=20, color=INK)
ax.set_yticks(range(3)); ax.set_yticklabels([r[0] for r in rows[::-1]]); ax.set_xlim(0, 13)
ax.axvline(0, color=MUTE, lw=2); ax.set_xlabel("mADM của log-likelihood trừ mADM của L1 (điểm)")
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
fig.tight_layout(); fig.savefig(OUT / "llgain_t2.png", dpi=100); plt.close(fig)

# ---- members and pairs, fold mean ----
cfg = [("CLIP gốc", None, ["B_s1", "B_s2"]), ("CLIP + nhánh văn bản", None, ["T_s1", "T_s2"]),
       ("CLIP + nhánh văn bản + WiSE", None, ["T_s1@wise0.70", "T_s2@wise0.70"]), ("H+", None, ["HB_s1"]),
       ("H+ + CLIP gốc", "HB_s1", ["B_s1", "B_s2"]), ("H+ + CLIP + nhánh văn bản", "HB_s1", ["T_s1", "T_s2"]),
       ("H+ + CLIP + nhánh văn bản + WiSE", "HB_s1", ["T_s1@wise0.70", "T_s2@wise0.70"])]
vals = []
for name, h, cs_ in cfg:
    fm = []
    for pre in ("", "M"):
        per = [madm(f[3], f[1], f[2]) for f in (fold(h, [c], pre) for c in cs_)]   # seeds scored separately
        fm.append(np.mean(per))
    vals.append(np.mean(fm)); print(f"{name:34s} PETA {fm[0]:.4f} Market {fm[1]:.4f} mean {np.mean(fm):.4f}")
fig, ax = plt.subplots(figsize=(14, 6))
for j, ((name, h, _), v) in enumerate(list(zip(cfg, vals))[::-1]):
    c = ORANGE if name.endswith("WiSE") and h else (BLUE if h else "#9DB3C6")
    ax.plot([0.68, v], [j, j], color=GRID, lw=6, solid_capstyle="round"); ax.plot(v, j, "o", ms=18, color=c)
    ax.text(v + 0.0012, j, f"{v:.4f}", va="center", fontsize=19, color=INK)
ax.set_yticks(range(len(cfg))); ax.set_yticklabels([c[0] for c in cfg][::-1], fontsize=19)
ax.set_xlim(0.68, 0.735); ax.set_xlabel("mADM, trung bình hai fold (PETA, Market)")
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
fig.tight_layout(); fig.savefig(OUT / "members_t2.png", dpi=100); plt.close(fig)

# ---- distillation and training length (T2 paper Table 5, ViT-L 176 student, PETA held out) ----
fig, ax = plt.subplots(figsize=(12, 5.2))
ax.plot([3, 5], [65.15, 64.96], "o-", ms=14, lw=4, color=MUTE, label="chỉ học từ nhãn")
ax.plot([3, 6, 10], [65.55, 66.35, 66.81], "o-", ms=14, lw=4, color=ORANGE, label="học thêm từ DINOv3-L và CLIP-L")
for x, y in ((3, 65.15), (5, 64.96)):
    ax.annotate(f"{y:.2f}", (x, y), xytext=(0, -30), textcoords="offset points", ha="center", fontsize=18, color=MUTE)
for x, y in ((3, 65.55), (6, 66.35), (10, 66.81)):
    ax.annotate(f"{y:.2f}", (x, y), xytext=(0, 14), textcoords="offset points", ha="center", fontsize=18, color=ORANGE)
ax.set_xticks([3, 5, 6, 10]); ax.set_xlabel("Số epoch huấn luyện"); ax.set_ylabel("mADM trên PETA (điểm)")
ax.set_ylim(64.5, 67.3); ax.legend(frameon=False, fontsize=18, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "kd_t2.png", dpi=100); plt.close(fig)

# ---- leaderboard history ----
lb = [("v3b", 0.6247, "ViT-B, log-likelihood,\ngroup CE"), ("v5", 0.6315, "+ chưng cất\n10 epoch"),
      ("v6", 0.6365, "+ ảnh trộn,\n20 epoch"), ("v10", 0.6722, "GPU: student\n+ H+ + CLIP"),
      ("v12", 0.6783, "+ nhánh văn bản,\nWiSE, kNN")]
fig, ax = plt.subplots(figsize=(15, 5.6))
x = np.arange(len(lb))
ax.plot(x, [v for _, v, _ in lb], "-", color=GRID, lw=5, zorder=1)
ax.scatter(x, [v for _, v, _ in lb], s=[260] * 4 + [420], color=[BLUE] * 4 + [ORANGE], zorder=3)
for j, (n, v, d) in enumerate(lb):
    ax.annotate(f"{v:.4f}", (j, v), xytext=(0, 16), textcoords="offset points", ha="center", fontsize=19,
                color=INK, fontweight="bold" if n == "v12" else "normal")
ax.axhline(0.6718, color=MUTE, lw=2, ls="--"); ax.text(1.0, 0.6733, "đội hạng 2 hiện tại: 0.6718", fontsize=17, color=MUTE)
ax.set_xticks(x); ax.set_xticklabels([f"{n}\n{d}" for n, _, d in lb], fontsize=16)
ax.set_ylim(0.615, 0.69); ax.set_ylabel("mADM trên leaderboard")
fig.tight_layout(); fig.savefig(OUT / "lb_t2.png", dpi=100); plt.close(fig)

# ---- kNN smoothing with the package's H+ embedding ----
if "--knn" in sys.argv:
    F = np.load(sys.argv[sys.argv.index("--knn") + 1])
    fr = dict(zip(F["rows"], range(len(F["rows"])))); feats = F["feats"].astype(np.float32)

    def smooth(z, rows, k=50, a=0.2, chunk=1024):
        """kNN smoothing as in the package, in row chunks (the laptop has little RAM); also each row's top-5 neighbours."""
        f = feats[[fr[r] for r in rows]]; nn = np.empty((len(f), k), int); top5 = np.empty((len(f), 5), int)
        for c0 in range(0, len(f), chunk):
            sc = f[c0:c0 + chunk] @ f.T; sc[np.arange(len(sc)), np.arange(c0, c0 + len(sc))] = -np.inf
            nn[c0:c0 + len(sc)] = np.argpartition(-sc, k, axis=1)[:, :k]
            t = np.argpartition(-sc, 5, axis=1)[:, :5]
            top5[c0:c0 + len(sc)] = np.take_along_axis(t, np.argsort(-np.take_along_axis(sc, t, 1), 1), 1)
        return (1 - a) * z + a * z[nn].mean(1), f, top5

    res = []
    for pre, name in (("", "PETA"), ("M", "Market1501")):
        idx_f, gt_f, gq_f, z_f = fold("HB_s1", ["T_s1@wise0.70"], pre)
        base = madm(z_f, gt_f, gq_f); zs, f, top5 = smooth(z_f, idx_f); sm = madm(zs, gt_f, gq_f)
        same = (gt_f[top5[:, 0]] == gt_f).all(1).mean()
        res.append((name, base, sm, same)); print(f"kNN {name}: {base:.4f} -> {sm:.4f} (+{sm - base:.4f}); nearest neighbour same 40 labels {same:.0%}")
        if name == "PETA":
            # one gallery image whose neighbours mostly share its labels, to show what the smoothing averages over
            ok5 = (gt_f[top5] == gt_f[:, None]).all(2).sum(1)
            cand = np.where((ok5 >= 3) & (ok5 <= 4))[0]
            j0 = cand[len(cand) // 3]; nb = top5[j0]; sim = f[nb] @ f[j0]
            fig, axs = plt.subplots(1, 6, figsize=(14, 5.2))
            for ax, g, lab in zip(axs, [j0] + list(nb), ["ảnh gốc"] + [f"{v:.2f}" for v in sim]):
                ax.imshow(img(idx_f[g]).resize((96, 192))); ax.set_xticks([]); ax.set_yticks([])
                same_g = (gt_f[g] == gt_f[j0]).all()
                for sp in ax.spines.values():
                    sp.set_visible(True); sp.set_color(ORANGE if g == j0 else (BLUE if same_g else GRID)); sp.set_linewidth(5 if g == j0 or same_g else 2)
                ax.set_xlabel(lab, fontsize=19, color=INK, labelpad=8)
            fig.tight_layout(); fig.savefig(OUT / "knn_t2.png", dpi=100); plt.close(fig)
            print("kNN example", va.image.values[idx_f[j0]], "neighbours", [va.image.values[idx_f[g]] for g in nb],
                  "same labels", [bool((gt_f[g] == gt_f[j0]).all()) for g in nb], "sim", np.round(sim, 3),
                  "wrong attrs", [[vi(A[k]) for k in np.where(gt_f[g] != gt_f[j0])[0]] for g in nb])
    fig, ax = plt.subplots(figsize=(10, 5))
    for j, (n, b, s_, _) in enumerate(res):
        ax.bar(j - 0.18, b, 0.34, color="#9DB3C6"); ax.bar(j + 0.18, s_, 0.34, color=ORANGE)
        ax.text(j - 0.18, b + 0.002, f"{b:.4f}", ha="center", fontsize=17); ax.text(j + 0.18, s_ + 0.002, f"{s_:.4f}", ha="center", fontsize=17)
    ax.set_xticks(range(len(res))); ax.set_xticklabels([r[0] for r in res]); ax.set_ylim(0.6, max(r[2] for r in res) + 0.03)
    ax.set_ylabel("mADM của fold"); ax.legend(["không làm mượt", "làm mượt kNN"], frameon=False, fontsize=17, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT / "knngain_t2.png", dpi=100); plt.close(fig)
print("ok")
