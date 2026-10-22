"""Figures of the Track 2 paper, from saved held-out logits of the fold models (remote_results/rare/runs, the student of
Upar-Challenge-Track-2/runs/remote2), the appearance features of the submitted DINOv3 model (v12_slides/data/t2_feats.npz)
and t2_numbers.json. No model is run.

../../../.venv/bin/python make_figs.py [--lang vi|en] [--pick]
figs/arch_{lang}.pdf   (a) the three models on one PETA image, (b) kNN smoothing over its neighbours and the likelihood
                       score, (c) one query head. Logits are those of the PETA-fold models (they never saw PETA);
                       neighbours come from the submitted DINOv3 model, which was trained on these validation images.
figs/l1ll_{lang}.pdf   top five of one PETA query under the L1 distance and under the likelihood score (fold ensemble)
figs/calib_{lang}.pdf  per-attribute Platt scale and bias fitted on held-out labels vs on the source datasets, both folds
--pick                 print candidate queries for l1ll
Style: DeepSeek-like blocks (Track 1 paper); Okabe-Ito colors plus marker shapes (scientific-visualization skill).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, "/media/tan/F/RWS2027/Upar-Challenge-Track-2/par")
from eval_retrieval import load_task2  # noqa: E402
from upar.data import DATA, attr_names, load_split  # noqa: E402
from upar.retrieval import score_l1, score_loglik  # noqa: E402

LANG = sys.argv[sys.argv.index("--lang") + 1] if "--lang" in sys.argv else "vi"
HERE = Path(__file__).parent
OUT = HERE.parent / "figs"; OUT.mkdir(exist_ok=True)
R = Path("/media/tan/F/RWS2027/remote_results/rare/runs")
STU = Path("/media/tan/F/RWS2027/Upar-Challenge-Track-2/runs/remote2/kd2_a08_e10_PETA/held_val_logits_last.npz")
FEATS = Path("/media/tan/F/RWS2027/Upar-Challenge-Track-2/report/v12_slides/data/t2_feats.npz")
A = attr_names(); va = load_split("val"); ids, Q = load_task2("val"); qid_of = dict(zip(ids.image, ids.qid))

HP, CL, ST, INK, MUTE = "#1F5BD8", "#D2731A", "#5A8F3C", "#18212D", "#626E7D"
GREEN, VERM, BLUE_OI = "#009E73", "#D55E00", "#0072B2"          # Okabe-Ito
YEL, GRY, LINE = "#F8E6BC", "#E1E6EB", "#2B3A4E"
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 7.5, "pdf.fonttype": 42, "mathtext.fontset": "stixsans",
                     "text.color": INK, "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
T = {  # all figure text, Vietnamese and English
    "vi": dict(a="(a) mỗi ảnh trong gallery", b="(b) trên cả gallery", c="(c) một query head", module="module",
               op="phép tính", frozen="đóng băng", blocks="block", student="student", text="+ điểm văn bản",
               qh="query head", mean="trung bình", logit="logit “người già”", appear="đặc trưng ngoại hình",
               meanpatch="trung bình patch token", l2="chuẩn hoá L2", nb="5 trong 50 láng giềng gần nhất (cosine)",
               own="ảnh g", smooth="kNN smoothing (Pt. 5)", score="điểm likelihood (Pt. 4)", query="truy vấn q (40 bit)",
               rank="hạng của g với truy vấn của nó", label="nhãn: không", queries="40 attribute query",
               tokens="CLS + patch token", cls="40 bộ phân loại", cls_sub="theo thuộc tính", heads="8 head",
               mlp_sub="độ rộng ẩn ×2", logits="40 logit", width="một lớp · độ rộng 1024 (DINOv3: 1280)",
               l1="khoảng cách L1", ll="điểm likelihood", exact="khớp đúng", wrong="sai {n}", rankword="hạng",
               scale="hệ số scale $a_k$", bias="bias $b_k$", heldfit="khớp trên dataset giữ lại (oracle)",
               srcfit="khớp trên dataset nguồn", peta="fold PETA", market="fold Market"),
    "en": dict(a="(a) each gallery image", b="(b) over the gallery", c="(c) one query head", module="module",
               op="operation", frozen="frozen", blocks="blocks", student="student", text="+ text score",
               qh="query head", mean="mean", logit="“old” logit", appear="appearance feature",
               meanpatch="mean of patch tokens", l2="L2 normalization", nb="5 of the 50 nearest neighbors (cosine)",
               own="image g", smooth="kNN smoothing (Eq. 5)", score="likelihood score (Eq. 4)", query="query q (40 bits)",
               rank="rank of g for its own query", label="label: no", queries="40 attribute queries",
               tokens="CLS + patch tokens", cls="40 classifiers", cls_sub="one per attribute", heads="8 heads",
               mlp_sub="hidden width ×2", logits="40 logits", width="one layer · width 1024 (DINOv3: 1280)",
               l1="L1 distance", ll="likelihood score", exact="exact", wrong="{n} wrong", rankword="rank",
               scale="scale $a_k$", bias="bias $b_k$", heldfit="fitted on the held-out dataset (oracle)",
               srcfit="fitted on the source datasets", peta="PETA fold", market="Market fold"),
}[LANG]


def npz(f):
    d = np.load(f)
    return dict(zip(d["idx"], d["logits"].astype(np.float64)))


def photo(row, w=112, h=224):
    return np.asarray(Image.open(DATA / va.image.values[row]).convert("RGB").resize((w * 4, h * 4), Image.LANCZOS), float) / 255


def blk(ax, x0, y0, x1, y1, text, sub=None, kind="main", color=INK, bold=False, fs=7.0):
    fc = {"main": YEL, "op": GRY, "frozen": "white"}[kind]
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=4", fc=fc, ec=LINE, lw=.75, zorder=2))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if sub:
        ax.text(cx, cy + 4.4, text, ha="center", va="center", fontsize=fs, color=color, weight="bold" if bold else "normal", zorder=3)
        ax.text(cx, cy - 5.2, sub, ha="center", va="center", fontsize=6.2, color=MUTE, zorder=3)
    else:
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=color, weight="bold" if bold else "normal", zorder=3)


def stack(ax, cx, cy, fills, layers=3):
    n = len(fills); w, h = n * 8.4 + 4, 10
    for L in range(layers - 1, -1, -1):
        x0, y0 = cx - w / 2 - 1.8 * L, cy - h / 2 + 1.8 * L
        ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=3", fc="white", ec=LINE, lw=.6, zorder=2))
        for i in range(n):
            ax.add_patch(plt.Circle((x0 + 6.2 + 8.4 * i, y0 + h / 2), 2.9, fc=fills[i] if L == 0 and fills[i] else "white",
                                    ec=LINE, lw=.55, zorder=3))
    return cx - w / 2 - 3.6, cy - h / 2, cx + w / 2, cy + h / 2 + 3.6


def path(ax, pts, dashed=False, color=LINE, lw=.8):
    xs, ys = zip(*pts)
    ls = (0, (3, 2)) if dashed else "-"
    if len(pts) > 2:
        ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, ls=ls, solid_capstyle="butt", zorder=1)
    ax.annotate("", xy=pts[-1], xytext=pts[-2], zorder=1, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls,
                                                                         shrinkA=0, shrinkB=0, mutation_scale=6))


def plus(ax, cx, cy, r=5):
    ax.add_patch(plt.Circle((cx, cy), r, fc="white", ec=LINE, lw=.8, zorder=3))
    ax.plot([cx - r, cx + r], [cy, cy], color=LINE, lw=.8, zorder=4); ax.plot([cx, cx], [cy - r, cy + r], color=LINE, lw=.8, zorder=4)


def lock(ax, cx, cy, s=1.0):
    ax.add_patch(Rectangle((cx - 3.2 * s, cy - 2.6 * s), 6.4 * s, 5.0 * s, fc=MUTE, ec="none", zorder=4))
    ax.add_patch(matplotlib.patches.Arc((cx, cy + 2.4 * s), 4.2 * s, 6.0 * s, theta1=0, theta2=180, color=MUTE, lw=1.0, zorder=4))


def gallery_peta():
    """PETA validation gallery: fold logits of the three models (CLIP seed 1) and the submitted model's neighbours."""
    hb = npz(R / "HB_s1/held_val_logits_last.npz"); tw = npz(R / "T_s1/held_val_logits_wise0.70.npz"); st = npz(STU)
    F = np.load(FEATS, allow_pickle=True); keep = F["ds"] == "PETA"
    rows = F["rows"][keep]; E = F["feats"][keep].astype(np.float32)
    idx = np.array([r for r in rows if r in hb and r in tw and r in st]); pos = {r: i for i, r in enumerate(rows)}
    E = E[[pos[r] for r in idx]]; E /= np.linalg.norm(E, axis=1, keepdims=True)
    Zm = {"h": np.stack([hb[r] for r in idx]), "c": np.stack([tw[r] for r in idx]), "s": np.stack([st[r] for r in idx])}
    return idx, E, Zm


# ================================================================ Figure 1
def fig_arch(row=31758, attr="Age-Old"):
    idx, E, Zm = gallery_peta()
    k = A.index(attr); g = int(np.where(idx == row)[0][0])
    Z = (Zm["h"] + Zm["c"] + Zm["s"]) / 3
    SS = E @ E.T; np.fill_diagonal(SS, -np.inf)               # kNN smoothing of the whole gallery, Eq. 4
    NN = np.argpartition(-SS, 50, axis=1)[:, :50]
    Zs = 0.8 * Z + 0.2 * Z[NN].mean(1)
    S = SS[g]; nn = np.argsort(-S)[:50]
    gt = va[A].values[idx]; q = Q[qid_of[va.image.values[row]]][None]
    P0, P1 = 1 / (1 + np.exp(-Z)), 1 / (1 + np.exp(-Zs))
    s0, s1 = score_loglik(q, P0)[0], score_loglik(q, P1)[0]
    r0, r1 = int((s0 > s0[g]).sum() + 1), int((s1 > s1[g]).sum() + 1)
    print("arch example", row, attr, "label", gt[g, k], "logits h/c/s", [round(Zm[m][g, k], 2) for m in "hcs"],
          "mean %.2f -> %.2f" % (Z[g, k], Zs[g, k]), "rank", r0, "->", r1, "nn sims", np.round(S[nn[:5]], 2))
    fig = plt.figure(figsize=(7.0, 3.05))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 700); ax.set_ylim(0, 305); ax.axis("off")
    S1, S2 = 340, 516
    for x, t in ((3, T["a"]), (S1 + 6, T["b"]), (S2 + 6, T["c"])):
        ax.text(x, 300, t, fontsize=8.3, weight="bold", va="top")
    for x in (S1, S2):
        ax.plot([x, x], [6, 296], color="#D5DAE0", lw=.6)
    lx, ly = 150, 297
    for i, (kind, lab) in enumerate([("main", T["module"]), ("op", T["op"]), ("frozen", T["frozen"])]):
        x = lx + i * 62
        ax.add_patch(FancyBboxPatch((x, ly - 4.5), 14, 9, boxstyle="round,pad=0,rounding_size=2",
                                    fc={"main": YEL, "op": GRY, "frozen": "white"}[kind], ec=LINE, lw=.6, zorder=2))
        if kind == "frozen":
            lock(ax, x + 7, ly - .3, .7)
        ax.text(x + 18, ly, lab, fontsize=6.6, va="center", color=MUTE)
    # ---------- (a) three models on image g
    ax.imshow(photo(row), extent=(4, 40, 112, 184), zorder=2, interpolation="none")
    ax.add_patch(Rectangle((4, 112), 36, 72, fill=False, ec=LINE, lw=.6, zorder=3))
    ax.text(22, 106, T["own"], ha="center", va="top", fontsize=7)
    rows3 = [(246, "DINOv3 ViT-H+/16", "224×112 · 24 " + T["blocks"], HP, "h", T["qh"]),
             (148, "CLIP ViT-L/14", "16 " + T["blocks"] + " · WiSE-FT", CL, "c", T["qh"]),
             (50, T["student"] + " ViT-L/16", "176×96 · 16 " + T["blocks"], ST, "s", T["qh"])]
    LX, AVX = 284, 318
    for (cy, name, sub, col, key, qh) in rows3:
        ax.annotate("", xy=(52, cy), xytext=(40, 148 + (cy - 148) * .25), zorder=1,
                    arrowprops=dict(arrowstyle="-|>", color=LINE, lw=.8, mutation_scale=6, shrinkA=0, shrinkB=0,
                                    connectionstyle="arc3,rad=%.2f" % (-.25 if cy > 148 else .25 if cy < 148 else 0)))
        blk(ax, 52, cy - 14, 82, cy + 14, "", kind="frozen"); lock(ax, 67, cy + 5)
        ax.text(67, cy - 6.5, "8", ha="center", va="center", fontsize=6.3, color=MUTE, zorder=3)
        blk(ax, 82, cy - 14, 186, cy + 14, name, sub, color=col, bold=True, fs=6.9)
        blk(ax, 196, cy - 12, 260, cy + 12, qh, T["text"] if key == "c" else None, color=col, bold=True, fs=6.8)
        path(ax, [(186, cy), (196, cy)], color=col)
        ax.add_patch(plt.Circle((LX, cy), 11.5, fc="white", ec=col, lw=1.0, zorder=3))
        ax.text(LX, cy, f"{Zm[key][g, k]:.2f}", ha="center", va="center", fontsize=6.0, zorder=4)
        path(ax, [(260, cy), (LX - 11.5, cy)], color=col)
    ax.text(LX, 263, T["logit"], ha="center", va="bottom", fontsize=6.4, color=MUTE)
    ax.add_patch(plt.Circle((AVX, 148), 9, fc=GRY, ec=LINE, lw=.75, zorder=3))
    ax.text(AVX, 148, "avg", ha="center", va="center", fontsize=6.0, zorder=4)
    path(ax, [(LX + 11.5, 246), (AVX, 246), (AVX, 157)]); path(ax, [(LX + 11.5, 148), (AVX - 9, 148)])
    path(ax, [(LX + 11.5, 50), (AVX, 50), (AVX, 139)])
    ax.text(AVX + 4, 133, f"$z_g$ = {Z[g, k]:.2f}", ha="left", va="top", fontsize=6.6)
    # appearance feature from the DINOv3 ViT-H+ tokens
    blk(ax, 196, 196, 306, 222, T["meanpatch"], T["l2"] + r" → $e_g$", kind="op", fs=6.4)
    path(ax, [(134, 232), (134, 209), (196, 209)], color=HP)
    path(ax, [(306, 209), (S1 + 10, 209)], color=HP)
    path(ax, [(AVX + 9, 148), (S1 + 14, 148)])
    # ---------- (b) neighbours, smoothing, likelihood score
    tw_, th_ = 24, 48; x0 = S1 + 10
    for j, n in enumerate([g] + list(nn[:5])):
        x = x0 + j * (tw_ + 3)
        ax.imshow(photo(idx[n], 56, 112), extent=(x, x + tw_, 192, 192 + th_), zorder=2, interpolation="none")
        ax.add_patch(Rectangle((x, 192), tw_, th_, fill=False, ec=INK if j == 0 else HP, lw=1.1 if j == 0 else .7, zorder=3))
        ax.text(x + tw_ / 2, 187, "g" if j == 0 else f"{S[n]:.2f}", ha="center", va="top", fontsize=6.2,
                color=INK if j == 0 else MUTE, style="italic" if j == 0 else "normal")
    ax.text(x0 + 3 * (tw_ + 3) - 2, 256, T["nb"], ha="center", va="bottom", fontsize=6.2, color=MUTE)
    blk(ax, S1 + 14, 136, S2 - 8, 162, T["smooth"], None, kind="op", fs=6.8)
    ax.text((S1 + S2) / 2 + 3, 128, r"$\tilde{z}_g = 0.8\,z_g + 0.2\,\overline{z}_{\mathcal{N}(g)}$", ha="center", va="top", fontsize=7.2)
    path(ax, [((S1 + S2) / 2 + 3, 176), ((S1 + S2) / 2 + 3, 162)], color=HP)
    ax.text((S1 + S2) / 2 + 3, 106, f"{T['logit']}: {Z[g, k]:.2f} → {Zs[g, k]:.2f}   ({T['label']})", ha="center",
            va="top", fontsize=6.4)
    blk(ax, S1 + 14, 52, S2 - 8, 78, T["score"], None, kind="op", fs=6.8)
    path(ax, [((S1 + S2) / 2 + 3, 92), ((S1 + S2) / 2 + 3, 78)])
    qb = q[0]; qx0 = S1 + 16
    for i in range(40):
        ax.add_patch(Rectangle((qx0 + i * 3.6, 26), 3.0, 8, fc=INK if qb[i] else "white", ec=LINE, lw=.3, zorder=2))
    ax.text(qx0, 38, T["query"], fontsize=6.2, color=MUTE, va="bottom")
    path(ax, [((S1 + S2) / 2 + 3, 35), ((S1 + S2) / 2 + 3, 52)])
    ax.text((S1 + S2) / 2 + 3, 16, f"{T['rank']}: {r0} → {r1}", ha="center", va="top", fontsize=6.4)
    # ---------- (c) one query head (as in the Track 1 paper, narrower)
    xq, xm, xs, xt = 546, 588, 642, 684
    q0 = stack(ax, xq, 34, ["#F0A868", "#8FC1E3", "#C9B6E4", "#9FD3B5"])
    ax.text(xq - 18, 18, T["queries"], ha="left", va="center", fontsize=6.0, color=MUTE)
    t0 = stack(ax, xt, 34, ["#8A96A6", None, None])
    ax.text(698, 9, T["tokens"], ha="right", va="center", fontsize=6.0, color=MUTE)
    blk(ax, xm - 15, 27, xm + 15, 41, "LN", kind="op", fs=6.3)
    path(ax, [(q0[2] + 1, 34), (xm - 15, 34)]); path(ax, [(xm + 15, 34), (xs, 34), (xs, 80)])
    ax.text(xs + 3, 58, "Q", fontsize=7, weight="bold", va="center")
    blk(ax, xt - 13, 52, xt + 13, 66, "LN", kind="op", fs=6.3)
    path(ax, [(xt, t0[3]), (xt, 52)]); path(ax, [(xt, 66), (xt, 93), (xs + 38, 93)])
    ax.text(xs + 40, 97, "K, V", fontsize=6.8, weight="bold", va="bottom")
    blk(ax, xs - 38, 80, xs + 38, 106, "cross-attention", T["heads"], fs=6.4)
    plus(ax, xq, 122); path(ax, [(xq, q0[3]), (xq, 117)]); path(ax, [(xs, 106), (xs, 122), (xq + 5, 122)])
    path(ax, [(xq, 127), (xq, 167)]); path(ax, [(xq, 143), (xm - 15, 143)])
    blk(ax, xm - 15, 136, xm + 15, 150, "LN", kind="op", fs=6.3)
    path(ax, [(xm + 15, 143), (xs - 38, 143)])
    blk(ax, xs - 38, 133, xs + 38, 153, "MLP", T["mlp_sub"], fs=6.4)
    plus(ax, xq, 172); path(ax, [(xs, 153), (xs, 172), (xq + 5, 172)])
    path(ax, [(xq, 177), (xq, 198), (xm - 15, 198)])
    blk(ax, xm - 15, 191, xm + 15, 205, "LN", kind="op", fs=6.3)
    path(ax, [(xm + 15, 198), (xs - 38, 198)])
    blk(ax, xs - 38, 187, xs + 38, 211, T["cls"], T["cls_sub"], fs=6.4)
    z0 = stack(ax, xs, 236, ["#8FC1E3"] * 4)
    path(ax, [(xs, 211), (xs, 230.5)])
    ax.text(z0[0] - 3, 236, T["logits"], ha="right", va="center", fontsize=6.2, color=MUTE)
    ax.text(698, 272, T["width"], ha="right", va="center", fontsize=6.0, color=MUTE)
    fig.savefig(OUT / f"arch_{LANG}.pdf"); fig.savefig(OUT / f"arch_{LANG}.png", dpi=250); plt.close(fig)


# ================================================================ Figure 2: L1 vs likelihood on one query
def fold_peta_pair():
    hb = npz(R / "HB_s1/held_val_logits_last.npz")
    tws = [npz(R / f"T_s{s}/held_val_logits_wise0.70.npz") for s in (1, 2)]
    idx = np.array(sorted(set(hb) & set(tws[0]) & set(tws[1])))
    z = np.stack([(hb[r] + (tws[0][r] + tws[1][r]) / 2) / 2 for r in idx])
    gt = va[A].values[idx].astype(np.float32); gq = np.array([qid_of[x] for x in va.image.values[idx]])
    return idx, z, gt, gq


def fig_l1ll(qid=None):
    idx, z, gt, gq = fold_peta_pair(); p = 1 / (1 + np.exp(-z)); qids = np.unique(gq)
    S_ll, S_l1 = score_loglik(Q[qids], p), score_l1(Q[qids], p)
    if qid is None:
        for i in range(len(qids)):
            m = gq == qids[i]
            if not 2 <= m.sum() <= 6:
                continue
            o_ll, o_l1 = np.argsort(-S_ll[i], kind="stable"), np.argsort(-S_l1[i], kind="stable")
            m_ll, m_l1 = m[o_ll][:5], m[o_l1][:5]
            if m_ll[:2].all() and m_l1.sum() == 0:
                wrong = [int((gt[j] != Q[qids[i]]).sum()) for j in o_l1[:5]]
                print("candidate query", qids[i], "matches", m.sum(), "L1 top5 wrong", wrong,
                      "LL first ranks", np.nonzero(m[o_ll])[0][:4] + 1, "L1 first", np.nonzero(m[o_l1])[0][0] + 1)
        return
    i = int(np.where(qids == qid)[0][0]); qv = Q[qid]
    short = {"vi": {"Gender-Female": "nữ", "Hair-Length-Short": "tóc ngắn", "Hair-Length-Long": "tóc dài",
                    "UpperBody-Color-Red": "áo đỏ", "Accessory-Hat": "mũ"},
             "en": {"Gender-Female": "female", "Hair-Length-Short": "short hair", "Hair-Length-Long": "long hair",
                    "UpperBody-Color-Red": "red top", "Accessory-Hat": "hat"}}[LANG]
    fig, axs = plt.subplots(2, 5, figsize=(3.35, 2.6), gridspec_kw=dict(wspace=.07, hspace=.62, left=.01, right=.99, top=.9, bottom=.1))
    for r, (S, name) in enumerate(((S_l1, T["l1"]), (S_ll, T["ll"]))):
        o = np.argsort(-S[i], kind="stable")
        axs[r, 0].text(0, 1.06, name, transform=axs[r, 0].transAxes, fontsize=7.3, weight="bold", va="bottom")
        for c in range(5):
            ax = axs[r, c]; j = o[c]
            ax.imshow(photo(idx[j], 56, 112), interpolation="none"); ax.set_xticks([]); ax.set_yticks([])
            w = np.where(gt[j] != qv)[0]; ok = len(w) == 0
            for s in ax.spines.values():
                s.set_color(GREEN if ok else "#B7BEC7"); s.set_linewidth(2.2 if ok else .6)
            if ok:
                lab = "✓ " + T["exact"]
            elif len(w) == 1:
                lab = f"{short.get(A[w[0]], A[w[0]])}: p = {p[j, w[0]]:.2f}"
            else:
                lab = T["wrong"].format(n=len(w))
            ax.text(.5, -.03, lab, transform=ax.transAxes, ha="center", va="top", fontsize=5.9,
                    family="DejaVu Sans" if ok else None, color=GREEN if ok else MUTE, weight="bold" if ok else "normal")
            ax.text(.05, .97, f"{c + 1}", transform=ax.transAxes, ha="left", va="top", fontsize=6, color="white",
                    bbox=dict(boxstyle="round,pad=.15", fc=INK, ec="none", alpha=.75))
    m = gq == qid
    for S, name in ((S_l1, "L1"), (S_ll, "LL")):
        o = np.argsort(-S[i], kind="stable"); print(name, "first exact ranks", np.nonzero(m[o])[0][:6] + 1)
    print("query attributes", [A[k] for k in np.where(qv == 1)[0]])
    fig.savefig(OUT / f"l1ll_{LANG}.pdf"); fig.savefig(OUT / f"l1ll_{LANG}.png", dpi=250); plt.close(fig)


# ================================================================ Figure 3: Platt scale and bias, held-out vs source fit
def fig_calib():
    d = json.load(open(HERE / "t2_numbers.json"))["calib"]
    fig, axs = plt.subplots(1, 2, figsize=(3.35, 1.95), gridspec_kw=dict(wspace=.42, left=.12, right=.985, top=.83, bottom=.2))
    styles = {"PETA": (BLUE_OI, "o", T["peta"]), "Market": (VERM, "^", T["market"])}
    for ax, key, lab in ((axs[0], "scale", T["scale"]), (axs[1], "bias", T["bias"])):
        allv, rs = [], []
        for fold, (col, mk, name) in styles.items():
            c = d[fold]; ah, as_, bh, bs = map(np.array, (c["scale_held"], c["scale_seen"], c["bias_held"], c["bias_seen"]))
            fit = ~(((ah == 1) & (bh == 0)) | ((as_ == 1) & (bs == 0)))
            x, y = (as_, ah) if key == "scale" else (bs, bh)
            x, y = x[fit], y[fit]; allv += [x, y]
            r = np.corrcoef(x, y)[0, 1]; rs.append(f"r = {r:.2f}")
            ax.scatter(x, y, s=11, marker=mk, fc="none" if mk == "^" else col, ec=col, lw=.8, alpha=.9, zorder=3,
                       label=f"{name} ({fit.sum()} {'thuộc tính' if LANG == 'vi' else 'attributes'})")
            print("calib", key, fold, "n", fit.sum(), "r %.2f" % r)
        v = np.concatenate(allv); lo, hi = v.min(), v.max(); pad = .06 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MUTE, lw=.7, ls=(0, (3, 2)), zorder=1)
        ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad); ax.set_aspect("equal")
        ax.set_title(lab, fontsize=7.5, pad=3)
        for n_, (s_, (col, mk, _)) in enumerate(zip(rs, styles.values())):
            ax.text(.97, .15 - n_ * .1, ("● " if mk == "o" else "△ ") + s_, transform=ax.transAxes, fontsize=6.2, color=col,
                    va="bottom", ha="right", family="DejaVu Sans")
        ax.set_xlabel(T["srcfit"], fontsize=6.6); ax.tick_params(labelsize=6.3, length=2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[0].set_ylabel(T["heldfit"], fontsize=6.6)
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=6.1, frameon=False, bbox_to_anchor=(.55, 1.0), handletextpad=.2,
               columnspacing=.8)
    fig.savefig(OUT / f"calib_{LANG}.pdf"); fig.savefig(OUT / f"calib_{LANG}.png", dpi=250); plt.close(fig)


if "--pick" in sys.argv:
    fig_l1ll(None); sys.exit()
fig_arch()
fig_l1ll(int(sys.argv[sys.argv.index("--q") + 1]) if "--q" in sys.argv else None)
fig_calib()
