"""Process figures for the Vietnamese Track 2 report, drawn from real images and real model outputs (no box diagrams).

  t2_pipeline.png    one real query: the 40 per-attribute log-probability terms of five gallery images, their sum, and
                     the resulting rank; one exact match is pushed down by a single confident attribute
  t2_l1_vs_ll.png    the same gallery ranked by the L1 baseline and by log-likelihood, top 6 of each
  t2_calib_attr.png  per-attribute calibration on held-out PETA: observed frequency vs logit, the source-fitted and the
                     oracle (held-out-fitted) Platt curves; and slope / intercept of all attributes, source vs held-out
  t2_gce_people.png  12 upper-body colour probabilities of three PETA people, same-seed models without / with group CE

Data: held-out PETA logits of the leave-one-dataset-out runs (runs/remote*/), the val queries.
Run from report/paper: ../../.venv/bin/python make_process_figs.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "par"))
from eval_retrieval import load_task2  # noqa: E402
from bayes_decode import platt_fit  # noqa: E402
from upar.data import DATA, attr_names, load_split  # noqa: E402
from upar.retrieval import score_l1, score_loglik  # noqa: E402

FIG = HERE / "figs"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 220, "savefig.bbox": "tight"})
INK, MUTED, POS, NEG, ACC, GOOD, NEAR, BAD = "#1a1e26", "#85837c", "#c2410c", "#2563a8", "#0f766e", "#1a7f4b", "#d99a06", "#b42318"
A = attr_names(); va = load_split("val")
VI = {"Age-Young": "trẻ", "Age-Adult": "trưởng thành", "Age-Old": "già", "Gender-Female": "nữ",
      "Hair-Length-Short": "tóc ngắn", "Hair-Length-Long": "tóc dài", "Hair-Length-Bald": "hói",
      "UpperBody-Length-Short": "áo ngắn tay", "LowerBody-Length-Short": "quần ngắn",
      "LowerBody-Type-Trousers&Shorts": "quần dài/ngắn", "LowerBody-Type-Skirt&Dress": "váy",
      "Accessory-Backpack": "ba lô", "Accessory-Bag": "túi", "Accessory-Glasses-Normal": "kính",
      "Accessory-Glasses-Sun": "kính râm", "Accessory-Hat": "mũ"}
COL = {"Black": "đen", "Blue": "xanh dương", "Brown": "nâu", "Green": "xanh lá", "Grey": "xám", "Orange": "cam",
       "Pink": "hồng", "Purple": "tím", "Red": "đỏ", "White": "trắng", "Yellow": "vàng", "Other": "khác"}


def vi(a):
    if "-Color-" in a:
        return ("áo " if a.startswith("Upper") else "quần ") + COL[a.split("-")[-1]]
    return VI[a]


def load(run, f="held_val_logits_last.npz"):
    z = np.load(ROOT / "runs" / run / f)
    return z["logits"].astype(np.float64), z["idx"]


ids, queries = load_task2("val")
qid_of = dict(zip(ids.image, ids.qid))


def gallery(idx):
    paths = va.image.values[idx]
    return paths, va[A].values[idx].astype(np.float32), np.array([qid_of[p] for p in paths])


def thumb(ax, path, color=None, lw=3):
    ax.imshow(Image.open(DATA / path).convert("RGB").resize((64, 128))); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(color is not None); sp.set_color(color or "white"); sp.set_linewidth(lw)


def query_text(qv):
    pos = [vi(A[k]) for k in np.nonzero(qv)[0]]
    if not qv[A.index("Gender-Female")]:
        pos.insert(0, "nam")
    return ", ".join(pos)


# ---------------------------------------------------------------- 1. from query to ranked list
z, idx = load("remote2/kd2_a08_e6_PETA")
paths, gt, gq = gallery(idx)
p = np.clip(1 / (1 + np.exp(-z)), 1e-4, 1 - 1e-4)
q_ids, counts = np.unique(gq, return_counts=True)
rng = np.random.default_rng(0)
pick = None
for q in rng.permutation(q_ids[counts == 1]):  # one exact match, ranked 3..12, pushed down by one confident term
    qv = queries[q]
    if qv[33:35].sum() != 1 or qv[8:20].sum() != 1 or qv[21:33].sum() != 1:
        continue
    s = score_loglik(qv[None], p)[0]; order = np.argsort(-s); r = int(np.nonzero(gq[order] == q)[0][0])
    m = order[r]; terms = qv * np.log(p[m]) + (1 - qv) * np.log(1 - p[m])
    if 3 <= r <= 12 and terms.min() < -2.5 and np.sort(terms)[1] > -1.0:
        pick = (q, s, order, r); break
q, s, order, r = pick; qv = queries[q]
rows = [order[0], order[1], order[2], order[r], order[len(order) // 2]]
ranks = [1, 2, 3, r + 1, len(order) // 2 + 1]
fig = plt.figure(figsize=(11, 5.6))
gs = fig.add_gridspec(len(rows), 4, width_ratios=[.45, 6.2, 1.0, .9], hspace=.35, wspace=.05)
for i, (g, rk) in enumerate(zip(rows, ranks)):
    wrong = int((gt[g] != qv).sum()); exact = gq[g] == q
    c = GOOD if exact else (NEAR if wrong <= 2 else BAD)
    thumb(fig.add_subplot(gs[i, 0]), paths[g], c)
    ax = fig.add_subplot(gs[i, 1])
    terms = qv * np.log(p[g]) + (1 - qv) * np.log(1 - p[g])
    ax.imshow(terms[None], cmap="Reds_r", vmin=-5, vmax=0, aspect="auto")
    for k in range(40):
        if gt[g, k] != qv[k]:
            ax.text(k, 0, "×", ha="center", va="center", fontsize=9, color=INK)
        if terms[k] < -1.5:
            ax.text(k, .32, f"{terms[k]:.1f}", ha="center", va="top", fontsize=6, color="white" if terms[k] < -3 else INK)
    ax.set_yticks([]); ax.set_xticks(np.arange(40) - .5, minor=True); ax.tick_params(which="minor", length=0)
    ax.grid(which="minor", axis="x", color="white", lw=.6); ax.set_xticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if i == 0:
        for name, a, b in (("tuổi", 0, 3), ("giới", 3, 4), ("tóc", 4, 7), ("", 7, 8), ("màu áo", 8, 20), ("", 20, 21),
                           ("màu quần", 21, 33), ("kiểu", 33, 35), ("phụ kiện", 35, 40)):
            ax.text((a + b - 1) / 2, -.62, name, ha="center", va="bottom", fontsize=7, color=MUTED)
    lab = "khớp chính xác" if exact else f"lệch {wrong} thuộc tính"
    axs = fig.add_subplot(gs[i, 2]); axs.axis("off")
    axs.text(.05, .5, f"tổng = {s[g]:.1f}", va="center", fontsize=8.5, color=INK)
    axs.text(.05, .12, lab, va="center", fontsize=7.2, color=c)
    axr = fig.add_subplot(gs[i, 3]); axr.axis("off")
    axr.text(.5, .5, f"hạng {rk}", ha="center", va="center", fontsize=9.5, color=c, fontweight="bold")
fig.text(.5, .955, "Truy vấn: " + query_text(qv) + " (các thuộc tính khác bằng 0)", ha="center", fontsize=8.5)
cb = fig.add_axes([.25, .045, .35, .015])
plt.colorbar(plt.cm.ScalarMappable(cmap="Reds_r", norm=plt.Normalize(-5, 0)), cax=cb, orientation="horizontal")
cb.set_xlabel("log xác suất mà mô hình gán cho đúng giá trị của truy vấn ở thuộc tính đó (× = label khác truy vấn)", fontsize=7.5)
cb.tick_params(labelsize=7)
fig.savefig(FIG / "t2_pipeline.png"); plt.close(fig)
print("pipeline query", q, query_text(qv), "exact match rank", r + 1, "gallery", len(order))

# ---------------------------------------------------------------- 2. L1 vs log-likelihood on one query
best = None
for q in rng.permutation(q_ids[counts <= 2])[:3000]:
    qv = queries[q]
    a = score_l1(qv[None], p)[0]; b = score_loglik(qv[None], p)[0]
    ra = int(np.nonzero(gq[np.argsort(-a, kind="stable")] == q)[0][0]); rb = int(np.nonzero(gq[np.argsort(-b, kind="stable")] == q)[0][0])
    if rb == 0 and ra >= 10:
        best = (q, a, b, ra); break
q, a, b, ra = best; qv = queries[q]
fig, axs = plt.subplots(2, 7, figsize=(9.2, 4.6), gridspec_kw={"width_ratios": [1.5] + [1] * 6, "hspace": .45, "wspace": .12})
for row, (sc, name) in enumerate(((a, "L1 (baseline)"), (b, "log-likelihood"))):
    o = np.argsort(-sc, kind="stable")
    axs[row, 0].axis("off")
    rr = int(np.nonzero(gq[o] == q)[0][0]) + 1
    axs[row, 0].text(1, .55, name, ha="right", va="center", fontsize=10, fontweight="bold")
    axs[row, 0].text(1, .35, f"ảnh khớp ở hạng {rr}", ha="right", va="center", fontsize=8, color=GOOD if rr == 1 else BAD)
    for j in range(6):
        g = o[j]; wrong = int((gt[g] != qv).sum()); exact = gq[g] == q
        c = GOOD if exact else (NEAR if wrong <= 2 else BAD)
        thumb(axs[row, j + 1], paths[g], c)
        axs[row, j + 1].set_title(f"#{j + 1}\n{'khớp' if exact else f'lệch {wrong}'}", fontsize=7.6)
fig.suptitle("Truy vấn: " + query_text(qv), fontsize=8.5, y=1.0)
fig.savefig(FIG / "t2_l1_vs_ll.png"); plt.close(fig)
print("l1 vs ll query", q, "L1 rank", ra + 1)

# ---------------------------------------------------------------- 3. calibration per attribute
zs, idx_s = load("remote2/kd2_a08_e6_PETA", "seen_val_logits_last.npz")
ys = va[A].values[idx_s].astype(float); yh = gt.astype(float)
ab_s = platt_fit(zs, ys, np.ones_like(ys)); ab_h = platt_fit(z, yh, np.ones_like(yh))
ok = [k for k in range(40) if 30 <= yh[:, k].sum() <= len(yh) - 30 and 30 <= ys[:, k].sum() <= len(ys) - 30
      and abs(ab_s[k, 0]) < 5 and abs(ab_h[k, 0]) < 5]
d_slope, d_int = np.abs(ab_s[:, 0] - ab_h[:, 0]), np.abs(ab_s[:, 1] - ab_h[:, 1])
show = sorted([k for k in ok if d_slope[k] < .15], key=lambda k: -d_int[k])[:3]
fig = plt.figure(figsize=(11.5, 3.4))
gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, .3, .85, .85], wspace=.35)
for c, k in enumerate(show):
    ax = fig.add_subplot(gs[0, c])
    x = z[:, k]; edges = np.quantile(x, np.linspace(0, 1, 13)); mid, fr = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x <= hi); mid.append(x[m].mean()); fr.append(yh[m, k].mean())
    xx = np.linspace(np.percentile(x, 1), np.percentile(x, 99.5), 200)
    ax.plot(xx, 1 / (1 + np.exp(-xx)), color=MUTED, lw=1.2, ls=":", label="xác suất mô hình trả về")
    ax.plot(xx, 1 / (1 + np.exp(-(ab_s[k, 0] * xx + ab_s[k, 1]))), color=NEG, lw=1.6, label="hiệu chỉnh khớp trên dữ liệu nguồn")
    ax.plot(xx, 1 / (1 + np.exp(-(ab_h[k, 0] * xx + ab_h[k, 1]))), color=POS, lw=1.6, label="hiệu chỉnh oracle khớp trên PETA")
    ax.plot(mid, fr, "o", color=INK, ms=4, label="tần suất thật trên PETA (12 nhóm ảnh)")
    ax.set_title(f"{vi(A[k])}\nnguồn: dốc {ab_s[k, 0]:.2f}, chặn {ab_s[k, 1]:+.2f}\nPETA: dốc {ab_h[k, 0]:.2f}, chặn {ab_h[k, 1]:+.2f}", fontsize=7.4)
    ax.set_xlabel("logit"); ax.set_ylim(-.03, 1.03)
    if c == 0:
        ax.set_ylabel("xác suất hoặc tần suất")
h_, l_ = ax.get_legend_handles_labels()
fig.legend(h_, l_, frameon=False, ncol=4, fontsize=7.4, loc="lower center", bbox_to_anchor=(.4, -.12))
for j, (col, name, color) in enumerate(((0, "hệ số dốc $a_k$", NEG), (1, "hệ số chặn $b_k$", POS))):
    ax = fig.add_subplot(gs[0, 4 + j])
    ax.axline((0, 0), slope=1, color=MUTED, lw=.8, ls="--")
    ax.scatter(ab_s[ok, col], ab_h[ok, col], s=13, color=color, zorder=3)
    ax.set_xlabel("khớp trên nguồn", fontsize=7.6)
    if j == 0:
        ax.set_ylabel("khớp trên PETA", fontsize=7.6)
    lo_, hi_ = min(ab_s[ok, col].min(), ab_h[ok, col].min()) - .1, max(ab_s[ok, col].max(), ab_h[ok, col].max()) + .1
    ax.set_xlim(lo_, hi_); ax.set_ylim(lo_, hi_)
    ax.set_title(f"{name}, {len(ok)} thuộc tính\nlệch trung vị {np.median([d_slope, d_int][col][ok]):.2f}", fontsize=7.6)
fig.savefig(FIG / "t2_calib_attr.png"); plt.close(fig)
print("calib", len(ok), "attrs; median |d| slope", np.median(d_slope[ok]), "intercept", np.median(d_int[ok]),
      "| median slope", np.median(ab_s[ok, 0]), np.median(ab_h[ok, 0]), "| shown", [(A[k], ab_s[k].round(2), ab_h[k].round(2)) for k in show])

# ---------------------------------------------------------------- 4. what group CE computes, on two random people
z0, i0 = load("remote/b_q_PETA_s2"); z1, i1 = load("remote/b_q_gce_PETA_s2"); assert (i0 == i1).all()
paths0, gt0, _ = gallery(i0)
up = [k for k in range(40) if A[k].startswith("UpperBody-Color")]
one = np.nonzero(gt0[:, up].sum(1) == 1)[0]
chosen = np.random.default_rng(7).choice(one, 2, replace=False)
names = [COL[A[k].split("-")[-1]] for k in up]
fig = plt.figure(figsize=(11.5, 4.6))
gs = fig.add_gridspec(2, 4, width_ratios=[.42, 1.4, 1.4, 1.4], hspace=.75, wspace=.12)
for r, i in enumerate(chosen):
    t = int(gt0[i, up].argmax()); zz0, zz1 = z0[i, up], z1[i, up]
    sig0, sm0, sig1 = 1 / (1 + np.exp(-zz0)), np.exp(zz0 - zz0.max()) / np.exp(zz0 - zz0.max()).sum(), 1 / (1 + np.exp(-zz1))
    thumb(fig.add_subplot(gs[r, 0]), paths0[i])
    for c, (v, title, color) in enumerate(((sig0, f"40 sigmoid độc lập, không group CE\ntổng 12 xác suất = {sig0.sum():.2f}", "#9aa0aa"),
                                           (sm0, f"softmax của cùng 12 logit\nloss group CE = $-\\log${sm0[t]:.2f} = {-np.log(sm0[t]):.2f}", ACC),
                                           (sig1, f"sigmoid, mô hình có group CE (cùng seed)\ntổng 12 xác suất = {sig1.sum():.2f}", NEG))):
        ax = fig.add_subplot(gs[r, 1 + c]); xx = np.arange(12)
        ax.bar(xx, v, .65, color=[POS if j == t else color for j in xx])
        ax.set_xticks(xx, names, rotation=70, fontsize=6.6); ax.set_ylim(0, 1.05)
        ax.set_yticks([0, .5, 1]) if c == 0 else ax.set_yticks([])
        ax.set_title(title, fontsize=7.4)
fig.text(.5, -.1, "Cột cam là màu áo theo label. Hai người được chọn ngẫu nhiên trong số người PETA có đúng một màu áo.", ha="center", fontsize=7.6, color=MUTED)
fig.savefig(FIG / "t2_gce_people.png"); plt.close(fig)
p0, p1 = 1 / (1 + np.exp(-z0[:, up])), 1 / (1 + np.exp(-z1[:, up]))
print("gce over", len(one), "people: mean sum", p0[one].sum(1).mean(), "->", p1[one].sum(1).mean())
print("saved")
