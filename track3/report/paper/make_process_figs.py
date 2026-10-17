"""Process figures for the Vietnamese Track 3 report, drawn from real images and real model outputs (no box diagrams).

  t3_pipeline.png   one PETA person through the v6 system: box, 256x192 crop, heatmaps of four joints, decoded
                    skeleton, the 17 heatmap maxima behind the default score, the rescorer's estimate and the true OKS
  t3_oks_rings.png  OKS tolerance drawn on two labelled people: per-joint radius where the joint similarity is 0.5
                    and 0.95
  t3_ap_order.png   one prediction per person: at OKS threshold 0.95 the same predictions ordered by true OKS, by the
                    default score and at random; precision along each list (held-out PETA, leave-PETA-out huge model)
  t3_ears.png       the two ear channels: heatmaps of one person, and per-channel rank correlation of the heatmap
                    maximum with OKS on held-out PETA
  t3_inversion.png  left/right inversions on held-out PETA, left joints and right joints in two colours

The pipeline and ear heatmaps come from the submitted v6 models (trained on all three datasets, PETA included); every
statistic and the inversion cases come from the leave-one-dataset-out huge models (runs/lodo/feats_*.npz).
Run from report/paper: ../../../Upar-Challenge-Track-2/.venv/bin/python make_process_figs.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from PIL import Image

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
FIG = HERE / "figs"
SUB = ROOT / "submissions/t3_ens_rescore_v6"
sys.path.insert(0, str(SUB))
import run as R  # noqa: E402

ANN = ROOT / "starter/data/annotations/specialization"
LODO = ROOT / "runs/lodo"
PETA_IMG = ROOT.parent / "Upar-Challenge-Track-1/UPAR-Challenge-2027/data/PETA/images"
SIGMAS = np.array([.026, .025, .025, .035, .035, .079, .079, .072, .072, .062, .062, .107, .107, .087, .087, .089, .089])
JN = ["mũi", "đáy đầu", "đỉnh đầu", "tai trái", "tai phải", "vai trái", "vai phải", "khuỷu trái", "khuỷu phải",
      "cổ tay trái", "cổ tay phải", "hông trái", "hông phải", "gối trái", "gối phải", "cổ chân trái", "cổ chân phải"]
LEFT, RIGHT = [3, 5, 7, 9, 11, 13, 15], [4, 6, 8, 10, 12, 14, 16]
EDGES = [(2, 1), (1, 0), (1, 5), (1, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
         (12, 14), (14, 16)]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 220, "savefig.bbox": "tight"})
INK, MUTED, LC, RC, MC, GOOD, BAD, ACC = "#1a1e26", "#85837c", "#2563a8", "#c2410c", "#7c3aed", "#1a7f4b", "#b42318", "#0f766e"

d = json.load(open(ANN / "train_peta.json"))
IMS = {im["id"]: im for im in d["images"]}; ANNS = {a["id"]: a for a in d["annotations"]}


def side_color(u, v):
    if u in LEFT and v in LEFT:
        return LC
    if u in RIGHT and v in RIGHT:
        return RC
    return MC


def skeleton(ax, kp, lw=1.8, alpha=1., gt=None, dots=True):
    """kp (17, 2+) image coords; joints without a label (gt[:, 2] == 0) are skipped when gt is given."""
    vis = np.ones(17, bool) if gt is None else gt[:, 2] > 0
    for u, v in EDGES:
        if vis[u] and vis[v]:
            ax.plot([kp[u, 0], kp[v, 0]], [kp[u, 1], kp[v, 1]], color=side_color(u, v), lw=lw, alpha=alpha, solid_capstyle="round")
    if dots:
        for k in range(17):
            if vis[k]:
                ax.plot(kp[k, 0], kp[k, 1], "o", ms=2.6, color=LC if k in LEFT else RC if k in RIGHT else MC, alpha=alpha)


def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def oks(pred, g, area):
    vis = g[:, 2] > 0
    e = ((pred[:, 0] - g[:, 0]) ** 2 + (pred[:, 1] - g[:, 1]) ** 2) / (2 * SIGMAS) ** 2 / (area + np.spacing(1)) / 2
    return float(np.exp(-e)[vis].mean())


@torch.no_grad()
def infer(a):
    """Runs the v6 pipeline on one annotated person and keeps every intermediate."""
    R.load_model()
    im = IMS[a["image_id"]]; path = PETA_IMG / Path(im["file_name"]).name
    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    c, s = R.box_to_center_scale(a["bbox"]); M = R.affine(c, s)
    crop = R.warp_affine(img, M, R.IN_H, R.IN_W)
    x = torch.from_numpy(((crop - R.MEAN) / R.STD).transpose(2, 0, 1)[None].astype(np.float32)).to(R._DEVICE)
    h0, h1, per = R._forward(x); hm = (h0 + h1) / 2
    xy, mx = R.decode(hm)
    score = float(R._rescore(R._features(hm, mx, h0, h1, per, [a["bbox"]]))[0])
    p = xy[0].cpu().numpy() * np.array([(R.IN_W - 1) / (R.HM_W - 1), (R.IN_H - 1) / (R.HM_H - 1)], np.float32)
    inv = np.linalg.inv(np.vstack([M, [0, 0, 1]]))[:2]
    kp = np.c_[p, np.ones(17)] @ inv.T
    return dict(img=img.astype(np.uint8), crop=np.clip(crop, 0, 255).astype(np.uint8), hm=hm[0].float().cpu().numpy(),
                kp=kp, kp_crop=p, mx=mx[0].float().cpu().numpy(), score=score, M=M)


g_all = [a for a in d["annotations"] if a["num_keypoints"] >= 14 and a["bbox"][3] > 110]
rng = np.random.default_rng(1)

# ---------------------------------------------------------------- 1. pipeline on one person
best = None
for a in rng.permutation(g_all)[:40]:
    r = infer(a); g = np.array(a["keypoints"], float).reshape(17, 3); o = oks(r["kp"], g, a["area"])
    if .93 < o < .975 and IMS[a["image_id"]]["width"] >= 70:
        best = (a, r, g, o); break
a, r, g, o = best
fig = plt.figure(figsize=(12, 4.3))
gs = fig.add_gridspec(2, 9, width_ratios=[1.05, 1.05, .55, .55, .2, 1.05, .25, 1.6, .05], hspace=.25, wspace=.12)
ax = fig.add_subplot(gs[:, 0]); ax.imshow(r["img"]); clean(ax)
x0, y0, w, h = a["bbox"]; ax.add_patch(Rectangle((x0, y0), w, h, fill=False, ec="#facc15", lw=1.5))
ax.set_title(f"(a) ảnh {r['img'].shape[1]}×{r['img'].shape[0]}\nvà khung người cho sẵn", fontsize=8)
ax = fig.add_subplot(gs[:, 1]); ax.imshow(r["crop"]); clean(ax); ax.set_title("(b) cắt và phóng\n256×192, đệm 1.25", fontsize=8)
for i, k in enumerate([2, 5, 10, 15]):
    ax = fig.add_subplot(gs[i // 2, 2 + i % 2]); ax.imshow(r["crop"]); clean(ax)
    hmk = np.kron(np.clip(r["hm"][k], 0, None), np.ones((4, 4)))
    ax.imshow(hmk, cmap="inferno", alpha=.7, extent=(-.5, 191.5, 255.5, -.5)); ax.set_title(f"{JN[k]}\nmax {r['mx'][k]:.2f}", fontsize=7)
    nz = np.nonzero(r["crop"].sum(2) > 0); ax.set_xlim(nz[1].min() - 4, nz[1].max() + 4); ax.set_ylim(nz[0].max() + 4, nz[0].min() - 4)
fig.text((fig.axes[2].get_position().x0 + fig.axes[3].get_position().x1) / 2, 1.07, "(c) heatmap 64×48 của\n4 trong 17 khớp", ha="center", va="top", fontsize=8)
ax = fig.add_subplot(gs[:, 5]); ax.imshow(r["img"]); clean(ax); skeleton(ax, r["kp"])
ax.set_title(f"(d) toạ độ sau DARK\nOKS thật {o:.3f}", fontsize=8)
ax.set_xlim(x0 - .1 * w, x0 + 1.1 * w); ax.set_ylim(y0 + 1.05 * h, y0 - .05 * h)
ax = fig.add_subplot(gs[:, 7])
cols = ["#c9ccd3" if k in (3, 4) else ACC for k in range(17)]
ax.barh(np.arange(17)[::-1], r["mx"], color=cols, height=.7)
ax.set_yticks(np.arange(17)[::-1], JN, fontsize=6.8); ax.set_xlim(0, 1); ax.set_xlabel("max của heatmap")
d17, d15 = r["mx"].mean(), np.delete(r["mx"], [3, 4]).mean()
ax.set_title(f"(e) điểm tin cậy\nmặc định (TB 17 max) {d17:.3f}, bỏ 2 tai {d15:.3f}\nbộ chấm học được {r['score']:.3f}, OKS thật {o:.3f}", fontsize=7.6)
fig.savefig(FIG / "t3_pipeline.png"); plt.close(fig)
print("pipeline", IMS[a["image_id"]]["file_name"], "oks", o, "default", d17, "no ears", d15, "rescorer", r["score"])

# ---------------------------------------------------------------- 2. OKS rings on two people
small = [a for a in d["annotations"] if a["num_keypoints"] >= 13 and 60 < a["bbox"][3] < 90]
pair = [best[0], small[int(rng.integers(len(small)))]]
fig, axs = plt.subplots(1, 2, figsize=(7.6, 5.4), gridspec_kw={"wspace": .1})
for ax, a in zip(axs, pair):
    im = IMS[a["image_id"]]; img = Image.open(PETA_IMG / Path(im["file_name"]).name).convert("RGB")
    g = np.array(a["keypoints"], float).reshape(17, 3); s = np.sqrt(a["area"])
    ax.imshow(img); clean(ax); skeleton(ax, g, lw=1.2, alpha=.8, gt=g)
    for k in range(17):
        if g[k, 2] == 0:
            continue
        for t, ls in ((.5, "--"), (.95, "-")):
            rad = s * 2 * SIGMAS[k] * np.sqrt(2 * np.log(1 / t))
            ax.add_patch(Circle(g[k, :2], rad, fill=False, ec="#facc15" if t == .95 else "white", lw=1 if t == .95 else .7, ls=ls))
    r95 = s * 2 * SIGMAS * np.sqrt(2 * np.log(1 / .95))
    ax.set_title(f"khung cao {a['bbox'][3]:.0f} px, $\\sqrt{{\\mathrm{{area}}}}$ = {s:.0f} px\nbán kính ở 0.95: mũi {r95[0]:.1f} px, vai {r95[5]:.1f} px, hông {r95[11]:.1f} px", fontsize=7.6)
fig.savefig(FIG / "t3_oks_rings.png"); plt.close(fig)

# ---------------------------------------------------------------- 3. ordering matters at OKS 0.95
z = dict(np.load(LODO / "feats_peta.npz")); z["oks"] = np.nan_to_num(z["oks"])
default = z["X"][:, :17].mean(1); t = .95; correct = z["oks"] > t
orders = {"xếp theo OKS thật (lý tưởng)": np.argsort(-z["oks"]), "xếp theo điểm mặc định": np.argsort(-default),
          "thứ tự ngẫu nhiên": rng.permutation(len(default))}
fig = plt.figure(figsize=(11, 4.0))
gs = fig.add_gridspec(3, 2, width_ratios=[1.7, 1], hspace=.9, wspace=.18)
sub = rng.choice(len(default), 60, replace=False)
axp = fig.add_subplot(gs[:, 1])
for i, (name, o_) in enumerate(orders.items()):
    ax = fig.add_subplot(gs[i, 0])
    so = sub[np.argsort(np.argsort(o_)[sub])]  # the 60 sampled people in this list's order
    ax.imshow(np.where(correct[so], 1, 0)[None], cmap=matplotlib.colors.ListedColormap([BAD, GOOD]), aspect="auto", vmin=0, vmax=1)
    ax.set_yticks([]); ax.set_xticks([0, 59], ["đầu danh sách", "cuối"]); ax.tick_params(length=0, labelsize=7)
    c = correct[o_]; prec = np.cumsum(c) / np.arange(1, len(c) + 1); rec = np.cumsum(c) / c.sum()
    ap_ = float(np.sum(prec * c) / c.sum())
    ax.set_title(f"{name}: AP ở ngưỡng 0.95 ≈ {ap_:.3f}", fontsize=8, loc="left")
    axp.plot(rec, prec, color=[GOOD, ACC, MUTED][i], lw=1.5, label=f"{name.split(' (')[0]} ({ap_:.3f})")
axp.axhline(correct.mean(), color=MUTED, lw=.8, ls=":"); axp.text(.02, correct.mean() - .04, f"tỉ lệ dự đoán đúng $f_{{0.95}}$ = {correct.mean():.3f}", fontsize=7, color=MUTED)
axp.set_xlabel("recall"); axp.set_ylabel("precision"); axp.set_ylim(.5, 1.02); axp.legend(frameon=False, fontsize=7, loc="lower left")
axp.set_title("PETA giấu, 10.314 người, ngưỡng OKS 0.95", fontsize=8)
fig.text(.33, .005, "Mỗi ô là một người trong 60 người lấy ngẫu nhiên. Xanh là dự đoán có OKS trên 0.95, đỏ là dưới.", ha="center", fontsize=7.4, color=MUTED)
fig.savefig(FIG / "t3_ap_order.png"); plt.close(fig)
print("f095", correct.mean(), "f095^2", correct.mean() ** 2)

# ---------------------------------------------------------------- 4. ear channels
from scipy.stats import spearmanr  # noqa: E402
rho = np.array([spearmanr(z["X"][:, k], z["oks"])[0] for k in range(17)])
r_ = best[1]
fig = plt.figure(figsize=(10.5, 3.6))
gs = fig.add_gridspec(1, 5, width_ratios=[.55, .55, .55, .25, 2.1], wspace=.08)
for i, k in enumerate([3, 4, 5]):
    ax = fig.add_subplot(gs[0, i]); ax.imshow(r_["crop"]); clean(ax)
    ax.imshow(np.kron(np.clip(r_["hm"][k], 0, None), np.ones((4, 4))), cmap="inferno", alpha=.72, extent=(-.5, 191.5, 255.5, -.5))
    ax.set_title(f"{JN[k]}\nmax {r_['mx'][k]:.2f}", fontsize=7.6, color=MUTED if k in (3, 4) else INK)
ax = fig.add_subplot(gs[0, 4])
ax.bar(np.arange(17), rho, color=["#c9ccd3" if k in (3, 4) else ACC for k in range(17)])
ax.set_xticks(np.arange(17), JN, rotation=60, ha="right", fontsize=6.8); ax.axhline(0, color=INK, lw=.6)
ax.set_ylabel("Spearman giữa max heatmap và OKS")
ax.set_title("PETA giấu, mô hình huge không thấy PETA", fontsize=8)
for k in (3, 4):
    ax.text(k, rho[k] + .01 * np.sign(rho[k] or 1), f"{rho[k]:.2f}", ha="center", va="bottom" if rho[k] >= 0 else "top", fontsize=6.8, color=MUTED)
fig.savefig(FIG / "t3_ears.png"); plt.close(fig)
print("rho per channel", dict(zip(JN, rho.round(3))))

# ---------------------------------------------------------------- 5. left/right inversions
swap = list(range(17))
for l_, r2 in zip(LEFT, RIGHT):
    swap[l_], swap[r2] = r2, l_
cands = []
for i in np.nonzero(z["oks"] < .5)[0]:
    a = ANNS[int(z["ann_id"][i])]; g = np.array(a["keypoints"], float).reshape(17, 3); p = z["keypoints"][i].reshape(17, 3)
    o2 = oks(p[swap], g, a["area"])
    if o2 > .85 and a["bbox"][3] > 90:
        cands.append((i, o2))
pick = [cands[j] for j in rng.choice(len(cands), min(4, len(cands)), replace=False)]
HT = 240
info = []
for i, o2 in pick:
    a = ANNS[int(z["ann_id"][i])]; im = IMS[a["image_id"]]; img = Image.open(PETA_IMG / Path(im["file_name"]).name).convert("RGB")
    f = HT / img.height; info.append((i, o2, a, img.resize((round(img.width * f), HT)), f))
wr = []
for _, _, _, img, _ in info:
    wr += [img.width, img.width, 18]
fig = plt.figure(figsize=(sum(wr) / 95, 3.2))
gs = fig.add_gridspec(1, len(wr), width_ratios=wr, wspace=.04)
for j, (i, o2, a, img, f) in enumerate(info):
    g = np.array(a["keypoints"], float).reshape(17, 3); p = z["keypoints"][i].reshape(17, 3).copy()
    gs_ = g.copy(); gs_[:, :2] *= f; p[:, :2] *= f
    for c, (kp, title) in enumerate(((gs_, "label\n"), (p, f"dự đoán\nOKS {z['oks'][i]:.2f}"))):
        ax = fig.add_subplot(gs[0, 3 * j + c]); ax.imshow(img); clean(ax); skeleton(ax, kp, lw=1.6, gt=g)
        ax.set_title(title, fontsize=7.4); ax.set_xlim(-.5, img.width - .5); ax.set_ylim(HT - .5, -.5)
        if c:
            ax.text(0, -.04, f"đổi trái phải: OKS {o2:.2f}", transform=ax.transAxes, ha="center", va="top", fontsize=7, color=GOOD)
fig.legend([plt.Line2D([], [], color=LC, lw=2), plt.Line2D([], [], color=RC, lw=2)], ["bên trái của người", "bên phải của người"],
           frameon=False, ncol=2, fontsize=7.6, loc="lower center", bbox_to_anchor=(.5, .0))
fig.savefig(FIG / "t3_inversion.png"); plt.close(fig)
print("inversion candidates (OKS<0.5, swapped>0.85, box>90px)", len(cands))
print("saved")
