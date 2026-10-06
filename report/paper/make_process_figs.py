"""Process figures for the Vietnamese Track 1 report, drawn from real images and real model outputs (no box diagrams).

  process_overview.png  one image through the whole system: patch grid, token features (PCA), per-attribute attention,
                        probabilities before and after the threshold shift
  token_pca.png         PCA of backbone token features on many people: body parts get consistent colours
  query_reading.png     what four attribute queries read: attention map and the most attended patches, zoomed
  sigmoid_people.png    real people placed on the sigmoid curve by their 'hat' logit, threshold 0.5 and shifted threshold
  distill_targets.png   hard labels versus the soft targets of the two teachers on unseen PETA images

Model: the v13 submission (DINOv3 ViT-L 176x96 + query head), unzipped at PKG. Teacher figure: held-out PETA logits of
the two teachers and the distilled student (Upar-Challenge-Track-2/runs/remote2).
Run from report/paper: ../../../Upar-Challenge-Track-2/.venv/bin/python make_process_figs.py <unzipped v13/v16 package>
"""
import json, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
FIG = HERE / "figs"
PKG = Path(sys.argv[1])
sys.path.insert(0, str(PKG / "vendor")); sys.path.insert(0, str(ROOT / "par"))
from upar.models import build_model  # noqa: E402
from upar.data import DATA, attr_names, load_split  # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 220, "savefig.bbox": "tight"})
INK, MUTED, POS, NEG, ACC = "#1a1e26", "#85837c", "#c2410c", "#2563a8", "#0f766e"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

cfg = json.load(open(PKG / "assets/config.json"))
mcfg = dict(cfg["model"], num_attrs=len(cfg["attribute_names"]))
model = build_model(mcfg, pretrained=False)
model.load_state_dict({k: v.float() for k, v in torch.load(PKG / "assets/model.pt", map_location="cpu", weights_only=True).items()})
model.eval().to(DEV)
H, W = cfg["size"]; GH, GW = H // 16, W // 16
MEAN = torch.tensor(mcfg["mean"]).view(1, 3, 1, 1).to(DEV); STD = torch.tensor(mcfg["std"]).view(1, 3, 1, 1).to(DEV)
OFF = np.array(cfg["offset"]) * cfg.get("offset_scale", 1.0)
NAMES = cfg["attribute_names"]
VI = {"Accessory-Hat": "mũ", "Accessory-Backpack": "ba lô", "Accessory-Bag": "túi", "Gender-Female": "nữ",
      "LowerBody-Type-Skirt&Dress": "váy", "LowerBody-Type-Trousers&Shorts": "quần", "Hair-Length-Long": "tóc dài",
      "Accessory-Glasses-Normal": "kính", "Age-Adult": "trưởng thành", "UpperBody-Length-Short": "áo ngắn tay"}
COL = {"Black": "đen", "Blue": "xanh dương", "Grey": "xám", "White": "trắng", "Red": "đỏ", "Brown": "nâu", "Green": "xanh lá",
       "Pink": "hồng", "Yellow": "vàng", "Purple": "tím", "Orange": "cam", "Other": "khác"}


def vi(n):
    if n in VI:
        return VI[n]
    if "-Color-" in n:
        return ("áo " if n.startswith("Upper") else "quần ") + COL[n.split("-")[-1]]
    return n


cap = {}
model.head.register_forward_hook(lambda m, i, o: cap.update(t=i[0], n=i[1]))


def run(paths):
    """Returns images (H, W, 3 uint8), logits (B, 40), patch tokens (B, GH*GW, D), attention (B, 40, GH*GW)."""
    ims = [np.asarray(Image.open(DATA / p).convert("RGB").resize((W, H), Image.BILINEAR)) for p in paths]
    x = torch.from_numpy(np.stack(ims).astype(np.float32) / 255).permute(0, 3, 1, 2).to(DEV)
    with torch.no_grad():
        lg = model((x - MEAN) / STD).float().cpu().numpy()
        t, n = cap["t"], cap["n"]
        kvt = torch.cat([t[:, :1], t[:, n:]], 1); kv = model.head.kv_norm(kvt)
        layer = model.head.layers[0]; mha = layer["attn"]
        q = layer["q_norm"](model.head.queries)[None]
        Wq, Wk, _ = mha.in_proj_weight.chunk(3); bq, bk, _ = mha.in_proj_bias.chunk(3)
        Q = q @ Wq.T + bq; K = kv @ Wk.T + bk; nh = mha.num_heads; d = Q.shape[-1] // nh
        Q = Q.view(1, -1, nh, d).transpose(1, 2); K = K.view(len(ims), -1, nh, d).transpose(1, 2)
        a = (Q @ K.transpose(-1, -2) / d ** 0.5).softmax(-1).mean(1)[:, :, 1:]
        a = a / a.sum(-1, keepdim=True)
    return np.stack(ims), lg, t[:, n:].float().cpu().numpy(), a.cpu().numpy()


va = load_split("val"); A = attr_names(); assert A == NAMES
peta = va[va.ds == "PETA"].reset_index(drop=True)


def pick(cond, n, seed=0):
    idx = np.nonzero(cond)[0]; rng = np.random.default_rng(seed)
    return peta.image.values[rng.choice(idx, n, replace=False)]


# PCA basis of patch tokens fitted on 160 PETA people
basis_paths = pick(np.ones(len(peta), bool), 160, 1)
_, _, tok_b, _ = run(basis_paths[:80]); _, _, tok_b2, _ = run(basis_paths[80:])
Tb = np.concatenate([tok_b, tok_b2]).reshape(-1, tok_b.shape[-1])
mu = Tb.mean(0); U, S, Vt = np.linalg.svd(Tb - mu, full_matrices=False); PC = Vt[:3]
proj = (Tb - mu) @ PC.T; lo, hi = np.percentile(proj, 2, 0), np.percentile(proj, 98, 0)


def pca_rgb(tokens):
    z = ((tokens - mu) @ PC.T - lo) / (hi - lo)
    return np.clip(z, 0, 1).reshape(GH, GW, 3)


def up(m):
    return np.kron(m, np.ones((16, 16)))


def arrow(fig, ax1, ax2, text=None):
    c = ConnectionPatch((1.02, .5), (-.02, .5), "axes fraction", "axes fraction", axesA=ax1, axesB=ax2,
                        arrowstyle="-|>", color=MUTED, lw=1.2, mutation_scale=12)
    fig.add_artist(c)
    if text:
        x = (ax1.get_position().x1 + ax2.get_position().x0) / 2
        fig.text(x, ax1.get_position().y0 + ax1.get_position().height * .53, text, ha="center", va="bottom", fontsize=7, color=MUTED)


# ---------------------------------------------------------------- 1. whole process on one image
cand = pick((peta["Accessory-Hat"] == 1) & (peta["Accessory-Backpack"] == 1), 6, 2)
ims, lg, tok, att = run(cand)
# choose the image whose hat and backpack are both predicted confidently
k_hat, k_bp = NAMES.index("Accessory-Hat"), NAMES.index("Accessory-Backpack")
best = int(np.argmax(np.minimum(lg[:, k_hat], lg[:, k_bp])))
im, lg1, tok1, att1 = ims[best], lg[best], tok[best], att[best]
row = peta[peta.image == cand[best]].iloc[0]
upper = max([n for n in NAMES if n.startswith("UpperBody-Color")], key=lambda n: lg1[NAMES.index(n)])
lower = max([n for n in NAMES if n.startswith("LowerBody-Color")], key=lambda n: lg1[NAMES.index(n)])
show = ["Accessory-Hat", "Accessory-Backpack", upper, lower]
fig = plt.figure(figsize=(11.5, 4.1))
gs = fig.add_gridspec(2, 9, width_ratios=[1, 1, 1, .3, .8, .8, 1.05, 1.6, .05], hspace=.35, wspace=.12)
ax_in = fig.add_subplot(gs[:, 0]); ax_in.imshow(im); ax_in.set_title("(a) ảnh đầu vào\n176×96 pixel", fontsize=8)
ax_g = fig.add_subplot(gs[:, 1]); ax_g.imshow(im)
for r in range(GH + 1):
    ax_g.axhline(r * 16 - .5, color="white", lw=.6)
for c in range(GW + 1):
    ax_g.axvline(c * 16 - .5, color="white", lw=.6)
ax_g.set_title("(b) 66 patch\n16×16 pixel", fontsize=8)
ax_p = fig.add_subplot(gs[:, 2]); ax_p.imshow(np.kron(pca_rgb(tok1), np.ones((16, 16, 1))))
ax_p.set_title("(c) feature 66 token\n(PCA → màu)", fontsize=8)
att_axes = []
for i, n in enumerate(show):
    ax = fig.add_subplot(gs[i // 2, 4 + i % 2]); k = NAMES.index(n)
    ax.imshow(im); ax.imshow(up(att1[k].reshape(GH, GW)), cmap="inferno", alpha=.62)
    ax.set_title(f"“{vi(n)}”", fontsize=7.8); att_axes.append(ax)
fig.text((att_axes[0].get_position().x0 + att_axes[1].get_position().x1) / 2, .99, "(d) attention của 4 query",
         ha="center", fontsize=8)
ax_b = fig.add_subplot(gs[:, 7])
ks = [NAMES.index(n) for n in show] + [NAMES.index(n) for n in ["Gender-Female", "Age-Adult", "LowerBody-Type-Trousers&Shorts", "Accessory-Bag"]]
p0 = 1 / (1 + np.exp(-lg1[ks])); p1 = 1 / (1 + np.exp(-(lg1[ks] + OFF[ks])))
yy = np.arange(len(ks))[::-1]
ax_b.barh(yy, p1, .6, color=[POS if v > .5 else NEG for v in p1])
ax_b.axvline(.5, color=INK, lw=1, ls="--")
for y_, v, k in zip(yy, p1, ks):
    ok = (v > .5) == bool(row[NAMES[k]])
    ax_b.text(1.03, y_, "✓" if ok else "✗", color="#1a7f4b" if ok else "#b42318", va="center", fontsize=9, transform=ax_b.get_yaxis_transform())
lab = [f"{vi(NAMES[k])} (label {int(row[NAMES[k]])})" for k in ks]
ax_b.set_yticks(yy, lab, fontsize=7.6); ax_b.set_xlim(0, 1); ax_b.set_xticks([0, .5, 1]); ax_b.set_xlabel("xác suất (threshold 0.5)")
ax_b.set_title("(e) xác suất 40 thuộc tính\n(8 thuộc tính đầu)", fontsize=8)
for ax in [ax_in, ax_g, ax_p] + att_axes:
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
fig.canvas.draw()
arrow(fig, ax_in, ax_g); arrow(fig, ax_g, ax_p)
fig.text((ax_g.get_position().x1 + ax_p.get_position().x0) / 2, ax_g.get_position().y0 - .05, "backbone\nViT-L", ha="center", va="top",
         fontsize=7.5, color=MUTED)
fig.text((ax_p.get_position().x1 + att_axes[0].get_position().x0) / 2, ax_g.get_position().y0 - .05, "query\nhead", ha="center", va="top",
         fontsize=7.5, color=MUTED)
for ax in (att_axes[0], att_axes[2]):
    fig.add_artist(ConnectionPatch((1.02, .5), (-.04, .5), "axes fraction", "axes fraction", axesA=ax_p, axesB=ax,
                                   arrowstyle="-|>", color=MUTED, lw=1, mutation_scale=10))
fig.savefig(FIG / "process_overview.png"); plt.close(fig)
print("overview:", cand[best], {vi(NAMES[k]): (round(float(a), 2), round(float(b), 2)) for k, a, b in zip(ks, p0, p1)})

# ---------------------------------------------------------------- 2. token PCA on many people
paths = pick(np.ones(len(peta), bool), 10, 7)
ims, _, tok, _ = run(paths)
fig, axs = plt.subplots(2, 10, figsize=(10, 4.2), gridspec_kw={"hspace": .05, "wspace": .05})
for i in range(10):
    axs[0, i].imshow(ims[i]); axs[1, i].imshow(np.kron(pca_rgb(tok[i]), np.ones((16, 16, 1))))
for ax in axs.flat:
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
axs[0, 0].set_ylabel("ảnh", fontsize=8); axs[1, 0].set_ylabel("feature (PCA)", fontsize=8)
fig.savefig(FIG / "token_pca.png"); plt.close(fig)

# ---------------------------------------------------------------- 3. what the queries read
cand = pick((peta["Accessory-Backpack"] == 1) & (peta["Accessory-Hat"] == 1), 8, 3)
ims, lg, tok, att = run(cand)
b = int(np.argsort(-np.minimum(lg[:, k_hat], lg[:, k_bp]))[1]); im, a1, lg1 = ims[b], att[b], lg[b]
show = ["Accessory-Hat", "Accessory-Backpack", max([n for n in NAMES if n.startswith("UpperBody-Color")], key=lambda n: lg1[NAMES.index(n)]),
        max([n for n in NAMES if n.startswith("LowerBody-Color")], key=lambda n: lg1[NAMES.index(n)])]
fig = plt.figure(figsize=(9.5, 6.4))
gs = fig.add_gridspec(4, 7, width_ratios=[.62, .25, .7, .7, .7, .3, 1.3], hspace=.5, wspace=.12)
for r, n in enumerate(show):
    k = NAMES.index(n); w = a1[k]; top = np.argsort(-w)[:3]
    ax = fig.add_subplot(gs[r, 0]); ax.imshow(im); ax.imshow(up(w.reshape(GH, GW)), cmap="inferno", alpha=.6)
    for t in top:
        rr, cc = divmod(int(t), GW); ax.add_patch(Rectangle((cc * 16 - .5, rr * 16 - .5), 16, 16, fill=False, ec="#22d3ee", lw=1.2))
    ax.set_ylabel(f"query\n“{vi(n)}”", fontsize=8, rotation=0, ha="right", va="center"); ax.set_xticks([]); ax.set_yticks([])
    for j, t in enumerate(top):
        rr, cc = divmod(int(t), GW); axp = fig.add_subplot(gs[r, 2 + j])
        axp.imshow(im[rr * 16:(rr + 1) * 16, cc * 16:(cc + 1) * 16], interpolation="nearest")
        axp.set_title(f"weight {w[t]:.2f}", fontsize=7.2); axp.set_xticks([]); axp.set_yticks([])
        for sp in axp.spines.values():
            sp.set_color("#22d3ee"); sp.set_linewidth(1.2)
        if j == 0:
            fig.add_artist(ConnectionPatch((1.03, .5), (-.08, .5), "axes fraction", "axes fraction", axesA=ax, axesB=axp,
                                           arrowstyle="-|>", color=MUTED, lw=1, mutation_scale=10))
    p = 1 / (1 + np.exp(-(lg1[k] + OFF[k])))
    axb = fig.add_subplot(gs[r, 6]); axb.barh([0], [p], color=POS if p > .5 else NEG, height=.5); axb.set_xlim(0, 1)
    axb.axvline(.5, color=INK, ls="--", lw=.8); axb.set_yticks([]); axb.text(p - .03 if p > .5 else p + .03, 0, f"p = {p:.2f}", va="center", ha="right" if p > .5 else "left", fontsize=8,
              color="white" if p > .5 else INK)
    axb.set_xticks([0, .5, 1]) if r == 3 else axb.set_xticks([])
    axb.spines["left"].set_visible(False)
    fig.add_artist(ConnectionPatch((1.06, .5), (-.04, .5), "axes fraction", "axes fraction", axesA=fig.axes[-2], axesB=axb,
                                   arrowstyle="-|>", color=MUTED, lw=1, mutation_scale=10))
fig.text(.16, .93, "attention trên 66 patch", ha="center", fontsize=8.5)
fig.text(.47, .93, "3 patch có weight lớn nhất (phóng to)", ha="center", fontsize=8.5)
fig.text(.83, .93, "xác suất sau MLP và classifier", ha="center", fontsize=8.5)
fig.savefig(FIG / "query_reading.png"); plt.close(fig)

# ---------------------------------------------------------------- 4. people on the sigmoid curve
from matplotlib.offsetbox import AnnotationBbox, OffsetImage  # noqa: E402
pos_paths = pick(peta["Accessory-Hat"] == 1, 60, 11); neg_paths = pick(peta["Accessory-Hat"] == 0, 120, 12)
ims_p, lg_p, _, _ = run(pos_paths[:60]); ims_n, lg_n, _, _ = run(neg_paths[:60]); ims_n2, lg_n2, _, _ = run(neg_paths[60:])
ims_n, lg_n = np.concatenate([ims_n, ims_n2]), np.concatenate([lg_n, lg_n2])
zp, zn = lg_p[:, k_hat], lg_n[:, k_hat]
shift = OFF[k_hat]; t1 = -shift  # decision boundary in logit units after the shift
# (label, target logit): clear negatives, a hard negative, a positive rescued by the shift, a missed positive, clear positives
plan = [("n", -5.0), ("n", -3.2), ("p", -2.2), ("p", t1 + .3), ("n", -.1), ("p", 1.8), ("p", 3.8)]
sel, used = [], set()
for s_, target in plan:
    z_ = zp if s_ == "p" else zn
    order = np.argsort(np.abs(z_ - target))
    i = next(int(j) for j in order if (s_, int(j)) not in used); used.add((s_, i)); sel.append((s_, i, float(z_[i])))
fig, ax = plt.subplots(figsize=(10, 4.4)); fig.subplots_adjust(top=.54, bottom=.12, left=.07, right=.98)
zz = np.linspace(-6.5, 5.5, 300); ax.plot(zz, 1 / (1 + np.exp(-zz)), color=INK, lw=1.4)
ax.axvspan(min(t1, 0), max(t1, 0), color=POS, alpha=.10, lw=0)
ax.axvline(0, color=INK, ls="--", lw=1); ax.text(.12, .04, "threshold 0.5\n(logit 0)", fontsize=7.5)
ax.axvline(t1, color=POS, ls="--", lw=1)
ax.text(t1 - .12, .9, f"threshold sau khi dời\n(logit {t1:.1f})", fontsize=7.5, color=POS, ha="right")
xs = np.linspace(.07, .93, len(sel))
for (s_, i, z), xb in zip(sorted(sel, key=lambda r: r[2]), xs):
    img = ims_p[i] if s_ == "p" else ims_n[i]; p = 1 / (1 + np.exp(-z)); c = POS if s_ == "p" else NEG
    ab = AnnotationBbox(OffsetImage(img, zoom=.5), (z, p), xybox=(xb, 1.42), boxcoords="axes fraction", frameon=True,
                        bboxprops=dict(edgecolor=c, lw=1.8), arrowprops=dict(arrowstyle="-", color=MUTED, lw=.7), annotation_clip=False)
    ax.add_artist(ab)
    ax.plot(z, p, "o", color=c, ms=6, zorder=4) if s_ == "p" else ax.plot(z, p, "o", mec=c, mfc="none", mew=1.8, ms=10, zorder=3)
    ax.text(xb, 1.84, f"logit {z:+.1f}", transform=ax.transAxes, ha="center", fontsize=7, color=c)
ax.set_xlim(-6.5, 5.5); ax.set_ylim(-.05, 1.05); ax.set_xlabel("logit của thuộc tính mũ"); ax.set_ylabel("xác suất")
ax.plot([], [], "o", color=POS, label="label: có mũ"); ax.plot([], [], "o", mec=NEG, mfc="none", mew=1.8, ms=8, label="label: không có mũ")
ax.legend(frameon=False, fontsize=7.5, loc="center left", bbox_to_anchor=(0, .55))
fig.savefig(FIG / "sigmoid_people.png"); plt.close(fig)
print("sigmoid:", [(s_, round(z, 2)) for s_, _, z in sel], "boundary", round(t1, 2))

# ---------------------------------------------------------------- 5. distillation targets
R2 = ROOT.parent / "Upar-Challenge-Track-2" / "runs" / "remote2"
zd = np.load(R2 / "T_dinoL_PETA/held_val_logits_last.npz"); zc = np.load(R2 / "T_clipL_PETA/held_val_logits_last.npz")
assert (zd["idx"] == zc["idx"]).all()
yv = va[A].values[zd["idx"]].astype(float); pv = va.image.values[zd["idx"]]
pd_, pc_ = 1 / (1 + np.exp(-zd["logits"])), 1 / (1 + np.exp(-zc["logits"]))
pm = 1 / (1 + np.exp(-(zd["logits"] + zc["logits"]) / 2))
attrs = ["Gender-Female", "Accessory-Backpack", "Hair-Length-Long", "UpperBody-Color-Black", "UpperBody-Color-Grey", "Age-Adult"]
kk = [A.index(a) for a in attrs]
# images where the teachers disagree with each other on at least one shown attribute, and are unsure somewhere
score = np.abs(pd_[:, kk] - pc_[:, kk]).max(1) + (np.abs(pm[:, kk] - .5) < .3).sum(1) * .1
cands = np.argsort(-score)[:40]; rng = np.random.default_rng(4); chosen = rng.choice(cands, 3, replace=False)
fig = plt.figure(figsize=(11.5, 3.3))
gs = fig.add_gridspec(1, 11, width_ratios=[.42, .62, 1.1, .2, .42, .62, 1.1, .2, .42, .62, 1.1], wspace=0)
for c, i in enumerate(chosen):
    axi, axb = fig.add_subplot(gs[4 * c]), fig.add_subplot(gs[4 * c + 2])
    axi.imshow(Image.open(DATA / pv[i]).convert("RGB").resize((W, H))); axi.set_xticks([]); axi.set_yticks([])
    for sp in axi.spines.values():
        sp.set_visible(False)
    yy = np.arange(len(kk))[::-1]; h = .2
    axb.barh(yy + 1.5 * h, yv[i, kk], h, color="#c9ccd3", label="label cứng")
    axb.barh(yy + .5 * h, pd_[i, kk], h, color=ACC, label="teacher DINOv3-L")
    axb.barh(yy - .5 * h, pc_[i, kk], h, color="#a855f7", label="teacher CLIP-L")
    axb.barh(yy - 1.5 * h, pm[i, kk], h, color=POS, label="mục tiêu mềm (trung bình logit)")
    axb.set_yticks(yy, [vi(a) for a in attrs], fontsize=7.6); axb.set_xlim(0, 1); axb.set_xticks([0, .5, 1])
    axb.tick_params(axis="x", labelsize=7); axb.tick_params(axis="y", length=0)
    if c == 0:
        h_, l_ = axb.get_legend_handles_labels()
fig.legend(h_, l_, frameon=False, fontsize=7.6, ncol=4, loc="lower center", bbox_to_anchor=(.5, -.08))
fig.savefig(FIG / "distill_targets.png"); plt.close(fig)
print("saved")
