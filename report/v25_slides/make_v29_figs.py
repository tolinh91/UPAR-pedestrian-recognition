"""Figures of the v29 deck (logit-adjusted loss, plain decision at probability 0.5), from the held-out fold logits.

../../../Upar-Challenge-Track-2/.venv/bin/python make_v29_figs.py [--pick]
results_v29.png   fold mean (PETA and Market held out) of the pair H+ + CLIP for the BCE configurations of v21/v24/v25
                  (H+ of v21, 2 CLIP seeds) and for v29 (H+ and CLIP-T both trained with logit_adj tau 0.42)
logitadj_v29.png  left: logit shift each attribute gets during training vs its training frequency;
                  right: recall of the rarest PETA attributes, BCE pair (v25 configuration) vs v29 pair, PETA held out
fusion_v29.png    per-attribute logits of one PETA image held out from training: H+, CLIP, their mean, verdict at 0
--pick            prints candidate images for the fusion example instead of drawing
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, "/media/tan/F/RWS2027/Upar-Challenge-Track-2/par")
from upar.data import attr_names, load_split  # noqa: E402

OUT = Path(__file__).parent / "figs"
R = Path("/media/tan/F/RWS2027/remote_results/rare/runs")
INK, PAPER, MUTE, BLUE, ORANGE, LORANGE = "#14202E", "#F5F3EE", "#6B7684", "#2F7FB5", "#B85A1E", "#F0A868"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 20, "axes.edgecolor": MUTE, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER})
A = attr_names(); va = load_split("val"); tr = load_split("train")
VI = {"Accessory-Glasses-Sun": "Kính râm", "Accessory-Hat": "Mũ", "Hair-Length-Bald": "Hói", "Age-Old": "Người già",
      "UpperBody-Color-Purple": "Áo tím", "UpperBody-Color-Orange": "Áo cam", "UpperBody-Color-Yellow": "Áo vàng",
      "UpperBody-Color-Pink": "Áo hồng", "UpperBody-Color-Green": "Áo xanh lá", "UpperBody-Color-Brown": "Áo nâu",
      "UpperBody-Color-Red": "Áo đỏ", "LowerBody-Color-Purple": "Quần tím", "LowerBody-Color-Pink": "Quần hồng",
      "LowerBody-Color-Yellow": "Quần vàng", "LowerBody-Color-Orange": "Quần cam", "LowerBody-Color-Green": "Quần xanh lá",
      "LowerBody-Color-Red": "Quần đỏ", "LowerBody-Color-Brown": "Quần nâu", "LowerBody-Color-White": "Quần trắng",
      "LowerBody-Type-Skirt&Dress": "Váy", "LowerBody-Type-Trousers&Shorts": "Quần dài", "Gender-Female": "Nữ",
      "Accessory-Backpack": "Balo", "Accessory-Bag": "Túi xách", "Accessory-Glasses-Normal": "Kính thường",
      "Age-Adult": "Trưởng thành", "Age-Young": "Trẻ", "Hair-Length-Long": "Tóc dài", "Hair-Length-Short": "Tóc ngắn",
      "UpperBody-Length-Short": "Áo ngắn tay", "LowerBody-Length-Short": "Quần ngắn", "UpperBody-Color-Black": "Áo đen",
      "UpperBody-Color-White": "Áo trắng", "UpperBody-Color-Grey": "Áo xám", "UpperBody-Color-Blue": "Áo xanh lam",
      "LowerBody-Color-Black": "Quần đen", "LowerBody-Color-Grey": "Quần xám", "LowerBody-Color-Blue": "Quần xanh lam"}


def load(run):
    d = np.load(R / run / "held_val_logits_last.npz")
    return dict(zip(d["idx"], d["logits"].astype(np.float64)))


def pair(h, cs, idx):
    """Logit mean of one H+ run and the seed-mean of CLIP runs (seed mean of logits, for the per-image figures)."""
    zh = np.stack([h[i] for i in idx]); zc = np.mean([np.stack([c[i] for i in idx]) for c in cs], 0)
    return zh, zc, (zh + zc) / 2


# ---- 1. results: fold means from remote_results/rare/la_compare.py (decision logit > 0, no post-processing) ----
res = [("v21", "CLIP gốc, BCE", 0.8330), ("v24", "+ nhánh văn bản, BCE", 0.8330),
       ("v25", "+ nhánh văn bản + WiSE-FT, BCE", 0.8314), ("v29", "+ nhánh văn bản, loss điều chỉnh", 0.8439)]
if "--pick" not in sys.argv:
    fig, ax = plt.subplots(figsize=(12, 5))
    lo = 0.828
    for i, (v, d, m) in enumerate(res[::-1]):
        c = ORANGE if v == "v29" else BLUE
        ax.plot([lo, m], [i, i], color="#D5DBE1", lw=6, solid_capstyle="round"); ax.plot(m, i, "o", ms=20, color=c)
        ax.text(m + 0.0003, i, f"  {m:.4f}", va="center", fontsize=21, color=INK)
    ax.set_yticks(range(4)); ax.set_yticklabels([f"{v}  {d}" for v, d, _ in res[::-1]], fontsize=21)
    ax.set_xlim(lo, 0.8495); ax.set_xticks([0.830, 0.835, 0.840, 0.845])
    ax.set_xlabel("Trung bình hai fold (PETA, Market)")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    fig.tight_layout(); fig.savefig(OUT / "results_v29.png", dpi=100); plt.close(fig)

# ---- 2. logit adjustment: shift per attribute, and recall of rare attributes on the PETA fold ----
prior = np.concatenate([tr[A].values, va[A].values]).mean(0).clip(1e-4, 1 - 1e-4)   # final members: train + val
shift = 0.42 * np.log((1 - prior) / prior)                                              # raw logit vs BCE logit
idx = np.array(sorted(load("TL_s1")))
gt = va[A].values[idx]
_, _, z_bce = pair(load("HB_s1"), [load(f"T_s{s}") for s in (1, 2)], idx)   # v25 configuration without WiSE: see note
_, _, z_la = pair(load("HL_s1"), [load(f"TL_s{s}") for s in (1, 2)], idx)
npos = gt.sum(0)
pp = tr[tr.ds != "PETA"][A].values.mean(0)
rare = [k for k in np.argsort(pp) if npos[k] >= 25][:6]
if "--pick" not in sys.argv:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14.4, 5.38), gridspec_kw={"width_ratios": [1, 1.15]})
    xs = np.logspace(np.log10(0.004), np.log10(0.96), 200)
    a1.plot(xs * 100, 0.42 * np.log((1 - xs) / xs), color=MUTE, lw=2.5)
    a1.scatter(prior * 100, shift, s=60, color=BLUE, zorder=3)
    for a, dx, dy in (("Accessory-Glasses-Sun", 0.15, 0.1), ("Accessory-Hat", 0.15, 0.15),
                      ("Accessory-Backpack", 0.15, 0.2), ("LowerBody-Type-Trousers&Shorts", -0.05, 0.35)):
        k = A.index(a)
        a1.scatter([prior[k] * 100], [shift[k]], s=130, color=ORANGE, zorder=4)
        a1.annotate(f"{VI[a]} {prior[k] * 100:.1f}%".replace(".", ","), (prior[k] * 100, shift[k]),
                    xytext=(prior[k] * 100 * (1 + dx), shift[k] + dy), fontsize=19, color=INK,
                    ha="right" if dx < 0 else "left")
    a1.axhline(0, color=MUTE, lw=1.5, ls="--")
    a1.set_xscale("log"); a1.set_xticks([0.5, 2, 10, 50]); a1.set_xticklabels(["0,5%", "2%", "10%", "50%"])
    a1.set_xlabel("Tần suất trong dữ liệu huấn luyện"); a1.set_ylabel("Mức dịch logit")
    w = 0.38; y = np.arange(len(rare))
    r_bce = [((z_bce[:, k] > 0) & (gt[:, k] == 1)).sum() / npos[k] * 100 for k in rare]
    r_la = [((z_la[:, k] > 0) & (gt[:, k] == 1)).sum() / npos[k] * 100 for k in rare]
    a2.barh(y + w / 2, r_bce, w, color=BLUE, label="BCE thường")
    a2.barh(y - w / 2, r_la, w, color=ORANGE, label="Loss điều chỉnh (v29)")
    for i, (b, l) in enumerate(zip(r_bce, r_la)):
        a2.text(b + 1, i + w / 2, f"{b:.0f}%", va="center", fontsize=17, color=INK)
        a2.text(l + 1, i - w / 2, f"{l:.0f}%", va="center", fontsize=17, color=INK)
    a2.set_yticks(y); a2.set_yticklabels([VI.get(A[k], A[k]) for k in rare]); a2.invert_yaxis()
    a2.set_xlim(0, 108); a2.set_xlabel("Tỉ lệ ảnh có thuộc tính được nhận ra")
    a2.legend(frameon=False, fontsize=18, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2); a2.spines["left"].set_visible(False)
    a2.tick_params(axis="y", length=0)
    fig.tight_layout(); fig.savefig(OUT / "logitadj_v29.png", dpi=100); plt.close(fig)
    for k, b, l in zip(rare, r_bce, r_la):
        print(f"{A[k]:32s} prior(train, no PETA) {pp[k] * 100:.2f}%  PETA positives {int(npos[k])}  recall {b:.0f} -> {l:.0f}")
    fp_b = ((z_bce > 0) & (gt == 0)).sum(1).mean(); fp_l = ((z_la > 0) & (gt == 0)).sum(1).mean()
    print(f"false positives per image {fp_b:.2f} -> {fp_l:.2f}; positives per image {(z_bce > 0).sum(1).mean():.2f} -> "
          f"{(z_la > 0).sum(1).mean():.2f}; true {gt.sum(1).mean():.2f}")
    for a in ("Accessory-Glasses-Sun", "Accessory-Hat", "Accessory-Backpack", "LowerBody-Type-Trousers&Shorts"):
        print(f"shift {a}: prior {prior[A.index(a)] * 100:.2f}% shift {shift[A.index(a)]:+.2f}")

# ---- 3. fusion example on a PETA image never seen in training ----
zh, zc, zm = pair(load("HL_s1"), [load("TL_s1")], idx)
if "--pick" in sys.argv:
    ok = ((zm > 0) == (gt == 1)).all(1)
    dis = ((zh > 0) != (zc > 0)) & ((zm > 0) == (gt == 1))
    for j in np.where(ok & (dis.sum(1) >= 2))[0][:40]:
        print(idx[j], va.image.values[idx[j]], "pos:", [VI.get(A[k], A[k]) for k in np.where(gt[j] == 1)[0]],
              "rescued:", [VI.get(A[k], A[k]) for k in np.where(dis[j])[0]])
    sys.exit()
ROW = int(sys.argv[sys.argv.index("--row") + 1]) if "--row" in sys.argv else None
if ROW is not None:
    j = list(idx).index(ROW)
    resc = [k for k in np.where(((zh[j] > 0) != (zc[j] > 0)) & ((zm[j] > 0) == (gt[j] == 1)))[0]]  # members disagree
    rows = (resc + [k for k in np.where(gt[j] == 1)[0] if k not in resc])[:6]
    fig, ax = plt.subplots(figsize=(14, 6.6))
    for i, k in enumerate(rows):
        y0 = len(rows) - 1 - i
        ax.plot([zh[j, k], zc[j, k]], [y0, y0], color="#D5DBE1", lw=3, zorder=1)
        ax.plot(zh[j, k], y0, "o", ms=17, color=BLUE, zorder=3); ax.plot(zc[j, k], y0, "s", ms=15, color=LORANGE, zorder=3)
        ax.plot(zm[j, k], y0, "D", ms=15, color=INK, zorder=4)
        good = (zm[j, k] > 0) == bool(gt[j, k])
        ax.text(5.4, y0, ("có" if zm[j, k] > 0 else "không") + (" · đúng" if good else " · sai"), fontsize=22,
                color=INK if good else ORANGE, va="center", fontweight="normal" if good else "bold")
        print(f"{VI.get(A[k], A[k]):14s} H+ {zh[j, k]:+.2f} CLIP {zc[j, k]:+.2f} mean {zm[j, k]:+.2f} label {gt[j, k]:.0f}")
    ax.axvline(0, color=MUTE, lw=2.5, ls="--")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([VI.get(A[k], A[k]) for k in rows][::-1], fontsize=22)
    ax.set_xlim(-5, 8.2); ax.set_xticks([-4, -2, 0, 2, 4]); ax.set_xlabel("Logit (âm là “không”, dương là “có”)")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    hs = [Line2D([], [], marker="o", ls="", ms=15, color=BLUE), Line2D([], [], marker="s", ls="", ms=14, color=LORANGE),
          Line2D([], [], marker="D", ls="", ms=14, color=INK)]
    ax.legend(hs, ["logit H+", "logit CLIP", "trung bình"], frameon=False, fontsize=20, ncol=3, loc="upper left",
              bbox_to_anchor=(0, 1.12))
    fig.tight_layout(); fig.savefig(OUT / "fusion_v29.png", dpi=100); plt.close(fig)
    print("image", va.image.values[ROW], "all 40 correct:", bool(((zm[j] > 0) == (gt[j] == 1)).all()))
print("ok")
