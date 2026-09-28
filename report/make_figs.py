"""Figures for the Track 1 situation report. Run with the project venv from the repo root:

.venv/bin/python report/make_figs.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "par"))
from data import DATA, attr_names, load_split  # noqa: E402
from metrics import challenge_metrics  # noqa: E402

FIG = ROOT / "report" / "figs"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 200, "savefig.bbox": "tight"})
C = {"Market1501": "#4C72B0", "PA100k": "#DD8452", "PETA": "#55A868"}
DS = ["Market1501", "PA100k", "PETA"]
ATTR = attr_names()
tr, va = load_split("train"), load_split("val")
meta = pd.read_csv(ROOT / "UPAR-Challenge-2027/eda/out/meta.csv")
rng = np.random.default_rng(3)


def img(p):
    return Image.open(DATA / p).convert("RGB")


# ---------- 1. samples per dataset, drawn at true relative pixel size ----------
def fig_samples():
    fig, axes = plt.subplots(3, 8, figsize=(10, 7.6))
    for r, ds in enumerate(DS):
        rows = meta[(meta.ds == ds) & (meta.split == "train")].sample(8, random_state=4)
        for c, (_, row) in enumerate(rows.iterrows()):
            ax = axes[r, c]; im = img(row.image)
            ax.imshow(im.resize((64, 160))); ax.axis("off")
            ax.set_title(f"{im.size[0]}×{im.size[1]} px", fontsize=7)
        axes[r, 0].text(-0.1, 1.17, ds, transform=axes[r, 0].transAxes, fontsize=11, color=C[ds], fontweight="bold")
    fig.subplots_adjust(hspace=0.35, wspace=0.08)
    fig.savefig(FIG / "samples.png"); plt.close(fig)


# ---------- 2. input -> output example ----------
def fig_io(pred):
    picks = ["Market1501/bounding_box_train/0002_c1s1_000451_03.jpg"]
    v = va.reset_index(drop=True)
    for ds in ["PA100k", "PETA"]:
        cand = v[(v.ds == ds)].index.values
        picks.append(v.image[rng.choice(cand)])
    fig = plt.figure(figsize=(10, 7.2))
    fig.subplots_adjust(top=0.9)
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 6.2], hspace=0.5, wspace=0.12)
    for k, p in enumerate(picks):
        split = tr if p in set(tr.image) else va
        y = split.loc[split.image == p, ATTR].values[0]
        ax0 = fig.add_subplot(gs[k, 0]); ax0.imshow(img(p)); ax0.axis("off")
        ax0.set_title(p.split("/")[0], fontsize=8, color=C[p.split("/")[0]])
        ax = fig.add_subplot(gs[k, 1])
        prob = pred.get(p)
        xs = np.arange(40)
        if prob is not None:
            ax.bar(xs, prob, color=["#2a9d8f" if yy else "#bbbbbb" for yy in y], width=0.8)
            ax.set_ylabel("xác suất dự đoán", fontsize=7)
        else:
            ax.bar(xs, y, color="#2a9d8f", width=0.8); ax.set_ylabel("nhãn", fontsize=7)
        ax.axhline(0.5, color="#c1121f", lw=0.8, ls="--")
        ax.set_ylim(0, 1.05); ax.set_xlim(-0.7, 39.7)
        ax.set_xticks(xs); ax.set_xticklabels(ATTR if k == 2 else [], rotation=90, fontsize=6.5)
        pos = [ATTR[i] for i in np.where(y)[0]]
        ax.text(0, 1.06, "Nhãn đúng = 1: " + ", ".join(pos), fontsize=6.5, transform=ax.get_xaxis_transform(),
                va="bottom", wrap=True)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color="#2a9d8f", label="thuộc tính có nhãn đúng = 1"),
                        Patch(color="#bbbbbb", label="thuộc tính có nhãn đúng = 0"),
                        plt.Line2D([], [], color="#c1121f", ls="--", label="ngưỡng 0.5")],
               loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.55, 0.99), fontsize=8)
    fig.savefig(FIG / "io_example.png"); plt.close(fig)


# ---------- 3. composition ----------
def fig_composition():
    fig, ax = plt.subplots(figsize=(7, 2.2))
    for i, (name, df) in enumerate([("train", tr), ("val", va)]):
        left = 0
        for ds in DS:
            n = (df.ds == ds).sum()
            ax.barh(i, n, left=left, color=C[ds], label=ds if i == 0 else None)
            ax.text(left + n / 2, i, f"{n:,}", ha="center", va="center", color="white", fontsize=8)
            left += n
        ax.text(left + 800, i, f"{len(df):,} ảnh", va="center", fontsize=8)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["train", "val"]); ax.set_xlim(0, 112000)
    ax.set_xlabel("số ảnh"); ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.35), frameon=False)
    fig.savefig(FIG / "composition.png"); plt.close(fig)


# ---------- 4. size / aspect ratio ----------
def fig_sizes():
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    for ds in DS:
        g = meta[meta.ds == ds]
        ax[0].hist(g.h, bins=np.arange(0, 620, 10), alpha=0.6, color=C[ds], label=ds, density=True)
        ax[1].hist(g.ar, bins=np.arange(0.5, 5, 0.05), alpha=0.6, color=C[ds], label=ds, density=True)
    ax[0].axvline(256, color="k", ls="--", lw=0.8); ax[0].text(260, ax[0].get_ylim()[1] * 0.8, "input model\n256 px", fontsize=7)
    ax[0].set_xlabel("chiều cao ảnh (px)"); ax[0].set_ylabel("mật độ"); ax[0].legend()
    ax[1].axvline(2.0, color="k", ls="--", lw=0.8); ax[1].text(2.05, ax[1].get_ylim()[1] * 0.8, "input model\n2:1", fontsize=7)
    ax[1].set_xlabel("tỉ lệ cao / rộng")
    fig.savefig(FIG / "sizes.png"); plt.close(fig)


# ---------- 5. label distribution shift between datasets ----------
def fig_shift():
    rate = tr.groupby("ds")[ATTR].mean().T[DS]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    x = np.arange(40); w = 0.27
    for i, ds in enumerate(DS):
        ax.bar(x + (i - 1) * w, rate[ds].values * 100, w, color=C[ds], label=ds)
    ax.set_xticks(x); ax.set_xticklabels(ATTR, rotation=90, fontsize=7)
    ax.set_ylabel("% ảnh có thuộc tính (train)"); ax.legend(ncol=3, frameon=False)
    ax.set_xlim(-0.6, 39.6)
    fig.savefig(FIG / "label_shift.png"); plt.close(fig)


# ---------- 6. per-attribute difficulty on val ----------
def fig_per_attr(logits, idx, off):
    gt = va[ATTR].values[idx].astype(bool); ds = va.ds.values[idx]
    p = 1 / (1 + np.exp(-(logits + 0.6 * off))) > 0.5
    ba = []
    for a in range(40):
        vals = []
        for d in DS:
            m = ds == d
            g, q = gt[m, a], p[m, a]
            if g.sum() == 0:
                continue
            vals.append(((g & q).sum() / g.sum() + (~g & ~q).sum() / (~g).sum()) / 2)
        ba.append(np.mean(vals))
    ba = np.array(ba); order = np.argsort(ba)
    rate = tr[ATTR].mean().values
    fig, ax = plt.subplots(figsize=(10, 3.2))
    cols = ["#c1121f" if ba[i] < 0.8 else "#2a9d8f" for i in order]
    ax.bar(np.arange(40), ba[order] * 100, color=cols)
    for j, i in enumerate(order):
        ax.text(j, 51, f"{rate[i] * 100:.1f}%", rotation=90, fontsize=5.5, color="white", ha="center", va="bottom")
    ax.set_xticks(np.arange(40)); ax.set_xticklabels([ATTR[i] for i in order], rotation=90, fontsize=7)
    ax.set_ylim(50, 100); ax.set_ylabel("balanced accuracy trên val (%)"); ax.set_xlim(-0.6, 39.6)
    fig.savefig(FIG / "per_attr.png"); plt.close(fig)
    return {ATTR[i]: float(ba[i]) for i in range(40)}


# ---------- 7. hard cases ----------
def fig_hard(logits, idx, off):
    v = va.iloc[idx].reset_index(drop=True)
    prob = 1 / (1 + np.exp(-(logits + 0.6 * off)))
    gt = v[ATTR].values
    m = meta.set_index("image")
    groups = []
    # (a) tiny crops
    small = m.loc[v.image].reset_index().sort_values("h").image.values[:40]
    groups.append(("Ảnh rất nhỏ (cao < 60 px)", list(rng.choice(small, 5, replace=False)), None))
    # (b) colour cast: strong green or purple mean colour in PETA
    peta = v[v.ds == "PETA"].image.values[:2500]
    cast = []
    for p in peta:
        a = np.asarray(img(p).resize((16, 32)), dtype=float).reshape(-1, 3).mean(0)
        cast.append(a[1] - (a[0] + a[2]) / 2)
    groups.append(("Ám màu, ánh sáng lệch, khung cắt bất thường (PETA)", list(peta[np.argsort(cast)[-5:]]), None))
    # (c) confident disagreements with the label: label noise or truly hard
    errs = []
    for a_name in ["Gender-Female", "Accessory-Backpack", "UpperBody-Color-Red", "LowerBody-Type-Skirt&Dress",
                   "Accessory-Glasses-Normal"]:
        a = ATTR.index(a_name)
        conf = np.where(gt[:, a] == 1, 1 - prob[:, a], prob[:, a])
        k = int(np.argmax(conf))
        errs.append((v.image[k], f"{a_name}\nnhãn={gt[k, a]} dự đoán={prob[k, a]:.2f}"))
    groups.append(("Model tự tin nhưng trái nhãn", [e[0] for e in errs], [e[1] for e in errs]))
    fig, axes = plt.subplots(3, 5, figsize=(10, 8.6))
    for r, (title, ims, caps) in enumerate(groups):
        for c in range(5):
            ax = axes[r, c]; ax.axis("off")
            im = img(ims[c]); ax.imshow(im)
            cap = caps[c] if caps else f"{im.size[0]}×{im.size[1]}"
            ax.set_title(cap, fontsize=6.5)
        axes[r, 0].text(-0.15, 1.18, title, transform=axes[r, 0].transAxes, fontsize=10, fontweight="bold")
    fig.subplots_adjust(hspace=0.45)
    fig.savefig(FIG / "hard_cases.png"); plt.close(fig)


# ---------- 8. co-occurrence ----------
def fig_corr():
    sel = ["Gender-Female", "Hair-Length-Long", "Hair-Length-Short", "LowerBody-Type-Skirt&Dress",
           "LowerBody-Type-Trousers&Shorts", "Accessory-Bag", "Accessory-Backpack", "Age-Young", "Age-Adult",
           "UpperBody-Length-Short", "LowerBody-Length-Short", "LowerBody-Color-Black", "LowerBody-Color-Blue"]
    c = tr[sel].corr().values
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(sel))); ax.set_xticklabels(sel, rotation=90, fontsize=7)
    ax.set_yticks(range(len(sel))); ax.set_yticklabels(sel, fontsize=7)
    for i in range(len(sel)):
        for j in range(len(sel)):
            if i != j and abs(c[i, j]) > 0.25:
                ax.text(j, i, f"{c[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                        color="white" if abs(c[i, j]) > 0.6 else "black")
    fig.colorbar(im, fraction=0.04)
    fig.savefig(FIG / "corr.png"); plt.close(fig)


# ---------- 9. leaderboard and val -> test gap ----------
def fig_results():
    lb = json.load(open(ROOT / "report" / "leaderboard.json"))
    lb = [o for o in lb if o["Challenge_Avg"] > 0.3]
    lb.sort(key=lambda o: o["Challenge_Avg"])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.5, 1]})
    names = [o["owner"] for o in lb]
    cols = ["#e76f51" if n == "tannguyenbmt47" else "#8d99ae" for n in names]
    ax[0].barh(names, [o["Challenge_Avg"] for o in lb], color=cols)
    for i, o in enumerate(lb):
        ax[0].text(o["Challenge_Avg"] + 0.003, i, f"{o['Challenge_Avg']:.4f}", va="center", fontsize=7)
    ax[0].set_xlim(0.45, 0.83); ax[0].set_xlabel("Challenge_Avg trên test (leaderboard 28/09/2026)")
    ax[0].tick_params(axis="y", labelsize=7)
    labels = ["Val\n(cùng domain)", "Test\n(domain lạ)"]
    vals = {"Avg": [0.8852, 0.7774], "mA": [0.8792, 0.7786], "Inst_F1": [0.8912, 0.7762]}
    x = np.arange(2); w = 0.25
    for i, (k, vv) in enumerate(vals.items()):
        b = ax[1].bar(x + (i - 1) * w, vv, w, label=k, color=["#264653", "#2a9d8f", "#e9c46a"][i])
        ax[1].bar_label(b, fmt="%.3f", fontsize=6.5)
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels); ax[1].set_ylim(0.6, 0.95)
    ax[1].legend(fontsize=7, frameon=False, ncol=3, loc="upper right"); ax[1].set_title("DINOv3 ViT-B/16, bài nộp đã được chấm", fontsize=9)
    fig.savefig(FIG / "results.png"); plt.close(fig)


def predictions_for(paths):
    """Predictions of the submitted model for a few images (for the I/O figure)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("run", ROOT / "submissions/dinov3_b16_v3/run.py")
    run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)
    run.load_model()
    S = [{"index": i, "image": p, "image_path": str(DATA / p), "attribute_names": ATTR} for i, p in enumerate(paths)]
    return dict(zip(paths, np.array(run.predict_batch(S))))


if __name__ == "__main__":
    z = np.load(ROOT / "runs/dinov3_b16/val_logits.npz")
    off = np.array(json.load(open(ROOT / "runs/dinov3_b16/offsets.json"))["offset"])
    fig_samples(); fig_composition(); fig_sizes(); fig_shift(); fig_corr(); fig_results()
    ba = fig_per_attr(z["logits"], z["idx"], off)
    fig_hard(z["logits"], z["idx"], off)
    rng = np.random.default_rng(3)
    v = va.reset_index(drop=True)
    picks = ["Market1501/bounding_box_train/0002_c1s1_000451_03.jpg"]
    for ds in ["PA100k", "PETA"]:
        picks.append(v.image[rng.choice(v[v.ds == ds].index.values)])
    rng = np.random.default_rng(3)
    fig_io(predictions_for(picks))
    m, per = challenge_metrics(va[ATTR].values[z["idx"]], 1 / (1 + np.exp(-(z["logits"] + 0.6 * off))),
                               va.ds.values[z["idx"]])
    json.dump({"per_attr_ba": ba, "val": m, "per_ds": {k: {kk: float(vv) for kk, vv in d.items()} for k, d in per.items()}},
              open(FIG / "stats.json", "w"), indent=1)
    print("figures in", FIG)
