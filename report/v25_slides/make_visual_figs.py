"""Visual (formula-free) figures for the v25 slides, all from real data.

loss_curves.png   weighted BCE (DeepMAR + smoothing 0.05) against the predicted probability, rare vs common attribute,
                  with the real example points of the H+ member
anchor_hist.png   text-anchor score 100*<normalize(P cls), w0_k> of the v25 CLIP member on val images, with vs without
                  the attribute (hat, blue top)
rule_hist.png     fold-PETA pair logits (H+ + CLIP-T WiSE 0.7, PETA held out) of the hat attribute, positives vs negatives,
                  with the plain threshold and the prior-corrected one
fusion_dots.png   per-attribute fusion of the PETA example (H+, CLIP, mean, + offset) on the logit axis
sigmoid.png       logit -> probability for the hat query example (z = +3.41)
"""
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, "/media/tan/F/RWS2027/Upar-Challenge-Track-2/par")
from upar.data import UPARDataset, attr_names, build_transforms, load_split  # noqa: E402
from upar.models import build_model  # noqa: E402

OUT = Path(__file__).parent / "figs"
INK, PAPER, MUTE, BLUE, ORANGE, LORANGE, GREY = "#14202E", "#F5F3EE", "#6B7684", "#2F7FB5", "#B85A1E", "#F0A868", "#AEB6BF"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 20, "axes.edgecolor": MUTE, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER})
A = attr_names(); va = load_split("val"); tr = load_split("train")
prior = pd.concat([tr, va])[A].values.mean(0)
nums = json.load(open(OUT / "numbers.json"))


def loss_curves():
    p = np.linspace(0.005, 0.995, 400)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.6), sharey=True)
    for a, name, col in (("Accessory-Glasses-Sun", "Kính râm (1,2% số ảnh)", ORANGE), ("LowerBody-Type-Trousers&Shorts", "Quần dài (89% số ảnh)", BLUE)):
        k = A.index(a); wp, wn = np.exp(1 - prior[k]), np.exp(prior[k])
        ax[0].plot(p, wp * -(0.975 * np.log(p) + 0.025 * np.log(1 - p)), color=col, lw=4, label=name)
        ax[1].plot(p, wn * -(0.025 * np.log(p) + 0.975 * np.log(1 - p)), color=col, lw=4, label=name)
    e = nums["images"]["19023"]
    for a, side, txt in (("LowerBody-Color-Grey", 0, "Quần xám, đúng"), ("Accessory-Glasses-Normal", 1, "Kính thường, sai")):
        k = A.index(a); z = e["hp_query"][k]; pp = 1 / (1 + np.exp(-z)); y = side == 0
        w = np.exp(1 - prior[k]) if y else np.exp(prior[k])
        l = w * -((0.975 if y else 0.025) * np.log(pp) + (0.025 if y else 0.975) * np.log(1 - pp))
        ax[side].plot([pp], [l], "o", ms=16, color=INK, zorder=5)
        ax[side].annotate(f"{txt}\np = {pp:.2f}, loss = {l:.2f}", (pp, l), xytext=(-250 if side else 30, 30 if side else 60),
                          textcoords="offset points", fontsize=19, arrowprops=dict(arrowstyle="-", color=INK))
    ax[0].set_title("Ảnh CÓ thuộc tính", fontsize=24, color=INK, loc="left")
    ax[1].set_title("Ảnh KHÔNG có thuộc tính", fontsize=24, color=INK, loc="left")
    for a in ax:
        a.set_xlabel("Xác suất mô hình dự đoán là có"); a.set_ylim(0, 8)
    ax[0].set_ylabel("Mức phạt (loss)"); ax[0].legend(frameon=False, fontsize=19, loc="upper right")
    fig.tight_layout(); fig.savefig(OUT / "loss_curves.png", dpi=100); plt.close(fig)


def load_clip():
    sub = Path("/media/tan/F/RWS2027/Upar-Challenge-Track-1/submissions")
    cfg = json.loads(zipfile.ZipFile(sub / "t1_hp_clipTw70_gpu_v25.zip").read("assets/config.json"))
    mcfg = dict([m for m in cfg["models"] if m["file"] == "m1.pt"][0]["model"], num_attrs=40)
    with torch.device("cuda"):
        m = build_model(mcfg, pretrained=False)
    m.load_state_dict({k: v.float() for k, v in torch.load(sub / "v25_parts/F_clipT_w70.pt", map_location="cpu", mmap=True,
                                                           weights_only=True).items()})
    return m.eval().cuda(), mcfg


def anchor_hist():
    m, mcfg = load_clip(); torch.set_grad_enabled(False)
    rng = np.random.default_rng(0)
    rows = np.concatenate([rng.choice(np.where(va.ds.values == d)[0], n, replace=False) for d, n in (("PA100k", 1500), ("PETA", 1500))])
    ds = UPARDataset("val", build_transforms((224, 112), False, mean=mcfg["mean"], std=mcfg["std"]), rows=rows)
    E = []
    for i in range(0, len(rows), 64):
        x = torch.stack([ds[j][0] for j in range(i, min(i + 64, len(rows)))]).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            t = m.tokens(x).float()
        E.append(torch.nn.functional.normalize(t[:, 0] @ m.head.proj, dim=-1).cpu())
    E = torch.cat(E); w0 = m.head.text_w0.cpu(); y = va[A].values[rows]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.4))
    for j, (a, name) in enumerate((("Accessory-Hat", "Câu “a person wearing a hat or a cap”"), ("UpperBody-Color-Blue", "Câu “a person wearing a blue top”"))):
        k = A.index(a); s = (100 * E @ w0[k]).numpy()
        bins = np.linspace(np.percentile(s, 0.5), np.percentile(s, 99.5), 40)
        ax[j].hist(s[y[:, k] == 0], bins=bins, density=True, color=GREY, alpha=0.9, label="ảnh không có")
        ax[j].hist(s[y[:, k] == 1], bins=bins, density=True, color=ORANGE, alpha=0.75, label="ảnh có")
        ax[j].set_title(name, fontsize=22, color=INK, loc="left"); ax[j].set_yticks([])
        ax[j].set_xlabel("Điểm khớp với câu (100 × chênh lệch cosine)")
        ax[j].legend(frameon=False, fontsize=19)
        print(a, "pos mean", s[y[:, k] == 1].mean().round(2), "neg mean", s[y[:, k] == 0].mean().round(2), "n pos", int(y[:, k].sum()))
    fig.tight_layout(); fig.savefig(OUT / "anchor_hist.png", dpi=100); plt.close(fig)
    d = m.head.text_delta.detach().cpu(); w = w0 + 0.1 * d
    ang = torch.rad2deg(torch.acos((torch.nn.functional.cosine_similarity(w0, w, dim=1)).clamp(-1, 1)))
    print("rotation of the text directions by delta (deg): mean", ang.mean().item(), "max", ang.max().item(),
          "hat", ang[A.index("Accessory-Hat")].item())


def rule_hist():
    R = Path("/media/tan/F/RWS2027/remote_results/rare/runs"); K = Path("/media/tan/F/RWS2027/remote_results/kdhp/runs")
    c = np.load(R / "T_s1/held_val_logits_wise0.70.npz"); h = np.load(K / "P_dinoHp224/held_val_logits_last.npz")
    hz = dict(zip(h["idx"], h["logits"])); idx = c["idx"]
    z = (c["logits"] + np.stack([hz[i] for i in idx])) / 2
    k = A.index("Accessory-Hat"); y = va[A].values[idx][:, k]
    p = tr[tr.ds != "PETA"][A].values.mean(0)[k]
    off = 0.42 * (np.log((1 - p) / p) - (1 - 2 * p))
    fig, ax = plt.subplots(figsize=(10, 5.6))
    bins = np.linspace(-9, 5, 57)
    ax.hist(z[:, k][y == 0], bins=bins, color=GREY, alpha=0.9, label=f"ảnh không đội mũ ({int((y == 0).sum())})", density=True)
    ax.hist(z[:, k][y == 1], bins=bins, color=ORANGE, alpha=0.75, label=f"ảnh đội mũ ({int((y == 1).sum())})", density=True)
    ax.axvline(0, color=MUTE, lw=3, ls="--"); ax.axvline(-off, color=INK, lw=4)
    tpr0, tpr1 = (z[y == 1, k] > 0).mean(), (z[y == 1, k] > -off).mean()
    tnr0, tnr1 = (z[y == 0, k] <= 0).mean(), (z[y == 0, k] <= -off).mean()
    ax.annotate("ngưỡng 0", (0, 0.25), xytext=(10, 0), textcoords="offset points", color=MUTE, fontsize=19)
    ax.annotate(f"ngưỡng sau khi dịch\n(−{off:.2f})", (-off, 0.32), xytext=(-215, 0), textcoords="offset points", color=INK, fontsize=19)
    ax.set_yticks([]); ax.set_xlabel("Logit trung bình của hai thành viên"); ax.legend(frameon=False, fontsize=19, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT / "rule_hist.png", dpi=100); plt.close(fig)
    print(f"hat PETA fold: prior {p:.3f} offset {off:.2f}; recall {tpr0:.2f} -> {tpr1:.2f}; specificity {tnr0:.3f} -> {tnr1:.3f}; "
          f"balanced acc {(tpr0 + tnr0) / 2:.3f} -> {(tpr1 + tnr1) / 2:.3f}")


def fusion_dots():
    e = nums["images"]["32671"]; off = np.array(nums["offset"])
    rows = [("Nữ", "Gender-Female"), ("Balo", "Accessory-Backpack"), ("Mũ", "Accessory-Hat"), ("Kính thường", "Accessory-Glasses-Normal"),
            ("Trưởng thành", "Age-Adult"), ("Hói", "Hair-Length-Bald")]
    fig, ax = plt.subplots(figsize=(12, 6.4))
    for i, (name, a) in enumerate(rows):
        k = A.index(a); y0 = len(rows) - 1 - i
        zh = e["hp_query"][k]; zc = e["clip_query"][k] + e["clip_text"][k]; mz = (zh + zc) / 2; fz = mz + off[k]
        ax.plot([zh, zc], [y0, y0], color="#D5DBE1", lw=3, zorder=1)
        ax.plot(zh, y0, "o", ms=16, color=BLUE, zorder=3); ax.plot(zc, y0, "s", ms=14, color=LORANGE, zorder=3)
        ax.plot(mz, y0, "D", ms=13, color=INK, mfc=PAPER, mew=3, zorder=4)
        ax.annotate("", (fz, y0 - 0.28), (mz, y0 - 0.28), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=3, mutation_scale=22))
        ax.plot(fz, y0 - 0.28, "o", ms=11, color=INK, zorder=4)
        ok = (fz > 0) == bool(e["label"][k])
        ax.text(5.6, y0 - 0.1, ("có" if fz > 0 else "không") + (", đúng" if ok else ", sai"), fontsize=20,
                color=INK if ok else ORANGE, va="center")
    ax.axvline(0, color=MUTE, lw=2.5, ls="--")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=21)
    ax.set_xlim(-5, 5.2); ax.set_xlabel("Logit (âm là “không”, dương là “có”)"); ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], marker="o", ls="", ms=14, color=BLUE), Line2D([], [], marker="s", ls="", ms=13, color=LORANGE),
          Line2D([], [], marker="D", ls="", ms=12, color=INK, mfc=PAPER, mew=3), Line2D([], [], color=ORANGE, lw=3, marker=">", ms=10)]
    ax.legend(hs, ["H+", "CLIP", "trung bình", "cộng offset"], frameon=False, fontsize=18, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout(); fig.savefig(OUT / "fusion_dots.png", dpi=100); plt.close(fig)


def sigmoid():
    z = np.linspace(-6, 6, 300); fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(z, 1 / (1 + np.exp(-z)), color=BLUE, lw=4); ax.axhline(0.5, color=MUTE, ls="--", lw=2)
    p = 1 / (1 + np.exp(-3.41)); ax.plot(3.41, p, "o", ms=16, color=ORANGE)
    ax.annotate(f"logit +3.41\nxác suất {p:.2f}", (3.41, p), xytext=(-190, -70), textcoords="offset points", fontsize=19,
                arrowprops=dict(arrowstyle="-", color=INK))
    ax.set_xlabel("Logit"); ax.set_ylabel("Xác suất có mũ"); ax.set_yticks([0, 0.5, 1])
    fig.tight_layout(); fig.savefig(OUT / "sigmoid.png", dpi=100); plt.close(fig)


if __name__ == "__main__":
    loss_curves(); rule_hist(); fusion_dots(); sigmoid(); anchor_hist()
    print("done")
