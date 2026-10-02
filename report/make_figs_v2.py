"""Figures of the 2026-10-02 report (track1_report_v2.tex). Data: experiments.json (collect_experiments.py) and the
proxy logits under runs/. Run from report/: python make_figs_v2.py"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "par"))
FIG = Path(__file__).parent / "figs"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e3e2db", "grid.linewidth": .6, "axes.edgecolor": "#9a988f",
                     "axes.axisbelow": True, "savefig.dpi": 220, "savefig.bbox": "tight"})
INK, MUTED, ACC, ORANGE, GRAY = "#121211", "#85837c", "#2a78d6", "#eb6834", "#b4b2a9"
QREF, NOISE = (0.7961 + 0.7956) / 2, 0.0008
EXP = json.load(open(Path(__file__).parent / "experiments.json"))
GROUPS = ["Mốc", "Kiến trúc head", "Loss / cách train", "Dữ liệu / tăng cường", "Backbone / độ phân giải"]


def fig_experiments():
    rows = []
    for g in GROUPS:
        rs = sorted([r for r in EXP if r["group"] == g and r["final"]["off06"]], key=lambda r: -r["final"]["off06"])
        rows += [(g, None)] + [(g, r) for r in rs]
    fig, ax = plt.subplots(figsize=(7.2, 7.6))
    y, yt, yl = 0, [], []
    for g, r in rows:
        if r is None:
            ax.text(0.7853, y, g, fontweight="bold", fontsize=8.3, va="center", color=INK)
        else:
            v = r["final"]["off06"]
            c = ACC if v >= QREF + 0.003 else (INK if v >= QREF - NOISE else GRAY)
            ax.scatter(min(max(v, 0.7862), 0.81), y, s=26, color=c, zorder=3, edgecolor="white", linewidth=.8)
            ax.text(0.8107, y, ("← " if v < 0.7862 else "") + f"{v:.4f}", va="center", fontsize=7.3, color=INK)
            yt.append(y); yl.append(r["label"])
        y -= 1
    ax.axvspan(QREF - NOISE, QREF + NOISE, color=ACC, alpha=.12, lw=0)
    ax.axvline(QREF, color=ACC, lw=.8); ax.axvline(0.7904, color=MUTED, lw=.8)
    ax.text(0.7904, 1.0, "head linear", color=MUTED, fontsize=7.3, ha="center")
    ax.text(QREF, 1.0, "head query", color=ACC, fontsize=7.3, ha="center")
    ax.set_yticks(yt, yl, fontsize=7.6); ax.set_ylim(y + .5, 1.6); ax.set_xlim(0.7853, 0.8155)
    ax.set_xticks(np.arange(0.786, 0.8101, 0.004)); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Challenge_Avg trên PETA (last.pt, offset × 0.6)"); ax.grid(axis="y", visible=False)
    fig.savefig(FIG / "v2_experiments.png"); plt.close(fig)


def fig_convergence():
    by = {r["name"]: r for r in EXP}
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for r in EXP:
        if r["group"] in ("Kiến trúc head", "Loss / cách train", "Dữ liệu / tăng cường") and r["name"] != "q_la" and len(r["epochs"]) == 5:
            axs[0].plot(range(5), [e["avg"] for e in r["epochs"]], color=GRAY, lw=.9)
    sel = (("ft_PETA_dinov3", "Head linear", ORANGE), ("ft_PETA_dinov3_q", "Head query", INK), ("q_L224", "ViT-L 224×112 + query", ACC))
    for name, lab, c in sel:
        a = [e["avg"] for e in by[name]["epochs"]]
        axs[0].plot(range(len(a)), a, color=c, lw=1.8, label=lab)
        k = int(np.argmax(a)); axs[0].scatter(k, a[k], color=c, s=18, zorder=4)
        pts = np.array(by[name]["loss"]); axs[1].plot(pts[:, 0], pts[:, 1], color=c, lw=1.4, label=lab)
    axs[0].set_ylim(0.70, 0.81); axs[0].set_xlabel("epoch"); axs[0].set_ylabel("Avg PETA (chưa offset)")
    axs[0].legend(frameon=False, fontsize=7.3, loc="lower right"); axs[0].set_title("Điểm PETA theo epoch (xám: các biến thể khác)", fontsize=8.5)
    axs[1].set_ylim(0.2, 0.7); axs[1].set_xlabel("epoch"); axs[1].set_ylabel("loss train (BCE + DeepMAR)")
    axs[1].set_title("Loss train trên Market + PA100k", fontsize=8.5)
    fig.tight_layout(); fig.savefig(FIG / "v2_convergence.png"); plt.close(fig)


def fig_gap():
    from upar.data import load_split, attr_names
    from upar.metrics import challenge_metrics
    from calibrate import fit_offsets
    va = load_split("val"); A = attr_names(); GT = va[A].values; DS = va.ds.values
    G = {"Màu áo (12)": list(range(8, 20)), "Màu quần (12)": list(range(21, 33)), "Tuổi (3)": [0, 1, 2],
         "Tóc (3)": [4, 5, 6], "Độc lập (8)": [3, 7, 20, 35, 36, 37, 38, 39]}
    sig = lambda z: 1 / (1 + np.exp(-z))
    def ba(gt, p):
        tp = (p & (gt == 1)).sum(0); fn = (~p & (gt == 1)).sum(0); tn = (~p & (gt == 0)).sum(0); fp = (p & (gt == 0)).sum(0)
        return .5 * (tp / np.maximum(tp + fn, 1) + tn / np.maximum(tn + fp, 1))
    out = {}
    for name, run, seen, held in (("Head linear", "ft_PETA_dinov3", "seen_val_logits.npz", "val_logits.npz"),
                                  ("Head query", "ft_PETA_dinov3_q", "seen_val_logits_last.npz", "held_val_logits_last.npz")):
        s, h = np.load(ROOT / "runs" / run / seen), np.load(ROOT / "runs" / run / held)
        off, _ = fit_offsets(s["logits"], GT[s["idx"]], DS[s["idx"]])
        gt, z, grp = GT[h["idx"]], h["logits"], DS[h["idx"]]
        orc, _ = fit_offsets(z, gt, grp)
        a6 = challenge_metrics(gt, sig(z + .6 * off), grp)[0]["Challenge_Avg"]; ao = challenge_metrics(gt, sig(z + orc), grp)[0]["Challenge_Avg"]
        b6, bo = ba(gt, sig(z + .6 * off) > .5), ba(gt, sig(z + orc) > .5)
        out[name] = {"fitted": a6, "oracle": ao, "groups": {g: float(bo[i].mean() - b6[i].mean()) for g, i in G.items()}}
    json.dump(out, open(FIG / "v2_gap.json", "w"), indent=1, ensure_ascii=False)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1, 1.6]})
    names = list(out); x = np.arange(2)
    axs[0].bar(x - .18, [out[n]["fitted"] for n in names], .32, color=GRAY, label="offset nguồn × 0.6")
    axs[0].bar(x + .18, [out[n]["oracle"] for n in names], .32, color=ACC, label="offset oracle (PETA)")
    for i, n in enumerate(names):
        axs[0].text(i - .18, out[n]["fitted"] + .001, f"{out[n]['fitted']:.4f}", ha="center", fontsize=6.8)
        axs[0].text(i + .18, out[n]["oracle"] + .001, f"{out[n]['oracle']:.4f}", ha="center", fontsize=6.8)
    axs[0].set_xticks(x, names); axs[0].set_ylim(.78, .83); axs[0].legend(frameon=False, fontsize=6.8, loc="upper left")
    axs[0].set_ylabel("Avg trên PETA"); axs[0].set_title("Offset nguồn so với oracle", fontsize=8.5)
    gs = list(out["Head query"]["groups"]); xx = np.arange(len(gs))
    axs[1].bar(xx - .18, [out["Head linear"]["groups"][g] for g in gs], .32, color=ORANGE, label="Head linear")
    axs[1].bar(xx + .18, [out["Head query"]["groups"][g] for g in gs], .32, color=INK, label="Head query")
    axs[1].axhline(0, color=MUTED, lw=.8); axs[1].set_xticks(xx, gs, fontsize=7.2)
    axs[1].set_ylabel("Δ balanced accuracy"); axs[1].legend(frameon=False, fontsize=7)
    axs[1].set_title("Oracle trừ offset nguồn, theo nhóm thuộc tính", fontsize=8.5)
    fig.tight_layout(); fig.savefig(FIG / "v2_gap.png"); plt.close(fig)
    return out


def fig_vitl():
    # ms/img on the server: ViT-B measured (log), ViT-L 176 measured in the evaluation container x 0.82, other ViT-L sizes
    # scaled from ViT-L 176 by the ratio of their CPU forward times (109.0 / 131.9 / 164.7 / 251.2 ms, batch 64, 6 threads)
    rows = [("ViT-B 256×128", 0.7958, 49.5), ("ViT-L 176×96", 0.8005, 98.5), ("ViT-L 192×96", 0.8014, 98.5 * 131.9 / 109.0),
            ("ViT-L 224×112", 0.8071, 98.5 * 164.7 / 109.0), ("ViT-L 256×128", 0.8023, 98.5 * 251.2 / 109.0)]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.5))
    labs = [r[0] for r in rows]; x = np.arange(len(rows)); cols = [GRAY, ACC, MUTED, MUTED, MUTED]
    axs[0].bar(x, [r[1] for r in rows], .55, color=cols)
    for i, r in enumerate(rows):
        axs[0].text(i, r[1] + .0006, f"{r[1]:.4f}", ha="center", fontsize=6.8)
    axs[0].set_ylim(.79, .81); axs[0].set_xticks(x, labs, fontsize=6.6, rotation=18); axs[0].set_title("Avg trên PETA (offset × 0.6)", fontsize=8.5)
    mins = [29248 * r[2] / 1000 / 60 for r in rows]
    axs[1].bar(x, mins, .55, color=cols); axs[1].set_ylim(0, 125)
    axs[1].axhline(60, color=INK, lw=.8, label="giới hạn 60 phút")
    axs[1].axhline(3400 / 60, color=ORANGE, lw=1, label="ngân sách run.py 3400 s")
    axs[1].legend(frameon=False, fontsize=6.6, loc="upper left")
    for i, m in enumerate(mins):
        axs[1].text(i, m + 1.2, f"{m:.0f}", ha="center", fontsize=6.8)
    axs[1].set_xticks(x, labs, fontsize=6.6, rotation=18); axs[1].set_title("Phút cho 29.248 ảnh trên server", fontsize=8.5)
    fig.tight_layout(); fig.savefig(FIG / "v2_vitl.png"); plt.close(fig)


def fig_proxy_test():
    pts = [("ViT-B linear (v4b)", 0.7904, 0.7775, ORANGE), ("ViT-B linear, offset × 1.0 (v5)", 0.7921, 0.7766, ORANGE),
           ("ViT-B query (v8)", 0.7958, 0.7750, INK), ("ViT-L 176 query, train (v9)", 0.8005, 0.7857, ACC),
           ("ViT-L 176 query, train+val (v10)", 0.8005, 0.7838, ACC)]
    fig, ax = plt.subplots(figsize=(4.8, 2.9))
    for lab, p, t, c in pts:
        ax.scatter(p, t, color=c, s=36, edgecolor="white", linewidth=.8, zorder=3)
        ax.annotate(lab, (p, t), xytext=(-6, -10) if "v10" in lab else (5, 3), textcoords="offset points", fontsize=6.8,
                    ha="right" if "v10" in lab else "left")
    ax.set_xlabel("Avg trên proxy PETA"); ax.set_ylabel("Avg trên test (Codabench)")
    ax.set_xlim(.789, .8085); ax.set_ylim(.774, .7875)
    fig.savefig(FIG / "v2_proxy_test.png"); plt.close(fig)


if __name__ == "__main__":
    fig_experiments(); fig_convergence(); fig_vitl(); fig_proxy_test()
    print(json.dumps(fig_gap(), ensure_ascii=False, indent=1))


def fig_metric_tradeoff():
    """mA, IF1, Avg and instance precision / recall against the offset scale (query head, PETA proxy, last.pt)."""
    d = json.load(open(ROOT / "runs" / "ft_PETA_dinov3_q" / "offset_proxy_last.json"))
    rows = sorted((float(k.split("x ")[1]), v) for k, v in d.items() if k.startswith("fitted"))
    s = [r[0] for r in rows]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.5))
    for key, lab, c in (("mA", "mA", ORANGE), ("Instance_F1", "IF1", INK), ("Challenge_Avg", "Avg", ACC)):
        axs[0].plot(s, [r[1][key] for r in rows], marker="o", ms=3.5, color=c, lw=1.6, label=lab)
    k = int(np.argmax([r[1]["Challenge_Avg"] for r in rows]))
    axs[0].axvline(0.6, color=MUTED, lw=.8); axs[0].text(0.62, 0.8075, "hệ số dùng khi nộp", fontsize=6.8, color=MUTED)
    axs[0].set_xlabel("hệ số co offset s"); axs[0].legend(frameon=False, fontsize=7.2, loc="lower right")
    axs[0].set_title("mA tăng, IF1 giảm, Avg phẳng ở giữa", fontsize=8.5)
    for key, lab, c in (("Instance_Prec", "Instance precision", ORANGE), ("Instance_Rec", "Instance recall", INK), ("Label_F1", "Label F1", MUTED)):
        axs[1].plot(s, [r[1][key] for r in rows], marker="o", ms=3.5, color=c, lw=1.6, label=lab)
    axs[1].set_xlabel("hệ số co offset s"); axs[1].legend(frameon=False, fontsize=7.2, loc="center right")
    axs[1].set_title("Dời ngưỡng đổi precision lấy recall", fontsize=8.5)
    fig.tight_layout(); fig.savefig(FIG / "v2_metric_tradeoff.png"); plt.close(fig)
