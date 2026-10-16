"""Results and fusion charts of the v25 deck with the plain decision (probability > 0.5, no prior offset).

../../../Upar-Challenge-Track-2/.venv/bin/python make_nooffset_figs.py
results_dots.png  fold mean (PETA and Market held out) of H+ (v21 run) paired with each CLIP variant, 2 CLIP seeds,
                  threshold 0 on the mean logit (remote_results/rare/pair_matrix.py 0)
fusion_dots.png   per-attribute logits of the PETA example 32671: H+, CLIP (query + text), their mean, verdict at 0
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

OUT = Path(__file__).parent / "figs"
INK, PAPER, MUTE, BLUE, ORANGE, LORANGE = "#14202E", "#F5F3EE", "#6B7684", "#2F7FB5", "#B85A1E", "#F0A868"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 20, "axes.edgecolor": MUTE, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER})
nums = json.load(open(OUT / "numbers.json")); A = nums["attrs"]

# fold means at threshold 0 (pair_matrix.py 0: HB + B / TI / T / Tw70)
res = [("v21", "CLIP gốc", 0.8330), ("v23", "+ nhánh văn bản + inject", 0.8319),
       ("v24", "+ nhánh văn bản", 0.8330), ("v25", "+ nhánh văn bản + WiSE-FT", 0.8314)]
fig, ax = plt.subplots(figsize=(12, 5))
lo = 0.8305
for i, (v, d, m) in enumerate(res[::-1]):
    c = ORANGE if v == "v25" else BLUE
    ax.plot([lo, m], [i, i], color="#D5DBE1", lw=6, solid_capstyle="round"); ax.plot(m, i, "o", ms=20, color=c)
    ax.text(m + 0.00008, i, f"  {m:.4f}", va="center", fontsize=21, color=INK)
ax.set_yticks(range(4)); ax.set_yticklabels([f"{v}  {d}" for v, d, _ in res[::-1]], fontsize=21)
ax.set_xlim(lo, 0.8338); ax.set_xticks([0.8305, 0.8315, 0.8325, 0.8335])
ax.set_xlabel("Trung bình hai fold (cặp với H+ của v21)")
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
fig.tight_layout(); fig.savefig(OUT / "results_dots_noff.png", dpi=100); plt.close(fig)

# fusion of the PETA example: mean of the two members' logits, decision at 0
e = nums["images"]["32671"]
rows = [("Nữ", "Gender-Female"), ("Balo", "Accessory-Backpack"), ("Mũ", "Accessory-Hat"),
        ("Kính thường", "Accessory-Glasses-Normal"), ("Trưởng thành", "Age-Adult"), ("Hói", "Hair-Length-Bald")]
fig, ax = plt.subplots(figsize=(14, 6.6))
for i, (name, a) in enumerate(rows):
    k = A.index(a); y0 = len(rows) - 1 - i
    zh = e["hp_query"][k]; zc = e["clip_query"][k] + e["clip_text"][k]; mz = (zh + zc) / 2
    ax.plot([zh, zc], [y0, y0], color="#D5DBE1", lw=3, zorder=1)
    ax.plot(zh, y0, "o", ms=17, color=BLUE, zorder=3); ax.plot(zc, y0, "s", ms=15, color=LORANGE, zorder=3)
    ax.plot(mz, y0, "D", ms=15, color=INK, zorder=4)
    ok = (mz > 0) == bool(e["label"][k])
    ax.text(5.4, y0, ("có" if mz > 0 else "không") + (" · đúng" if ok else " · sai"), fontsize=22,
            color=INK if ok else ORANGE, va="center", fontweight="normal" if ok else "bold")
    print(f"{name:14s} H+ {zh:+.2f} CLIP {zc:+.2f} mean {mz:+.2f} label {e['label'][k]:.0f}")
ax.axvline(0, color=MUTE, lw=2.5, ls="--")
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=22)
ax.set_xlim(-5, 8.2); ax.set_xticks([-4, -2, 0, 2, 4]); ax.set_xlabel("Logit (âm là “không”, dương là “có”)")
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
hs = [Line2D([], [], marker="o", ls="", ms=15, color=BLUE), Line2D([], [], marker="s", ls="", ms=14, color=LORANGE),
      Line2D([], [], marker="D", ls="", ms=14, color=INK)]
ax.legend(hs, ["logit H+", "logit CLIP", "trung bình"], frameon=False, fontsize=20, ncol=3, loc="upper left",
          bbox_to_anchor=(0, 1.12))
fig.tight_layout(); fig.savefig(OUT / "fusion_dots_noff.png", dpi=100); plt.close(fig)
print("ok")
