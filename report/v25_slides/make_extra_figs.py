"""Dataset strips, results dot chart and the fixed fusion chart for the restructured v25 deck."""
import io, json, sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/media/tan/F/RWS2027/Upar-Challenge-Track-2/par")
from upar.data import attr_names, image_cache, load_split
from make_visual_figs import INK, PAPER, MUTE, BLUE, ORANGE, LORANGE, OUT, nums, A
va = load_split("val"); buf, off = image_cache("val"); rng = np.random.default_rng(7)
for ds, name in (("PETA", "peta"), ("PA100k", "pa100k"), ("Market1501", "market")):
    rows = rng.choice(np.where(va.ds.values == ds)[0], 4, replace=False)
    strip = Image.new("RGB", (4 * 160 + 3 * 12, 320), PAPER)
    for i, r in enumerate(rows):
        im = Image.open(io.BytesIO(buf[off[r]:off[r + 1]].tobytes())).convert("RGB")
        if i == 0: print(ds, "native size", im.size)
        strip.paste(im.resize((160, 320), Image.BICUBIC), (i * 172, 0))
    strip.save(OUT / f"ds_{name}.png")
# results: fold mean of the pair (H+ of v21 + CLIP variant), 2 CLIP seeds
res = [("v21", "CLIP gốc", 0.8434, "0.7997"), ("v23", "+ nhánh văn bản + inject", 0.84355, "0.8000"),
       ("v24", "+ nhánh văn bản", 0.8438, "chưa nộp"), ("v25", "+ nhánh văn bản + WiSE-FT", 0.8443, "[ __ ]")]
fig, ax = plt.subplots(figsize=(12, 5))
for i, (v, d, m, lb) in enumerate(res[::-1]):
    c = ORANGE if v == "v25" else BLUE
    ax.plot([0.8430, m], [i, i], color="#D5DBE1", lw=6, solid_capstyle="round"); ax.plot(m, i, "o", ms=20, color=c)
    ax.text(m + 0.00004, i, f"  {m:.4f}", va="center", fontsize=21, color=INK)
ax.set_yticks(range(4)); ax.set_yticklabels([f"{v}  {d}" for v, d, _, _ in res[::-1]], fontsize=21)
ax.set_xlim(0.8430, 0.8447); ax.set_xlabel("Trung bình hai fold (cặp với H+ của v21)")
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
ax.set_xticks([0.8430, 0.8435, 0.8440, 0.8445])
fig.tight_layout(); fig.savefig(OUT / "results_dots.png", dpi=100); plt.close(fig)
# fusion chart with room for the verdict column
e = nums["images"]["32671"]; offv = np.array(nums["offset"])
rows = [("Nữ", "Gender-Female"), ("Balo", "Accessory-Backpack"), ("Mũ", "Accessory-Hat"), ("Kính thường", "Accessory-Glasses-Normal"),
        ("Trưởng thành", "Age-Adult"), ("Hói", "Hair-Length-Bald")]
fig, ax = plt.subplots(figsize=(14, 6.6))
for i, (name, a) in enumerate(rows):
    k = A.index(a); y0 = len(rows) - 1 - i
    zh = e["hp_query"][k]; zc = e["clip_query"][k] + e["clip_text"][k]; mz = (zh + zc) / 2; fz = mz + offv[k]
    ax.plot([zh, zc], [y0, y0], color="#D5DBE1", lw=3, zorder=1)
    ax.plot(zh, y0, "o", ms=17, color=BLUE, zorder=3); ax.plot(zc, y0, "s", ms=15, color=LORANGE, zorder=3)
    ax.plot(mz, y0, "D", ms=14, color=INK, mfc=PAPER, mew=3, zorder=4)
    ax.annotate("", (fz, y0 - 0.3), (mz, y0 - 0.3), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=3, mutation_scale=24))
    ax.plot(fz, y0 - 0.3, "o", ms=12, color=INK, zorder=4)
    ok = (fz > 0) == bool(e["label"][k])
    ax.text(5.5, y0 - 0.12, ("có" if fz > 0 else "không") + (" · đúng" if ok else " · sai"), fontsize=22,
            color=INK if ok else ORANGE, va="center", fontweight="bold" if not ok else "normal")
ax.axvline(0, color=MUTE, lw=2.5, ls="--")
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=22)
ax.set_xlim(-5, 8.2); ax.set_xticks([-4, -2, 0, 2, 4]); ax.set_xlabel("Logit (âm là “không”, dương là “có”)")
ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
from matplotlib.lines import Line2D
hs = [Line2D([], [], marker="o", ls="", ms=15, color=BLUE), Line2D([], [], marker="s", ls="", ms=14, color=LORANGE),
      Line2D([], [], marker="D", ls="", ms=13, color=INK, mfc=PAPER, mew=3), Line2D([], [], color=ORANGE, lw=3)]
ax.legend(hs, ["logit H+", "logit CLIP", "trung bình", "cộng offset"], frameon=False, fontsize=20, ncol=4, loc="upper left", bbox_to_anchor=(0, 1.12))
fig.tight_layout(); fig.savefig(OUT / "fusion_dots.png", dpi=100); plt.close(fig)
print("ok")
