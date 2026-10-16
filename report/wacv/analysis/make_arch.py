"""Architecture figure: submitted configuration (encoders, patch grids, per-block learning rates) plus real outputs of
the ViT-L members on one unseen PETA image (token PCA, query attention, logits) from arch_data.json."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from PIL import Image

HERE = Path(__file__).parent; FIG = HERE.parent / "figs"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
                     "font.size": 7, "axes.linewidth": 0.4, "pdf.fonttype": 42})
BLUE, ORANGE, INK, MUTED, GREY = "#1F6FB4", "#D9661F", "#222222", "#8A8A8A", "#BDBDBD"
d = json.load(open(HERE / "arch_data.json"))
img = np.asarray(Image.open(HERE / "arch_input.png")).astype(float) / 255; H, W = img.shape[:2]
NAMES = {"LowerBody-Type-Skirt&Dress": "skirt", "Accessory-Hat": "hat", "UpperBody-Length-Short": "short sleeves",
         "LowerBody-Color-Black": "black lower", "Gender-Female": "female", "Hair-Length-Long": "long hair",
         "Accessory-Bag": "bag", "Accessory-Backpack": "backpack"}
names = [NAMES[a] for a in d["attrs"]]
ROWS = (("D", BLUE, "DINOv3 ViT-H+/16", 32, (14, 7)), ("C", ORANGE, "CLIP ViT-L/14", 24, (16, 8)))

fig = plt.figure(figsize=(6.9, 2.75))
gs = fig.add_gridspec(2, 11, width_ratios=[1, 0.62, 1, 2.1, 1, 0.25, 1, 1, 1, 1, 0.05], wspace=0.08, hspace=0.22,
                      left=0.005, right=0.725, top=0.80, bottom=0.07)
def clean(ax, col=None):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.6); s.set_color(col or "#555555")

axs = {}
ax = fig.add_subplot(gs[:, 0]); ax.imshow(img); clean(ax); ax.set_title("input\n224$\\times$112", fontsize=6.5, pad=3); axs["in"] = ax
for r, (key, col, name, L, (gh, gw)) in enumerate(ROWS):
    ax = fig.add_subplot(gs[r, 2]); ax.imshow(img); clean(ax, col)
    for i in range(1, gh): ax.axhline(i * H / gh, color="white", lw=0.3)
    for j in range(1, gw): ax.axvline(j * W / gw, color="white", lw=0.3)
    ax.set_ylabel(f"{name}\n{gh}$\\times${gw} patches", fontsize=6.3, color=col, labelpad=2)
    axs[f"pat{r}"] = ax
    # per-block learning rate of the submitted recipe: blocks 1-8 frozen, then 0.75^(L-l)
    ax = fig.add_subplot(gs[r, 3]); axs[f"blk{r}"] = ax
    l = np.arange(1, L + 1); lr = np.where(l <= 8, 0, 0.75 ** (L - l))
    ax.bar(l[l <= 8], 0.06, width=0.8, color=GREY)
    ax.bar(l[l > 8], lr[l > 8], width=0.8, color=col)
    ax.set_xlim(0.3, 32.7); ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["0", r"$\eta$"], fontsize=5.5)
    ax.set_xticks([1, 8, L]); ax.tick_params(axis="x", labelsize=5.5, length=1.5, pad=1); ax.tick_params(axis="y", length=1.5, pad=1)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.text(4.5, 0.14, "frozen", fontsize=5.5, ha="center", color=MUTED)
    ax.text(L - 6, 0.82, r"$\eta\,0.75^{L-l}$", fontsize=6, ha="right", color=col)
    if r == 1: ax.set_xlabel("transformer block $l$", fontsize=6, labelpad=0)
    ax = fig.add_subplot(gs[r, 4]); ax.imshow(np.array(d[f"pca{key}"]), interpolation="nearest", aspect="auto"); clean(ax, col)
    axs[f"pca{r}"] = ax
    for k in range(4):
        ax = fig.add_subplot(gs[r, 6 + k]); ax.imshow(img)
        a = np.array(d[f"att{key}"][k]); a = a / a.max()
        up = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)) / 255
        ax.imshow(up, cmap="magma", alpha=0.62, vmin=0, vmax=1); clean(ax, col)
        if r == 0: ax.set_title(names[k], fontsize=6.5, pad=2)
        axs[f"q{r}{k}"] = ax
axs["pca0"].set_title("patch tokens\n(PCA)", fontsize=6.5, pad=2)

# combination and decision rule
ax = fig.add_axes([0.825, 0.16, 0.165, 0.58])
y = np.arange(4)[::-1] * 1.0
zD, zC = np.array(d["zD"]), np.array(d["zC"]); zr = (zD + zC) / 2 + d["tau"] * np.array(d["b"])
ax.barh(y + 0.27, zD, height=0.22, color=BLUE, label="DINOv3")
ax.barh(y + 0.03, zC, height=0.22, color=ORANGE, label="CLIP")
ax.barh(y - 0.21, zr, height=0.22, color=INK, label=r"mean + $\tau b_k$")
ax.axvline(0, color=INK, lw=0.6)
ax.set_yticks(y); ax.set_yticklabels([f"{n} ({'yes' if l else 'no'})" for n, l in zip(names, d["labels"])], fontsize=6.3)
ax.set_xlim(-5, 5); ax.set_xticks([-4, 0, 4]); ax.tick_params(labelsize=5.8)
ax.set_xlabel("logit (yes if > 0)", fontsize=6.3, labelpad=1)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(fontsize=5.5, frameon=False, loc="upper center", bbox_to_anchor=(0.42, 1.2), ncol=3, handlelength=1.0, columnspacing=0.6)

fig.canvas.draw()
P = {k: a.get_position() for k, a in axs.items()}
def arrow(x0, x1, yy):
    fig.add_artist(FancyArrowPatch((x0, yy), (x1, yy), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=7, lw=0.7, color=INK))
ymid = (P["pat0"].y0 + P["pat1"].y1) / 2
arrow(P["in"].x1 + 0.004, P["pat0"].x0 - 0.004, ymid)
arrow(P["pca0"].x1 + 0.004, P["q00"].x0 - 0.004, ymid)
arrow(P["q03"].x1 + 0.004, 0.745, ymid)
TOP = 0.955
fig.text((P["pat0"].x0 + P["pca0"].x1) / 2, TOP, "encoder: patch embedding and transformer blocks, 8 frozen, layer-wise lr decay",
         fontsize=6.5, ha="center", va="top")
fig.text((P["q00"].x0 + P["q03"].x1) / 2, TOP, "query head: 40 learned queries,\ncross-attention to the patch tokens",
         fontsize=6.5, ha="center", va="top")
fig.text(0.885, TOP, "equal-weight logit mean\nand one-parameter rule", fontsize=6.5, ha="center", va="top")
fig.savefig(FIG / "architecture.pdf"); fig.savefig(FIG / "architecture.png", dpi=220)
print("ok")
