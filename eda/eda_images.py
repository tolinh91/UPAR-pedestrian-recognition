"""EDA on the downloaded images: existence, sizes, aspect ratios, sample grids per attribute."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / "UPAR-Challenge-2027"
DATA = ROOT / "data"
OUT = ROOT / "eda" / "out"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)


def load(split):
    df = pd.read_csv(DATA / f"annotations/task1/{split}/gt.csv")
    df.columns = ["image"] + list(df.columns[1:])
    df["ds"] = df.image.str.split("/").str[0]
    df["split"] = split
    return df


df = pd.concat([load("train"), load("val")], ignore_index=True)
ATTR = list(df.columns[1:41])

# 1. Existence + sizes
df["exists"] = [(DATA / p).exists() for p in df.image]
print("missing files per ds/split:\n", df[~df.exists].groupby(["split", "ds"]).size())
sizes = []
for p in df.image[df.exists]:
    with Image.open(DATA / p) as im:
        sizes.append(im.size)
df.loc[df.exists, "w"] = [s[0] for s in sizes]
df.loc[df.exists, "h"] = [s[1] for s in sizes]
df["ar"] = df.h / df.w
print(df.groupby(["split", "ds"])[["w", "h", "ar"]].describe(percentiles=[.05, .5, .95]).round(2).T.to_string())
df.to_csv(OUT / "meta.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for ds, g in df.groupby("ds"):
    ax[0].hist(g.h.dropna(), bins=60, alpha=.5, label=ds, range=(0, 600))
    ax[1].hist(g.ar.dropna(), bins=60, alpha=.5, label=ds, range=(0.5, 5))
ax[0].set_title("height (px)"); ax[1].set_title("aspect ratio h/w"); ax[0].legend()
fig.savefig(OUT / "sizes.png", dpi=110, bbox_inches="tight")

# 2. Random sample grid per dataset
fig, axes = plt.subplots(3, 12, figsize=(18, 7))
for r, (ds, g) in enumerate(df[df.exists].groupby("ds")):
    for c, p in enumerate(g.image.sample(12, random_state=0)):
        axes[r, c].imshow(Image.open(DATA / p).convert("RGB")); axes[r, c].axis("off")
    axes[r, 0].set_title(ds, loc="left")
fig.savefig(OUT / "samples.png", dpi=110, bbox_inches="tight")

# 3. Positive examples per attribute (one row per attribute, mixed datasets)
for chunk in range(0, 40, 10):
    fig, axes = plt.subplots(10, 9, figsize=(12, 26))
    for r, a in enumerate(ATTR[chunk:chunk + 10]):
        for k, ds in enumerate(["Market1501", "PA100k", "PETA"]):
            pos = df[(df[a] == 1) & (df.ds == ds) & df.exists]
            picks = pos.image.sample(min(3, len(pos)), random_state=0) if len(pos) else []
            for j in range(3):
                axx = axes[r, k * 3 + j]; axx.axis("off")
                if j < len(picks):
                    axx.imshow(Image.open(DATA / list(picks)[j]).convert("RGB"))
            axes[r, k * 3].set_title(f"{a} | {ds} (n={len(pos)})" if k == 0 else f"{ds} (n={len(pos)})",
                                     fontsize=7, loc="left")
    fig.savefig(OUT / f"attr_{chunk:02d}.png", dpi=90, bbox_inches="tight")
    plt.close(fig)

# 4. Attribute co-occurrence (correlation) on train
corr = df[df.split == "train"][ATTR].corr()
fig, ax = plt.subplots(figsize=(13, 11))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(40)); ax.set_xticklabels(ATTR, rotation=90, fontsize=7)
ax.set_yticks(range(40)); ax.set_yticklabels(ATTR, fontsize=7)
fig.colorbar(im); fig.savefig(OUT / "corr.png", dpi=100, bbox_inches="tight")

# 5. Label conflicts: same Market identity with different labels, duplicates across splits
mk = df[df.ds == "Market1501"].copy()
mk["pid"] = mk.image.str.split("/").str[-1].str[:4]
nuniq = mk.groupby(["split", "pid"])[ATTR].nunique().max(1)
print("Market ids with inconsistent labels:", (nuniq > 1).sum(), "/", len(nuniq))
tr_ids = set(mk[mk.split == "train"].pid); va_ids = set(mk[mk.split == "val"].pid)
print("Market id overlap train/val:", len(tr_ids & va_ids))
print("image overlap train/val:", len(set(df[df.split == "train"].image) & set(df[df.split == "val"].image)))
print("saved to", OUT)
