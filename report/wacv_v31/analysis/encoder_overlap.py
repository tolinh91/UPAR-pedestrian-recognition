"""Why DINOv3 and CLIP complement each other: error overlap and per-attribute strengths (plain threshold 0.5).

../../../../Upar-Challenge-Track-2/.venv/bin/python encoder_overlap.py
(1) Error overlap on the weighted-BCE runs of Tab. two (remote_results/rare/runs; Market fold runs have prefix M):
    DINOv3 HB_s1 / HP_s1 (two fine-tunes), CLIP-L B_s1 / B_s2 (two seeds). A model errs on a cell (held-out image,
    labelled attribute) when (logit > 0) disagrees with the label. Overlap(A, B) = P(B errs | A errs).
(2) Per-attribute balanced accuracy, DINOv3 alone vs CLIP alone of the seed-2 separate ensemble of Tab. main
    (remote_results/duf2/logits HL32 / TL32, tau 0.32): attributes where one beats the other by more than 1 point.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/media/tan/F/RWS2027/Upar-Challenge-Track-2/par")
from upar.data import attr_names, load_split  # noqa: E402
from upar.metrics import challenge_metrics  # noqa: E402

R = Path("/media/tan/F/RWS2027/remote_results")
A = attr_names(); va = load_split("val"); GT = va[A].values; DS = va.ds.values
FOLDS = {"PETA": ("", "PETA"), "Market": ("M", "Market1501")}


def load(f, key="logits"):
    d = np.load(f); o = np.argsort(d["idx"])
    return d["idx"][o], d[key][o].astype(np.float64)


def score(idx, z):
    return 100 * challenge_metrics(GT[idx], (z > 0).astype(float), DS[idx], thr=0.5)[0]["Challenge_Avg"]


def bacc(y, d):
    v = y >= 0; y, d = y[v], d[v]
    return 0.5 * (d[y == 1].mean() + (~d[y == 0]).mean()) if 0 < y.sum() < len(y) else np.nan


for fold, (pre, hold) in FOLDS.items():
    Z = {n: load(R / "rare/runs" / f"{pre}{n}" / "held_val_logits_last.npz") for n in ("HB_s1", "HP_s1", "B_s1", "B_s2")}
    idx = Z["HB_s1"][0]; assert all((z[0] == idx).all() for z in Z.values()) and set(DS[idx]) == {hold}
    y = GT[idx]; valid = y >= 0
    err = {n: ((z[1] > 0) != (y == 1)) & valid for n, z in Z.items()}

    def ov(a, b):
        return 100 * err[b][err[a]].mean()
    same = {"DINOv3 x2": [ov("HB_s1", "HP_s1"), ov("HP_s1", "HB_s1")], "CLIP x2": [ov("B_s1", "B_s2"), ov("B_s2", "B_s1")]}
    diff = [ov(h, c) for h in ("HB_s1", "HP_s1") for c in ("B_s1", "B_s2")] + \
           [ov(c, h) for h in ("HB_s1", "HP_s1") for c in ("B_s1", "B_s2")]
    print(f"\n=== {fold} ({len(idx)} images) alone: " + " ".join(f"{n} {score(idx, z[1]):.2f}" for n, z in Z.items()))
    print("  error rate %: " + " ".join(f"{n} {100 * e[valid].mean():.2f}" for n, e in err.items()))
    for n, v in same.items():
        print(f"  overlap {n}: {min(v):.1f}-{max(v):.1f}%")
    print(f"  overlap DINOv3 vs CLIP: {min(diff):.1f}-{max(diff):.1f}%")
    sym = lambda a, b: (ov(a, b) + ov(b, a)) / 2
    print(f"  symmetric: DINOv3 x2 {sym('HB_s1', 'HP_s1'):.1f}%  CLIP x2 {sym('B_s1', 'B_s2'):.1f}%  DINOv3 vs CLIP "
          + " ".join(f"{sym(h, c):.1f}" for h in ("HB_s1", "HP_s1") for c in ("B_s1", "B_s2")))

    i2, h2 = load(R / f"duf2/logits/HL32_{fold}_s2.npz", "logits_w1.00")
    _, c2 = load(R / f"duf2/logits/TL32_{fold}_s2.npz", "logits_w1.00")
    m = DS[i2] == hold; i2, h2, c2 = i2[m], h2[m], c2[m]; y2 = GT[i2]
    gap = np.array([bacc(y2[:, k], h2[:, k] > 0) - bacc(y2[:, k], c2[:, k] > 0) for k in range(40)])
    print(f"  Tab. main seed 2: DINOv3 {score(i2, h2):.2f}  CLIP {score(i2, c2):.2f}  ensemble {score(i2, (h2 + c2) / 2):.2f}"
          f" | DINOv3 > CLIP by >1 pt on {(gap > .01).sum()} attrs, CLIP > DINOv3 on {(gap < -.01).sum()} "
          f"(of {np.isfinite(gap).sum()})")
