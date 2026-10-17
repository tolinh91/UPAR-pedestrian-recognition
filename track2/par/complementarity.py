"""Can we tell, without labels of the unseen domain, which backbone complements DINOv3-L best?

Target: AUC gain of (DINOv3-L + X) over DINOv3-L on the held-out dataset (PETA).
Predictors computed on the seen datasets' val only (what we could use for the hidden test):
  seen gain   AUC gain of the pair on seen val
  err corr    correlation of the two models' errors (sigmoid(logit) - label) on labelled seen cells
  logit corr  correlation of the two models' centred logits (needs no labels at all)
  alone       X's own seen-val AUC
python complementarity.py ../runs/remote2 T_dinoL_PETA T_clipL_PETA sig2L_176 ...
"""
import sys
from pathlib import Path

import numpy as np

from probe_backbones import mean_auc
from upar.data import attr_names, load_split

A = attr_names()
va = load_split("val")


def load(root, name, split):
    return np.load(root / name / f"{split}_val_logits_last.npz")


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]


def corr_valid(a, b, valid):
    a, b = a[valid], b[valid]
    return np.corrcoef(a - a.mean(), b - b.mean())[0, 1]


if __name__ == "__main__":
    root, base, others = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    sb, hb = load(root, base, "seen"), load(root, base, "held")
    sgt, hgt = va[A].values[sb["idx"]].astype(float), va[A].values[hb["idx"]].astype(float)
    sds = va.ds.values[sb["idx"]]
    svalid = np.zeros_like(sgt, bool)  # labelled cells: attribute annotated in that dataset
    for d in np.unique(sds):
        svalid[sds == d] = sgt[sds == d].any(0)
    auc_s0, auc_h0 = mean_auc(sgt, sb["logits"], sgt.any(0)), mean_auc(hgt, hb["logits"], hgt.any(0))
    sig = lambda z: 1 / (1 + np.exp(-z))
    rows = []
    for name in others:
        s, h = load(root, name, "seen"), load(root, name, "held")
        target = mean_auc(hgt, (hb["logits"] + h["logits"]) / 2, hgt.any(0)) - auc_h0
        seen_gain = mean_auc(sgt, (sb["logits"] + s["logits"]) / 2, sgt.any(0)) - auc_s0
        err = corr_valid(sig(sb["logits"]) - sgt, sig(s["logits"]) - sgt, svalid)
        lc = corr_valid(sb["logits"], s["logits"], np.ones_like(svalid))
        hl = corr_valid(hb["logits"], h["logits"], np.ones_like(hgt, bool))  # unlabelled target images: also usable
        alone = mean_auc(sgt, s["logits"], sgt.any(0))
        rows.append((name, target, seen_gain, err, lc, hl, alone))
    print(f"{'run':18s} {'PETA gain':>9s} {'seen gain':>9s} {'err corr':>8s} {'logit corr':>10s} {'tgt logit corr':>14s} {'alone':>6s}")
    for r in sorted(rows, key=lambda r: -r[1]):
        print(f"{r[0]:18s} {r[1]:+9.4f} {r[2]:+9.4f} {r[3]:8.3f} {r[4]:10.3f} {r[5]:14.3f} {r[6]:6.4f}")
    t = np.array([r[1] for r in rows])
    for j, nm in [(2, "seen gain"), (3, "err corr"), (4, "logit corr"), (5, "tgt logit corr"), (6, "alone")]:
        print(f"Spearman(PETA gain, {nm:14s}) = {spearman(t, np.array([r[j] for r in rows])):+.2f}")
