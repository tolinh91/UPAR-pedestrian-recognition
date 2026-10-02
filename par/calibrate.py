"""Per-attribute logit offsets that maximise val Challenge_Avg (threshold is fixed at 0.5 by the challenge).

Offsets are fit by coordinate ascent. Honest gain is estimated with 2-fold CV over val identities/images;
the final offsets are fit on the full val set and saved next to the checkpoint.

python par/calibrate.py ../runs/dinov3_b16
"""
import json
import sys
from pathlib import Path

import numpy as np

from upar.data import attr_names, load_split
from upar.metrics import challenge_metrics, fmt

GRID = np.linspace(-2.5, 2.5, 26)


def score(logits, off, gt, groups):
    return challenge_metrics(gt, 1 / (1 + np.exp(-(logits + off))), groups)[0]["Challenge_Avg"]


def fit_offsets(logits, gt, groups, rounds=2):
    off = np.zeros(logits.shape[1])
    best = score(logits, off, gt, groups)
    for _ in range(rounds):
        for a in range(logits.shape[1]):
            cand = off.copy()
            for v in GRID:
                cand[a] = v
                s = score(logits, cand, gt, groups)
                if s > best + 1e-6:
                    best, off[a] = s, v
            cand[a] = off[a]
    return off, best


def main(run):
    run = Path(run)
    z = np.load(run / "val_logits.npz")
    val = load_split("val")
    logits, idx = z["logits"], z["idx"]
    gt, groups = val[attr_names()].values[idx], val.ds.values[idx]

    base = challenge_metrics(gt, 1 / (1 + np.exp(-logits)), groups)[0]
    print("no offset     ", fmt(base))

    # 2-fold CV: Market split by identity so the same person is not in both folds
    key = np.array([p.split("/")[-1][:4] if p.startswith("Market") else p for p in val.image.values[idx]])
    rng = np.random.default_rng(0)
    uk = np.unique(key); fold_of = dict(zip(uk, rng.integers(0, 2, len(uk))))
    fold = np.array([fold_of[k] for k in key])
    cv = []
    for f in (0, 1):
        tr, te = fold != f, fold == f
        off, _ = fit_offsets(logits[tr], gt[tr], groups[tr])
        m0 = challenge_metrics(gt[te], 1 / (1 + np.exp(-logits[te])), groups[te])[0]
        m1 = challenge_metrics(gt[te], 1 / (1 + np.exp(-(logits[te] + off))), groups[te])[0]
        print(f"fold {f} held-out: base {m0['Challenge_Avg']:.4f} -> offsets {m1['Challenge_Avg']:.4f}")
        cv.append(m1["Challenge_Avg"] - m0["Challenge_Avg"])
    print(f"CV gain {np.mean(cv):+.4f}")

    off, best = fit_offsets(logits, gt, groups)
    m = challenge_metrics(gt, 1 / (1 + np.exp(-(logits + off))), groups)[0]
    print("full-val fit  ", fmt(m))
    json.dump({"attribute_names": attr_names(), "offset": off.tolist(), "cv_gain": float(np.mean(cv))},
              open(run / "offsets.json", "w"), indent=1)
    for a, o in zip(attr_names(), off):
        if o != 0:
            print(f"  {a:32s} {o:+.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
