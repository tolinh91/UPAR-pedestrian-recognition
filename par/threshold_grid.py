"""Threshold rules on leave-one-dataset-out runs: logits + tau * theory + s * fitted offsets.

theory = log((1-p)/p) - (1-2p) with p the positive rate of the training data (label-shift correction towards the
balanced decision that mA rewards, minus the part DeepMAR weighting already applied); fitted = per-attribute offsets
fit on the seen datasets' val. python threshold_grid.py ../runs/kdL_s2 ../runs/kdL_s3 ...
"""
import sys
from pathlib import Path

import numpy as np

from calibrate import fit_offsets
from upar.data import attr_names, load_split
from upar.metrics import challenge_metrics

TAUS = [0.0, 0.3, 0.42, 0.5, 0.6]
SCALES = [0.0, 0.2, 0.4, 0.6]

if __name__ == "__main__":
    A = attr_names()
    tr, va = load_split("train"), load_split("val")
    tot = np.zeros((len(TAUS), len(SCALES)))
    for run in map(Path, sys.argv[1:]):
        s, h = np.load(run / "seen_val_logits_last.npz"), np.load(run / "held_val_logits_last.npz")
        held = va.ds.values[h["idx"][0]]
        off, _ = fit_offsets(s["logits"], va[A].values[s["idx"]], va.ds.values[s["idx"]])
        p = tr[tr.ds != held][A].values.mean(0).clip(1e-3, 1 - 1e-3)
        theory = np.log((1 - p) / p) - (1 - 2 * p)
        gt, ds = va[A].values[h["idx"]], va.ds.values[h["idx"]]
        g = np.array([[challenge_metrics(gt, 1 / (1 + np.exp(-(h["logits"] + t * theory + sc * off))), ds)[0]["Challenge_Avg"]
                       for sc in SCALES] for t in TAUS])
        tot += g
        print(run.name, held, "best", f"{g.max():.4f}", "at tau", TAUS[g.argmax() // len(SCALES)], "s", SCALES[g.argmax() % len(SCALES)],
              "| x0.6 only", f"{g[0, 3]:.4f}", "| tau0.42 only", f"{g[2, 0]:.4f}", flush=True)
    tot /= len(sys.argv[1:])
    print("mean over runs (rows tau, cols s):", SCALES)
    for t, r in zip(TAUS, tot):
        print(f"tau {t:4.2f}", " ".join(f"{v:.4f}" for v in r))
