"""Where does the cross-dataset threshold error come from, and can unlabelled target images fix it?

For every leave-one-dataset-out run (needs seen_val_logits_last.npz and held_val_logits_last.npz from offset_proxy):
  source   offsets fit on the seen datasets' val, x0.6 (what the submission does)
  em       + label-prior shift estimated by EM on the unlabelled held-out images (Saerens et al. 2002), per attribute
  oracle   offsets fit on the held-out labels themselves (upper bound for any per-attribute threshold fix)
python prior_shift.py ../runs/b_q_PETA_s1 ../runs/b_q_Market ...
"""
import sys
from pathlib import Path

import numpy as np

from calibrate import fit_offsets
from upar.data import attr_names, load_split
from upar.metrics import challenge_metrics


def sig(x):
    return 1 / (1 + np.exp(-x))


def em_prior(p, prior_s, iters=200):
    """Target positive rate per attribute from posteriors p (N, K) calibrated to the source prior prior_s (K,)."""
    pt = prior_s.copy()
    for _ in range(iters):
        w1, w0 = pt / prior_s, (1 - pt) / (1 - prior_s)
        q = w1 * p / (w1 * p + w0 * (1 - p))
        pt = q.mean(0).clip(1e-4, 1 - 1e-4)
    return pt


def main(runs):
    A = attr_names()
    va = load_split("val")
    print(f"{'run':24s} {'source':>7s} {'em x.5':>7s} {'em x1':>7s} {'oracle':>7s}  gap closed (em x.5)")
    for run in map(Path, runs):
        s, h = np.load(run / "seen_val_logits_last.npz"), np.load(run / "held_val_logits_last.npz")
        sgt, sds = va[A].values[s["idx"]], va.ds.values[s["idx"]]
        gt, ds = va[A].values[h["idx"]], va.ds.values[h["idx"]]
        off, _ = fit_offsets(s["logits"], sgt, sds)
        orc, _ = fit_offsets(h["logits"], gt, ds)
        prior_s = sig(s["logits"]).mean(0).clip(1e-4, 1 - 1e-4)       # the model's own prior on seen data
        prior_t = em_prior(sig(h["logits"]), prior_s)
        delta = np.log(prior_t / (1 - prior_t)) - np.log(prior_s / (1 - prior_s))

        def avg(shift):
            return challenge_metrics(gt, sig(h["logits"] + shift), ds)[0]["Challenge_Avg"]

        src, e5, e1, o = avg(0.6 * off), avg(0.6 * off + 0.5 * delta), avg(0.6 * off + delta), avg(orc)
        print(f"{run.name:24s} {src:.4f} {e5:.4f} {e1:.4f} {o:.4f}  {(e5 - src) / max(o - src, 1e-9):+.0%}")


if __name__ == "__main__":
    main(sys.argv[1:])
