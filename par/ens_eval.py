"""How much does a second backbone add to DINOv3-L? Average logits on the leave-PETA-out proxy (offsets x0.6).

python ens_eval.py ../runs/remote2 T_dinoL_PETA T_clipL_PETA sig2L_176 ...   (first run = the base)
"""
import sys
from pathlib import Path

import numpy as np

from calibrate import fit_offsets
from probe_backbones import mean_auc
from upar.data import attr_names, load_split
from upar.metrics import challenge_metrics

A = attr_names()
va = load_split("val")


def score(runs):
    s = [np.load(r / "seen_val_logits_last.npz") for r in runs]
    h = [np.load(r / "held_val_logits_last.npz") for r in runs]
    sgt, sds = va[A].values[s[0]["idx"]], va.ds.values[s[0]["idx"]]
    gt, ds = va[A].values[h[0]["idx"]], va.ds.values[h[0]["idx"]]
    sl, hl = np.mean([z["logits"] for z in s], 0), np.mean([z["logits"] for z in h], 0)
    off, _ = fit_offsets(sl, sgt, sds)
    m = challenge_metrics(gt, 1 / (1 + np.exp(-(hl + 0.6 * off))), ds)[0]
    return m["Challenge_Avg"], mean_auc(gt, hl, gt.any(0))


if __name__ == "__main__":
    root, base, others = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    print(f"{'run':22s} {'alone T1':>8s} {'AUC':>7s} | {'+ base T1':>9s} {'AUC':>7s}")
    for name in [base] + others:
        a = score([root / name])
        e = score([root / base, root / name]) if name != base else a
        print(f"{name:22s} {a[0]:8.4f} {a[1]:7.4f} | {e[0]:9.4f} {e[1]:7.4f}", flush=True)
