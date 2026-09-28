"""Do per-attribute logit offsets transfer to a dataset they were not fit on?

For each dataset d: fit offsets on the val logits of the other datasets, then score d with the offsets
scaled by several factors. Also scores the offsets fit on all of val (what submission v3 used).

python par/offset_lodo.py ../runs/dinov3_b16
"""
import json
import sys
from pathlib import Path

import numpy as np

from calibrate import fit_offsets
from upar.data import attr_names, load_split
from upar.metrics import challenge_metrics

SCALES = [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0]


def main(run):
    run = Path(run)
    z = np.load(run / "val_logits.npz")
    val = load_split("val")
    logits, idx = z["logits"], z["idx"]
    gt, groups = val[attr_names()].values[idx], val.ds.values[idx]
    full = np.array(json.load(open(run / "offsets.json"))["offset"])

    rows = {}
    for d in np.unique(groups):
        tr, te = groups != d, groups == d
        off, _ = fit_offsets(logits[tr], gt[tr], groups[tr])
        for name, o in [("lodo", off), ("full", full)]:
            for s in SCALES:
                m = challenge_metrics(gt[te], 1 / (1 + np.exp(-(logits[te] + s * o))), groups[te])[0]
                rows[(d, name, s)] = m
        print(f"held-out {d} done", flush=True)

    out = {}
    for name in ["lodo", "full"]:
        print(f"\n== offsets fit on {'the other datasets' if name == 'lodo' else 'all of val (in-domain, optimistic)'}")
        print(f"{'scale':>6} " + " ".join(f"{d[:10]:>22}" for d in np.unique(groups)) + f" {'mean Avg':>9} {'mA':>6} {'F1':>6} {'Prec':>6} {'Rec':>6}")
        for s in SCALES:
            ms = [rows[(d, name, s)] for d in np.unique(groups)]
            cells = " ".join(f"{m['Challenge_Avg']:.4f} (P{m['Instance_Prec']:.2f}/R{m['Instance_Rec']:.2f})" for m in ms)
            mean = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
            print(f"{s:>6} {cells} {mean['Challenge_Avg']:>9.4f} {mean['mA']:>6.4f} {mean['Instance_F1']:>6.4f} "
                  f"{mean['Instance_Prec']:>6.4f} {mean['Instance_Rec']:>6.4f}")
            out[f"{name}_{s}"] = mean
    json.dump(out, open(run / "offset_lodo.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
