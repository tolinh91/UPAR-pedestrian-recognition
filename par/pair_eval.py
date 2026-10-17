"""Held-out Challenge_Avg of T1 ensembles from cached no-flip logits (LODO folds PETA / Market, threshold 0.5, no offset).

python pair_eval.py --seeds 1 2
Baseline = v30 type: H+ HL32 + CLIP TL32 (fusion_extract caches in /dev/shm/feats). Candidates from logit_extract caches
in /dev/shm/logits: CLIP TL32 / TR32 with WiSE alphas next to HL32, and the dual model (DU32) heads h, c, j and their means.
"""
import argparse
from pathlib import Path

import numpy as np

from upar.data import attr_names, load_split
from upar.metrics import challenge_metrics

FEATS, LOGITS = Path("/dev/shm/feats"), Path("/dev/shm/logits")
FOLDS = {"PETA": "PETA", "Market": "Market1501"}


def score(z, gt, grp):
    m, _ = challenge_metrics(gt, 1 / (1 + np.exp(-z)), grp)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    a = ap.parse_args()
    va = load_split("val"); A = attr_names()
    gt_all, grp_all = va[A].values, va.ds.values
    table = {}
    for s in a.seeds:
        for f, h in FOLDS.items():
            rows = np.where(grp_all == h)[0]
            gt, grp = gt_all[rows], grp_all[rows]
            cand = {}
            hl = FEATS / f"HL32_{f}_s1.npz"   # H+ seed 1 is the partner for every CLIP variant
            tl = FEATS / f"TL32_{f}_s{s}.npz"
            if hl.exists():
                zH = np.load(hl)["logits"][rows]
                cand["H+ alone"] = zH
                if tl.exists():
                    cand["L0: HL32 + TL32 (v30 type)"] = (zH + np.load(tl)["logits"][rows]) / 2
                for kind in ("TL32", "TR32"):
                    p = LOGITS / f"{kind}_{f}_s{s}.npz"
                    if p.exists():
                        z = np.load(p)
                        for k in sorted(z.files):
                            if k.startswith("logits_w"):
                                cand[f"HL32 + {kind} wise {k[8:]}"] = (zH + z[k][rows]) / 2
                                cand[f"{kind} wise {k[8:]} alone"] = z[k][rows]
            p = LOGITS / f"DU32_{f}_s{s}.npz"
            if p.exists():
                z = np.load(p)
                d = {k: z[f"{k}_w1.00"][rows] for k in "hcj"}
                cand["DU h (H+ head)"] = d["h"]; cand["DU c (CLIP head)"] = d["c"]; cand["DU j (joint query)"] = d["j"]
                cand["DU mean(h, c)"] = (d["h"] + d["c"]) / 2
                cand["DU mean(h, c, j)"] = (d["h"] + d["c"] + d["j"]) / 3
                cand["DU mean(h, j)"] = (d["h"] + d["j"]) / 2
                cand["DU mean(c, j)"] = (d["c"] + d["j"]) / 2
                if hl.exists() and tl.exists():
                    cand["L0 + DU j (3 members)"] = (zH + np.load(tl)["logits"][rows] + d["j"]) / 3
            for name, z in cand.items():
                m = score(z, gt, grp)
                table.setdefault(name, {})[(s, f)] = (m["Challenge_Avg"], m["mA"], m["Instance_F1"], (z > 0).sum(1).mean())
    keys = [(s, f) for s in a.seeds for f in FOLDS]
    print(f"{'candidate':34s} " + " ".join(f"{f[:6]}s{s}" for s, f in keys) + "   mean(PETA,Market) per seed")
    for name, r in table.items():
        vals = " ".join(f"{r[k][0]:.4f} " if k in r else "   -    " for k in keys)
        means = " ".join(f"s{s} {np.mean([r[(s, f)][0] for f in FOLDS]):.4f}" for s in a.seeds if all((s, f) in r for f in FOLDS))
        print(f"{name:34s} {vals}  {means}")


if __name__ == "__main__":
    main()
