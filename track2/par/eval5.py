"""Held-out Challenge_Avg of the 16/10 runs (no flip, threshold 0.5) next to the numbers recorded on 15-16/10.

python eval5.py [logit dir]
HLR + TLR = H+ and CLIP-T trained with logit_adj_rule (the v25 decision learned in training);
DTB = dual_tr model (H+ + CLIP region head + joint query with branch dropout): its heads and their means.
"""
import sys
from pathlib import Path

import numpy as np

from upar.data import attr_names, load_split
from upar.metrics import challenge_metrics

D = Path(sys.argv[1] if len(sys.argv) > 1 else "../logits")
REF = {"v30 type: HL32 + TL32 (15/10)": {"PETA": 0.8245, "Market": 0.8630},
       "DU mean of 3 heads (15/10)": {"PETA": 0.8255, "Market": 0.8637},
       "DU joint query alone (15/10)": {"PETA": 0.8210, "Market": 0.8641}}

va = load_split("val"); A = attr_names()
gt_all, grp_all = va[A].values, va.ds.values
res = {k: dict(v) for k, v in REF.items()}
extra = {}
for f, h in {"PETA": "PETA", "Market": "Market1501"}.items():
    rows = np.where(grp_all == h)[0]
    gt, grp = gt_all[rows], grp_all[rows]

    def sc(z, name):
        m = challenge_metrics(gt, 1 / (1 + np.exp(-z)), grp)[0]
        res.setdefault(name, {})[f] = m["Challenge_Avg"]
        extra.setdefault(name, {})[f] = (m["mA"], m["Instance_F1"], (z > 0).sum(1).mean())

    z = {}
    for k in ("HLR", "TLR"):
        p = D / f"{k}_{f}_s1.npz"
        if p.exists():
            z[k] = np.load(p)["logits_w1.00"][rows]
            sc(z[k], f"{k} alone")
    if len(z) == 2:
        sc((z["HLR"] + z["TLR"]) / 2, "HLR + TLR (v25 rule in training)")
    p = D / f"DTB_{f}_s1.npz"
    if p.exists():
        d = np.load(p)
        parts = {k: d[f"{k}_w1.00"][rows] for k in "hcj"}
        sc(parts["j"], "DTB joint query alone")
        sc((parts["h"] + parts["c"]) / 2, "DTB mean(h, c)")
        sc((parts["h"] + parts["c"] + parts["j"]) / 3, "DTB mean of 3 heads")
base = REF["v30 type: HL32 + TL32 (15/10)"]
print(f"{'candidate':36s} {'PETA':>7s} {'Market':>7s} {'mean':>7s} {'vs v30':>7s}   mA / F1 / positives per image")
for name, r in res.items():
    if len(r) < 2:
        print(f"{name:36s} " + " ".join(f"{r.get(f, float('nan')):7.4f}" for f in ("PETA", "Market")))
        continue
    m = (r["PETA"] + r["Market"]) / 2
    e = extra.get(name, {})
    det = "  ".join(f"{f[:4]} {e[f][0]:.4f}/{e[f][1]:.4f}/{e[f][2]:.2f}" for f in e)
    print(f"{name:36s} {r['PETA']:7.4f} {r['Market']:7.4f} {m:7.4f} {m - (base['PETA'] + base['Market']) / 2:+7.4f}   {det}")
