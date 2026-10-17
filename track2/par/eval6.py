"""Held-out Challenge_Avg (no flip, threshold 0.5) of joint-model runs: mean of all heads, each joint head alone.

python eval6.py [logit dir]     References recorded earlier: v30 type 0.8245 / 0.8630, DU 0.8255 / 0.8637, DTB 0.8253 / 0.8644
"""
import sys
from pathlib import Path

import numpy as np

from upar.data import attr_names, load_split
from upar.metrics import challenge_metrics

D = Path(sys.argv[1] if len(sys.argv) > 1 else "../logits")
REF = {"v30 type (HL32 + TL32)": (0.8245, 0.8630), "DU mean of heads": (0.8255, 0.8637), "DTB mean of heads": (0.8253, 0.8644)}
va = load_split("val"); A = attr_names(); gt_all, grp_all = va[A].values, va.ds.values
FOLDS = {"PETA": "PETA", "Market": "Market1501"}
rows_res = {k: {"PETA": v[0], "Market": v[1]} for k, v in REF.items()}
for kind in sorted({p.name.split("_")[0] for p in D.glob("*_s1.npz")}):
    for f, h in FOLDS.items():
        p = D / f"{kind}_{f}_s1.npz"
        if not p.exists():
            continue
        d = np.load(p); rows = np.where(grp_all == h)[0]; gt, grp = gt_all[rows], grp_all[rows]
        parts = {k[:-6]: d[k][rows] for k in d.files if k.endswith("_w1.00") and k != "logits_w1.00"}
        if not parts:
            parts = {"member": d["logits_w1.00"][rows]}
        cands = {f"{kind} mean of {len(parts)} heads": np.mean(list(parts.values()), 0)}
        for j in ("j", "j2"):
            if j in parts:
                cands[f"{kind} {j} alone"] = parts[j]
        for name, z in cands.items():
            rows_res.setdefault(name, {})[f] = challenge_metrics(gt, 1 / (1 + np.exp(-z)), grp)[0]["Challenge_Avg"]
b = sum(REF["v30 type (HL32 + TL32)"]) / 2
print(f"{'candidate':28s} {'PETA':>7s} {'Market':>7s} {'mean':>7s} {'vs v30':>7s}")
for name, r in rows_res.items():
    if len(r) == 2:
        m = (r["PETA"] + r["Market"]) / 2
        print(f"{name:28s} {r['PETA']:7.4f} {r['Market']:7.4f} {m:7.4f} {m - b:+7.4f}")
    else:
        print(f"{name:28s} " + " ".join(f"{r.get(f, float('nan')):7.4f}" for f in FOLDS))
