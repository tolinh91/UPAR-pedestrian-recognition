"""Greedy teacher-set selection (in the spirit of KAGES, Liu & Jiang 2026): start from the base model and keep adding
the candidate whose logit average raises the seen-domain val AUC most; stop when nothing adds >= min_gain.
Only seen-domain labels are used, so the same rule can pick the teachers for the hidden test.
The held-out (PETA) AUC of every prefix is printed to check the rule.
python select_teachers.py ../runs/remote2 T_dinoL_PETA T_clipL_PETA so400m_168 ...
"""
import sys
from pathlib import Path

import numpy as np

from probe_backbones import mean_auc
from upar.data import attr_names, load_split

A = attr_names()
va = load_split("val")

if __name__ == "__main__":
    root, base, cands = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    load = lambda n, s: np.load(root / n / f"{s}_val_logits_last.npz")
    S = {n: load(n, "seen")["logits"] for n in [base] + cands}
    H = {n: load(n, "held")["logits"] for n in [base] + cands}
    z = load(base, "seen"); h = load(base, "held")
    sgt, hgt = va[A].values[z["idx"]], va[A].values[h["idx"]]
    auc = lambda gt, members, D: mean_auc(gt, np.mean([D[m] for m in members], 0), gt.any(0))
    chosen, left = [base], list(cands)
    cur = auc(sgt, chosen, S)
    print(f"{'+'.join(chosen):60s} seen {cur:.4f}  PETA {auc(hgt, chosen, H):.4f}")
    while left:
        gains = {c: auc(sgt, chosen + [c], S) - cur for c in left}
        best = max(gains, key=gains.get)
        if gains[best] < float(0.0005):
            print(f"stop: best next {best} adds {gains[best]:+.4f} on seen val")
            break
        chosen.append(best); left.remove(best); cur += gains[best]
        print(f"{'+'.join(chosen):60s} seen {cur:.4f}  PETA {auc(hgt, chosen, H):.4f}  (+{gains[best]:.4f})")
