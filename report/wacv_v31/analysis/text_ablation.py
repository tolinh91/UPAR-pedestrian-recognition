"""Does the CLIP text score help? UniQuery with (DU32) and without (NT32) the text score, seed 1, both folds, per head.

../../../../Upar-Challenge-Track-2/.venv/bin/python text_ablation.py
Logits: remote_results/duf/logits/DU32_<fold>_s1.npz and remote_results/na/logits/NT32_<fold>_s1.npz
(keys logits_w1.00 = mean of the three heads, h_/c_/j_w1.00 = DINOv3 / CLIP / joint query head).
Decision logit > 0. Only the CLIP query head reads the text score; the DINOv3 query head shows the run-to-run shift.
Also prints per-attribute balanced accuracy changes of the CLIP query head.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/media/tan/F/RWS2027/Upar-Challenge-Track-2/par")
from upar.data import attr_names, load_split  # noqa: E402
from upar.metrics import challenge_metrics  # noqa: E402

R = Path("/media/tan/F/RWS2027/remote_results")
A = attr_names(); va = load_split("val"); GT = va[A].values; DS = va.ds.values
FILES = {"text": "duf/logits/DU32_{f}_s1.npz", "no text": "na/logits/NT32_{f}_s1.npz"}
HEADS = {"mean": "logits_w1.00", "DINOv3": "h_w1.00", "CLIP": "c_w1.00", "joint": "j_w1.00"}


def bacc(y, d):
    v = y >= 0; y, d = y[v], d[v]
    return 0.5 * (d[y == 1].mean() + (~d[y == 0]).mean()) if 0 < y.sum() < len(y) else np.nan


for fold, hold in (("PETA", "PETA"), ("Market", "Market1501")):
    S, Z = {}, {}
    for k, f in FILES.items():
        d = np.load(R / f.format(f=fold)); m = DS[d["idx"]] == hold; idx = d["idx"][m]
        Z[k] = {h: d[key][m].astype(np.float64) for h, key in HEADS.items()}
        S[k] = {h: 100 * challenge_metrics(GT[idx], (z > 0).astype(float), DS[idx], thr=0.5)[0]["Challenge_Avg"]
                for h, z in Z[k].items()}
        S[k + "_idx"] = idx
    assert (S["text_idx"] == S["no text_idx"]).all()
    print(f"\n=== {fold} ({len(S['text_idx'])} images)")
    for h in HEADS:
        print(f"  {h:7s} text {S['text'][h]:.2f}  no text {S['no text'][h]:.2f}  delta {S['text'][h] - S['no text'][h]:+.3f}")
    y = GT[S["text_idx"]]
    g = np.array([bacc(y[:, k], Z["text"]["CLIP"][:, k] > 0) - bacc(y[:, k], Z["no text"]["CLIP"][:, k] > 0)
                  for k in range(40)])
    ok = np.isfinite(g)
    print(f"  CLIP head per-attribute bacc change: up {(g[ok] > 0).sum()} / down {(g[ok] < 0).sum()} of {ok.sum()}, "
          f"mean {100 * np.nanmean(g):+.2f} pts")
