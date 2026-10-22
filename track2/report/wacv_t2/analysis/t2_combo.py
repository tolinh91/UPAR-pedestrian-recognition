"""Group cross-entropy + kNN smoothing (+ query expansion) on both held-out folds, from compact.npz (one seed each).
../../../.venv/bin/python t2_combo.py -> t2_combo.json"""
import json
import sys
from pathlib import Path

import numpy as np

src = open(Path(__file__).with_name("t2_gallery.py")).read().split("C = np.load(")[0]
sys.argv = ["x"]; exec(src)
C = np.load("/media/tan/F/RWS2027/remote_results/t2k/compact.npz")
out = {}
for fold, (pre, hold) in FOLDS.items():
    rows = C[f"{fold}_rows"]; G = Gallery(hold, rows)
    lg = {n: C[f"{fold}_{n}"].astype(np.float64) for n in ("HB", "HG", "TG")}
    cd = saved("T_s1@0.70", pre); clip1 = np.stack([cd[r] for r in rows])
    I = C[f"{fold}_ft_mean_nn"].astype(np.int64); Ip = C[f"{fold}_pre_mean_nn"].astype(np.int64)
    E = C[f"{fold}_ft_mean_pca64"].astype(np.float32); E /= np.linalg.norm(E, axis=1, keepdims=True)
    base = (lg["HB"] + clip1) / 2; gce = (lg["HG"] + lg["TG"]) / 2
    r = {}
    for name, z in (("base", base), ("gce", gce)):
        r[name] = G.metrics(z)
        r[name + " +knn k50 a0.2"] = G.metrics(smooth(z, I, .2))
        r[name + " +knn pretrained k50 a0.2"] = G.metrics(smooth(z, Ip, .2))
        r[name + " +knn +qe m1 b1"] = G.metrics(S=qe_scores(smooth(z, I, .2), E, 1, 1))
    out[fold] = {k: {m: float(v) for m, v in d.items()} for k, d in r.items()}
    print(fold, {k: round(100 * d["mADM"], 2) for k, d in out[fold].items()}, flush=True)
json.dump(out, open(Path(__file__).with_name("t2_combo.json"), "w"), indent=1)
