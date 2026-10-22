"""Extra numbers for the Track 2 paper (numpy on saved held-out logits, no model runs).

../../../.venv/bin/python t2_extra.py      (writes t2_extra.json)
1. the pair used in the kNN experiments of 2026-10-11 (vessl_t1/runs/P_dinoHp224 + P_clipL224, PETA held out), to tie
   knn_logs_2026-10-11.txt to the saved logits;
2. the distilled ViT-L student (only trained for the PETA fold, 10 epochs: runs/remote2/kd2_a08_e10_PETA) added to the
   pair of the submitted type (HB + Tw70);
3. kNN smoothing with neighbours found by the cosine of the predicted probability vectors (the ablation that needs no
   image features), on HB + Tw70, both folds, at the submitted k = 50, alpha = 0.2.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import t2_numbers as T  # noqa: E402  (re-running it is cheap; it also rewrites t2_numbers.json)

RV = Path("/media/tan/F/RWS2027/remote_results/vessl_t1/runs")
STU = Path("/media/tan/F/RWS2027/Upar-Challenge-Track-2/runs/remote2/kd2_a08_e10_PETA/held_val_logits_last.npz")


def npz(f):
    d = np.load(f)
    return dict(zip(d["idx"], d["logits"].astype(np.float64)))


def knn_prob(z, k=50, a=0.2):
    p = 1 / (1 + np.exp(-z)); e = p / np.linalg.norm(p, axis=1, keepdims=True)
    S = e @ e.T; np.fill_diagonal(S, -np.inf)
    nn = np.argpartition(-S, k, axis=1)[:, :k]
    return (1 - a) * z + a * z[nn].mean(1)


out = {}
idx, z = T.stack([npz(RV / "P_dinoHp224/held_val_logits_last.npz"), npz(RV / "P_clipL224/held_val_logits_last.npz")])
out["knn_pair_PETA"] = T.madm(z, idx, full=True)
hb = T.load("HB_s1", ""); st = npz(STU)
vals2, vals3 = [], []
for tw in T.runs_of("Tw70", ""):
    idx, z = T.stack([hb, tw]); vals2.append(T.madm(z, idx))
    idx, z = T.stack([hb, tw, st]); vals3.append(T.madm(z, idx))
out["student_PETA"] = {"pair": float(np.mean(vals2)), "pair+student": float(np.mean(vals3))}
for fold, (pre, _) in T.FOLDS.items():
    base, sm = [], []
    for tw in T.runs_of("Tw70", pre):
        idx, z = T.stack([T.load("HB_s1", pre), tw])
        base.append(T.madm(z, idx)); sm.append(T.madm(knn_prob(z), idx))
    out.setdefault("knn_probvec", {})[fold] = {"base": float(np.mean(base)), "smoothed": float(np.mean(sm))}
json.dump(out, open(Path(__file__).with_name("t2_extra.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
