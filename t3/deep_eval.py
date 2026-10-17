"""Leave-one-dataset-out evaluation of rescorers with and without backbone tokens.

python deep_eval.py ../runs/l_lodo_{name} [tag]
Reads ../runs/l_lodo_<name>/deep_<name>.npz for the three datasets (each written by the model that never saw that
dataset), fits each rescorer on two datasets and reports OKS AP on the third, keypoints unchanged.
"""
import contextlib, copy, io, json, sys, time
T0 = time.time()
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from pose import ANN, SIGMAS

pattern = sys.argv[1] if len(sys.argv) > 1 else "../runs/l_lodo_{name}"
NAMES = ("peta", "pa100k", "market")
D = {n: dict(np.load(Path(pattern.format(name=n)) / f"deep_{n}.npz")) for n in NAMES}
for d in D.values():
    d["oks"] = np.nan_to_num(d["oks"])


def ap(name, d, score):
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(ANN / f"train_{name}.json"))
        preds = [{"image_id": int(i), "category_id": 1, "score": float(s), "keypoints": k.tolist()}
                 for i, s, k in zip(d["image_id"], score, d["keypoints"])]
        ev = COCOeval(gt, gt.loadRes(copy.deepcopy(preds)), "keypoints"); ev.params.kpt_oks_sigmas = SIGMAS
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return ev.stats[0]


def heat(d):
    F = d["X"][:, :70].astype(np.float64).copy()
    F[:, 17:34] = np.log1p(F[:, 17:34]); F[:, 51:68] = np.log1p(F[:, 51:68])
    return F


def gbm():
    return HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_leaf_nodes=7, min_samples_leaf=100,
                                         early_stopping=False, random_state=0)


res = {}
for held in NAMES:
    tr = [D[n] for n in NAMES if n != held]; te = D[held]
    y = np.concatenate([t["oks"] for t in tr])
    Htr, Hte = np.concatenate([heat(t) for t in tr]), heat(te)
    r = {"default (17 max)": ap(held, te, te["X"][:, :17].mean(1)),
         "no ears (15 max)": ap(held, te, np.delete(te["X"][:, :17], [3, 4], 1).mean(1)),
         "GBM heatmap (v6 recipe)": ap(held, te, gbm().fit(Htr, y).predict(Hte))}
    for key in ("G", "K"):
        Ztr = np.concatenate([t[key].astype(np.float32) for t in tr]); Zte = te[key].astype(np.float32)
        sc = StandardScaler().fit(Ztr); Ztr, Zte = sc.transform(Ztr), sc.transform(Zte)
        pca = PCA(32, random_state=0).fit(Ztr); Ptr, Pte = pca.transform(Ztr), pca.transform(Zte)
        r[f"GBM heatmap + PCA32({key})"] = ap(held, te, gbm().fit(np.c_[Htr, Ptr], y).predict(np.c_[Hte, Pte]))
        r[f"ridge {key}"] = ap(held, te, Ridge(alpha=100.0).fit(Ztr, y).predict(Zte))
        hs = StandardScaler().fit(Htr)
        mlp = MLPRegressor(hidden_layer_sizes=(256,), alpha=1e-3, early_stopping=True, max_iter=200, random_state=0)
        r[f"MLP heatmap + {key}"] = ap(held, te, mlp.fit(np.c_[hs.transform(Htr), Ztr], y).predict(np.c_[hs.transform(Hte), Zte]))
    r["oracle (true OKS)"] = ap(held, te, te["oks"])
    res[held] = r
    print(held, f"{time.time() - T0:.0f}s", {k: round(v, 4) for k, v in r.items()}, flush=True)
print("mean", {k: round(float(np.mean([res[h][k] for h in NAMES])), 4) for k in res[NAMES[0]]})
json.dump(res, open(Path(pattern.format(name="peta")).parent / "deep_eval.json", "w"), indent=1)
