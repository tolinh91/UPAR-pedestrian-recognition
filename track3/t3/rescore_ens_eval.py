"""LODO AP: huge alone vs large+huge ensemble keypoints, each with mean-max score and with a rescorer (fit on the other two)."""
import numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor
from rescore_eval import ap
from rescore_export import features
P = dict(max_iter=150, learning_rate=0.05, max_leaf_nodes=7, min_samples_leaf=100, early_stopping=False, random_state=0)
for tag in ("", "_ens"):
    d = {n: dict(np.load(f"../runs/lodo/feats_{n}{tag}.npz")) for n in ("peta", "pa100k", "market")}
    r0, r1 = [], []
    for held, z in d.items():
        tr = [d[n] for n in d if n != held]
        m = HistGradientBoostingRegressor(**P).fit(features(np.concatenate([t["X"] for t in tr])),
                                                   np.nan_to_num(np.concatenate([t["oks"] for t in tr])))
        r0.append(ap(held, z, z["X"][:, 4 * 17 + 2])); r1.append(ap(held, z, m.predict(features(z["X"]))))
        print(f"{tag or 'huge'} {held}: mean-max {r0[-1]:.4f} rescored {r1[-1]:.4f}", flush=True)
    print(f"{tag or 'huge'} MEAN: mean-max {np.mean(r0):.4f} rescored {np.mean(r1):.4f}", flush=True)
