"""Leave-one-dataset-out test of learned instance scores: fit on two datasets' held-out features, score the third.

python rescore_eval.py ../runs/h_lodo_{name}/feats_{name}.npz ...   (one file per dataset)
Prints OKS AP with the current score (mean keypoint maximum), the learned scores and the oracle (true OKS).
"""
import contextlib
import io
import json
import sys

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from pose import ANN, SIGMAS


def ap(name, z, score):
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(ANN / f"train_{name}.json"))
        preds = [{"image_id": int(i), "category_id": 1, "score": float(s), "keypoints": k.tolist()}
                 for i, s, k in zip(z["image_id"], score, z["keypoints"])]
        ev = COCOeval(gt, gt.loadRes(preds), "keypoints")
        ev.params.kpt_oks_sigmas = SIGMAS
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return ev.stats[0]


def fit_mlp(X, y, seed=0, epochs=300):
    """Small MLP regressing OKS (sigmoid output, BCE on soft targets); standardised inputs. Returns a predict fn."""
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xt, yt = torch.tensor((X - mu) / sd), torch.tensor(y)[:, None]
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 64), torch.nn.GELU(), torch.nn.Linear(64, 1))
    opt = torch.optim.AdamW(net.parameters(), 3e-3, weight_decay=1e-3)
    for _ in range(epochs):
        for b in torch.randperm(len(Xt)).split(1024):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Xt[b]), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
    return lambda Z: torch.sigmoid(net(torch.tensor((Z - mu) / sd))).detach().numpy()[:, 0]


def main(files):
    data = {f.split("feats_")[1].split(".npz")[0]: dict(np.load(f)) for f in files}
    for z in data.values():
        z["oks"] = np.nan_to_num(z["oks"], nan=0.0).astype(np.float32)
    res = {}
    for held, z in data.items():
        tr = [data[n] for n in data if n != held]
        X, y = np.concatenate([t["X"] for t in tr]), np.concatenate([t["oks"] for t in tr])
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.linear_model import Ridge
        gbm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15).fit(X, y)
        lin = Ridge(1.0).fit(X, y)
        mlp = fit_mlp(X, y)
        r = {"current (mean max)": ap(held, z, z["X"][:, -2]),
             "ridge": ap(held, z, lin.predict(z["X"])),
             "gbm": ap(held, z, gbm.predict(z["X"])),
             "mlp": ap(held, z, mlp(z["X"])),
             "oracle (true OKS)": ap(held, z, z["oks"])}
        res[held] = r
        print(held, " ".join(f"{k} {v:.4f}" for k, v in r.items()), flush=True)
    print("mean", {k: round(float(np.mean([r[k] for r in res.values()])), 4) for k in next(iter(res.values()))})


if __name__ == "__main__":
    main(sys.argv[1:])
