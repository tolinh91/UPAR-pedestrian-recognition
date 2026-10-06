"""Fit the instance rescorer on all leave-one-out features and export it as plain numpy trees for run.py.

python rescore_export.py ../runs/lodo/feats_{peta,pa100k,market}.npz --out ../runs/lodo/rescorer.npz
Features (70): 17 keypoint maxima, 17 log1p flip-disagreement distances (heatmap cells), 17 mass/peak ratios,
17 log1p heatmap spreads, log(box area + 1), box aspect w/h; ensemble features (_ens.npz) add 17 log1p distances
between the large and huge keypoints. Target: OKS of the instance (leaf values include the lr).
"""
import argparse

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

K = 17


def features(X):
    F = X[:, :4 * K + 2].astype(np.float64).copy()
    F[:, K:2 * K] = np.log1p(F[:, K:2 * K])
    F[:, 3 * K:4 * K] = np.log1p(F[:, 3 * K:4 * K])
    if X.shape[1] > 4 * K + 4:  # model disagreement columns after the 4 summary columns
        F = np.concatenate([F, np.log1p(X[:, 4 * K + 4:].astype(np.float64))], 1)
    return F


def predict_numpy(f, feat, thr, left, right, leaf, value, base):
    """Same traversal run.py uses: every tree in lockstep, nodes indexed per tree."""
    T = feat.shape[0]
    node = np.zeros((len(f), T), np.int64)
    t = np.arange(T)[None]
    for _ in range(64):
        is_leaf = leaf[t, node]
        if is_leaf.all():
            break
        go_left = f[np.arange(len(f))[:, None], feat[t, node]] <= thr[t, node]
        node = np.where(is_leaf, node, np.where(go_left, left[t, node], right[t, node]))
    return base + value[t, node].sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("feats", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    zs = [np.load(f) for f in a.feats]
    X = features(np.concatenate([z["X"] for z in zs]))
    y = np.nan_to_num(np.concatenate([z["oks"] for z in zs]))
    m = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_leaf_nodes=7, min_samples_leaf=100,
                                      early_stopping=False, random_state=0).fit(X, y)
    trees = [p[0].nodes for p in m._predictors]
    n = max(len(t) for t in trees)
    pad = lambda field, dt: np.stack([np.pad(t[field].astype(dt), (0, n - len(t))) for t in trees])
    arr = dict(feat=pad("feature_idx", np.int64), thr=pad("num_threshold", np.float64), left=pad("left", np.int64),
               right=pad("right", np.int64), leaf=pad("is_leaf", bool), value=pad("value", np.float64),
               base=np.float64(np.ravel(m._baseline_prediction)[0]))
    ref, mine = m.predict(X[:5000]), predict_numpy(X[:5000], **arr)
    print(f"{len(trees)} trees, max {n} nodes; numpy vs sklearn max abs diff {np.abs(ref - mine).max():.2e}")
    np.savez(a.out, **arr)


if __name__ == "__main__":
    main()
