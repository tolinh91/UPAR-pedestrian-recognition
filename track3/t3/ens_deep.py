"""Ensemble rescorer with backbone tokens: features, leave-one-dataset-out evaluation, export.

python ens_deep.py feats ../runs/l_lodo_peta ../runs/h_lodo_peta peta   -> ../runs/ensdeep_peta.npz
python ens_deep.py eval                                                -> LODO AP of the rescorers
python ens_deep.py export --out ../runs/rescorer_deep.npz              -> trees + scaler + PCA for run.py

Features are computed with the submission's own run.py code (fp16 models, flip test, heatmap average,
run._features = the 87 columns of the v6 rescorer). G is the large model's last-layer tokens averaged over the grid
and over the plain and flipped view. The deep rescorer adds PCA(32) of the standardised G to the 87 columns.
"""
import contextlib, copy, io, json, sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "submissions/t3_ens_rescore_v6"))
import run as R  # noqa: E402
from pose import ANN, SIGMAS, PoseDataset, ViTPose  # noqa: E402
from rescore_feats import oks  # noqa: E402
from rescore_export import predict_numpy  # noqa: E402

NAMES = ("peta", "pa100k", "market")
RUNS = HERE.parent / "runs"
NPC = 32


def load_model(run):
    m = ViTPose(json.load(open(Path(run) / "args.json"))["model"])
    m.load_state_dict(torch.load(Path(run) / "last.pt", map_location="cpu", weights_only=False)["model"])
    return m.cuda().eval().half()


@torch.no_grad()
def tokens_mean(model, x):
    kw = {"dataset_index": torch.zeros(x.shape[0], dtype=torch.long, device=x.device)} if model.plus else {}
    f = lambda v: model.m.backbone.forward_with_filtered_kwargs(v, **kw).feature_maps[-1].float().mean(1)
    return (f(x) + f(x.flip(-1))) / 2


@torch.no_grad()
def feats(lrun, hrun, name):
    ml, mh = load_model(lrun), load_model(hrun)
    R._MODELS, R._TREES = [ml, mh], None
    R.CFG["gpu_half"] = True
    ds = PoseDataset([name], train=False)
    areas = {a["id"]: a["area"] for a in json.load(open(ANN / f"train_{name}.json"))["annotations"]}
    F, G, Y, IID, KP = [], [], [], [], []
    for x, _, _, idx, _ in DataLoader(ds, 128, num_workers=32):
        x = x.cuda()
        h0, h1, per = R._forward(x)
        hm = (h0 + h1) / 2
        xy, mx = R.decode(hm)
        boxes = [ds.items[i][2] for i in idx.tolist()]
        F.append(R._features(hm, mx, h0, h1, per, boxes)); G.append(tokens_mean(ml, x.half()).cpu().numpy())
        xy, mx = xy.cpu().numpy(), mx.cpu().numpy()
        for j, i in enumerate(idx.tolist()):
            path, kp, box, aid, iid, _ = ds.items[i]
            c, s = R.box_to_center_scale(box); M = R.affine(c, s)
            p = xy[j] * np.array([(R.IN_W - 1) / (R.HM_W - 1), (R.IN_H - 1) / (R.HM_H - 1)], np.float32)
            p = np.c_[p, np.ones(R.NK)] @ np.linalg.inv(np.vstack([M, [0, 0, 1]]))[:2].T
            Y.append(oks(p, kp, areas[aid])); IID.append(iid); KP.append(np.c_[p, mx[j]].reshape(-1))
    out = RUNS / f"ensdeep_{name}.npz"
    np.savez(out, F=np.concatenate(F), G=np.concatenate(G).astype(np.float32), oks=np.nan_to_num(np.array(Y, np.float32)),
             image_id=np.array(IID), keypoints=np.array(KP, np.float32))
    print(out, len(Y), "instances, mean OKS", float(np.nanmean(Y)), flush=True)


def ap(name, d, score):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(ANN / f"train_{name}.json"))
        preds = [{"image_id": int(i), "category_id": 1, "score": float(s), "keypoints": k.tolist()}
                 for i, s, k in zip(d["image_id"], score, d["keypoints"])]
        ev = COCOeval(gt, gt.loadRes(copy.deepcopy(preds)), "keypoints"); ev.params.kpt_oks_sigmas = SIGMAS
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return ev.stats[0]


def gbm():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_leaf_nodes=7, min_samples_leaf=100,
                                         early_stopping=False, random_state=0)


def fit_deep(Fs, Gs, y):
    """Standardise G, PCA to NPC components, boosted trees on [F, PCA(G)]. Returns (model, mu, sd, pca_mean, comps)."""
    G = np.concatenate(Gs).astype(np.float64)
    mu, sd = G.mean(0), G.std(0) + 1e-6
    Z = (G - mu) / sd
    pm = Z.mean(0)
    _, _, Vt = np.linalg.svd(Z - pm, full_matrices=False)
    comps = Vt[:NPC]
    X = np.c_[np.concatenate(Fs), (Z - pm) @ comps.T]
    return gbm().fit(X, y), mu, sd, pm, comps


def deep_inputs(F, G, mu, sd, pm, comps):
    return np.c_[F, ((G.astype(np.float64) - mu) / sd - pm) @ comps.T]


def evaluate():
    D = {n: dict(np.load(RUNS / f"ensdeep_{n}.npz")) for n in NAMES}
    res = {}
    for held in NAMES:
        tr = [D[n] for n in NAMES if n != held]; te = D[held]; y = np.concatenate([t["oks"] for t in tr])
        m, *pp = fit_deep([t["F"] for t in tr], [t["G"] for t in tr], y)
        r = {"default (17 max)": ap(held, te, te["F"][:, :17].mean(1)),
             "no ears": ap(held, te, np.delete(te["F"][:, :17], [3, 4], 1).mean(1)),
             "v6 rescorer recipe (87 feats)": ap(held, te, gbm().fit(np.concatenate([t["F"] for t in tr]), y).predict(te["F"])),
             "deep rescorer (87 + PCA32 tokens)": ap(held, te, m.predict(deep_inputs(te["F"], te["G"], *pp))),
             "oracle": ap(held, te, te["oks"])}
        res[held] = r; print(held, {k: round(v, 4) for k, v in r.items()}, flush=True)
    print("mean", {k: round(float(np.mean([res[h][k] for h in NAMES])), 4) for k in res[NAMES[0]]})


def export(out):
    D = [dict(np.load(RUNS / f"ensdeep_{n}.npz")) for n in NAMES]
    m, mu, sd, pm, comps = fit_deep([d["F"] for d in D], [d["G"] for d in D], np.concatenate([d["oks"] for d in D]))
    trees = [p[0].nodes for p in m._predictors]
    n = max(len(t) for t in trees)
    pad = lambda field, dt: np.stack([np.pad(t[field].astype(dt), (0, n - len(t))) for t in trees])
    arr = dict(feat=pad("feature_idx", np.int64), thr=pad("num_threshold", np.float64), left=pad("left", np.int64),
               right=pad("right", np.int64), leaf=pad("is_leaf", bool), value=pad("value", np.float64),
               base=np.float64(np.ravel(m._baseline_prediction)[0]))
    X = deep_inputs(D[0]["F"][:5000], D[0]["G"][:5000], mu, sd, pm, comps)
    print("numpy vs sklearn max abs diff", np.abs(m.predict(X) - predict_numpy(X, **arr)).max())
    np.savez(out, **arr, g_mu=mu, g_sd=sd, g_pm=pm, g_comps=comps)
    print("saved", out)


if __name__ == "__main__":
    if sys.argv[1] == "feats":
        feats(sys.argv[2], sys.argv[3], sys.argv[4])
    elif sys.argv[1] == "eval":
        evaluate()
    else:
        export(sys.argv[sys.argv.index("--out") + 1])
