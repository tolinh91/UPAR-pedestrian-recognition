"""Label-noise detection by cross-fitting + confident learning (Northcutt et al. 2021), on the training split.

Each training image gets an out-of-fold prediction (a model that never saw it). Per attribute k, with
t1_k = mean p over cells labelled 1 and t0_k = mean (1 - p) over cells labelled 0:
  labelled 1 is flagged when 1 - p >= t0_k and p < 0.5   (the model is as sure it is 0 as it is, on average, about real 0s)
  labelled 0 is flagged when p >= t1_k and p > 0.5
Flagged cells are ignored in training (mask, never relabel). Market is split by person id so one person is never in
both folds.

python noise_cv.py --holdout PETA --out ../runs/noise_PETA -- --head query --head-kw n_layers=1 --compile ...
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train import predict
from upar.data import UPARDataset, attr_names, build_transforms, invalid_mask, load_split
from upar.models import build_model, cfg_from_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default=None)
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("train_args", nargs=argparse.REMAINDER, help="after --: extra train.py arguments")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    extra = [a for a in args.train_args if a != "--"]

    tr = load_split("train")
    rows = np.arange(len(tr)) if args.holdout is None else np.where(tr.ds.values != args.holdout)[0]
    key = np.array([p.split("/")[-1][:4] if d == "Market1501" else p for p, d in zip(tr.image.values, tr.ds.values)])
    rng = np.random.default_rng(0)
    uk = np.unique(key[rows]); fold_of = dict(zip(uk, rng.integers(0, args.folds, len(uk))))
    fold = np.array([fold_of[k] for k in key[rows]])

    A = attr_names()
    prob = np.full((len(tr), len(A)), np.nan, dtype=np.float32)
    for f in range(args.folds):
        fit_rows, held_rows = rows[fold != f], rows[fold == f]
        np.save(out / f"fold{f}_rows.npy", fit_rows)
        run = out / f"fold{f}"
        if not (run / "last.pt").exists():
            cmd = [sys.executable, "-u", "train.py", "--train-rows", str(out / f"fold{f}_rows.npy"), "--out", str(run)]
            cmd += ["--holdout", args.holdout] if args.holdout else []
            subprocess.run(cmd + extra, check=True)
        ck = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
        cfg = cfg_from_checkpoint(ck)
        model = build_model(cfg, pretrained=False).cuda(); model.load_state_dict(ck["model"])
        ds = UPARDataset("train", build_transforms(tuple(cfg["size"]), False, mean=cfg["mean"], std=cfg["std"]), rows=held_rows)
        lg, idx = predict(model, DataLoader(ds, 256, num_workers=8))
        prob[idx] = 1 / (1 + np.exp(-lg))

    inv = invalid_mask(tr)
    labels = tr[A].values
    valid = np.ones_like(labels, dtype=bool)
    for d, m in inv.items():
        valid[np.ix_(tr.ds.values == d, m)] = False
    has = ~np.isnan(prob[:, 0])
    flag = np.zeros_like(labels, dtype=bool)
    for k in range(len(A)):
        v = valid[:, k] & has
        p, y = prob[v, k], labels[v, k]
        if (y == 1).sum() == 0 or (y == 0).sum() == 0:
            continue
        t1, t0 = p[y == 1].mean(), (1 - p[y == 0]).mean()
        f = np.zeros(v.sum(), bool)
        f[(y == 1) & (1 - p >= t0) & (p < 0.5)] = True
        f[(y == 0) & (p >= t1) & (p > 0.5)] = True
        flag[np.where(v)[0][f], k] = True

    np.savez(out / "noise_mask.npz", train=flag)
    np.save(out / "oof_prob.npy", prob)
    n = (valid & has[:, None]).sum()
    print(f"flagged {flag.sum()} of {n} labelled cells ({flag.sum() / n:.2%}); images with >=1 flag {flag.any(1).mean():.1%}")
    for d in np.unique(tr.ds.values[has]):
        r = (tr.ds.values == d) & has
        print(f"  {d:11s} {flag[r].sum() / valid[r].sum():.2%} of its cells")
    groups = {"Age": [0, 1, 2], "Gender": [3], "Hair": [4, 5, 6], "UpperColor": list(range(8, 20)),
              "LowerColor": list(range(21, 33)), "LowerType": [33, 34], "Accessory": list(range(35, 40))}
    print("  per group: " + "  ".join(f"{g} {flag[has][:, i].sum() / valid[has][:, i].sum():.2%}" for g, i in groups.items()))


if __name__ == "__main__":
    main()
