"""Choose the threshold strategy on the leave-one-dataset-out proxy, mirroring the submission pipeline.

The proxy model never saw the held-out dataset. We
  1. predict val of the seen datasets and fit per-attribute offsets there (what calibrate.py does for a submission)
  2. apply them to the held-out dataset with a shrink factor s  ->  logits + s * offset
  3. compare with a one-parameter rule from label-shift theory ->  logits + tau * (log((1-p)/p) - (1-2p))
     (p = training prior, 1-2p = shift already applied by DeepMAR weighting)

python offset_proxy.py ../runs/ft_PETA_dinov3
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from calibrate import fit_offsets
from probe_backbones import mean_auc
from train import predict
from upar.data import UPARDataset, attr_names, build_transforms, load_split
from upar.metrics import challenge_metrics, fmt
from upar.models import build_model, cfg_from_checkpoint


def predict_rows(cfg, ck, rows):
    model = build_model(cfg, pretrained=False).cuda()
    model.load_state_dict(ck["model"])
    norm = {k: cfg[k] for k in ("mean", "std") if cfg.get(k)}
    ds = UPARDataset("val", build_transforms(tuple(cfg["size"]), False, **norm), rows=rows)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    return predict(model, DataLoader(ds, 128, num_workers=3), dtype=dtype)


def main(run, ckpt="best.pt"):
    """ckpt: best.pt is selected on the held-out dataset itself (optimistic); last.pt is the honest proxy score."""
    run = Path(run)
    ck = torch.load(run / ckpt, map_location="cpu", weights_only=False)
    tag = "" if ckpt == "best.pt" else "_" + Path(ckpt).stem
    cfg, held = cfg_from_checkpoint(ck), ck["args"]["holdout"]
    A = attr_names()
    tr, va = load_split("train"), load_split("val")

    # 1. logits on val of the datasets the proxy model was trained on
    cache = run / f"seen_val_logits{tag}.npz"
    if not cache.exists():
        lg, idx = predict_rows(cfg, ck, np.where(va.ds.values != held)[0])
        np.savez(cache, logits=lg, idx=idx)
    z = np.load(cache)
    seen_gt, seen_ds = va[A].values[z["idx"]], va.ds.values[z["idx"]]
    off, _ = fit_offsets(z["logits"], seen_gt, seen_ds)
    print("offsets fit on seen-domain val:", fmt(challenge_metrics(seen_gt, 1 / (1 + np.exp(-(z["logits"] + off))), seen_ds)[0]))

    # 2. held-out logits saved by train.py
    if ckpt == "best.pt":
        h = np.load(run / "val_logits.npz")
    else:
        hcache = run / f"held_val_logits{tag}.npz"
        if not hcache.exists():
            lg, idx = predict_rows(cfg, ck, np.where(va.ds.values == held)[0])
            np.savez(hcache, logits=lg, idx=idx)
        h = np.load(hcache)
    gt, grp = va[A].values[h["idx"]], va.ds.values[h["idx"]]
    print(f"held-out {held}: AUC {mean_auc(gt, h['logits'], gt.any(0)):.4f}")

    def score(shift):
        return challenge_metrics(gt, 1 / (1 + np.exp(-(h["logits"] + shift))), grp)[0]

    rows = []
    for s in [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0]:
        rows.append(("fitted offsets x %.2f" % s, score(s * off)))
    p = (tr[tr.ds != held][A].values.mean(0)).clip(1e-3, 1 - 1e-3)
    theory = np.log((1 - p) / p) - (1 - 2 * p)
    for tau in [0.1, 0.2, 0.3, 0.42, 0.5]:
        rows.append(("theory tau = %.2f" % tau, score(tau * theory)))
    print(f"\n{'strategy':24s} {'Avg':>7s} {'mA':>7s} {'F1':>7s} {'Prec':>7s} {'Rec':>7s}")
    for name, m in rows:
        print(f"{name:24s} {m['Challenge_Avg']:.4f} {m['mA']:.4f} {m['Instance_F1']:.4f} "
              f"{m['Instance_Prec']:.4f} {m['Instance_Rec']:.4f}")
    json.dump({n: m for n, m in rows}, open(run / f"offset_proxy{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:3])
