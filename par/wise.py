"""WiSE-FT style weight interpolation between the pretrained backbone and a fine-tuned checkpoint.

theta(alpha) = (1 - alpha) * theta_pretrained + alpha * theta_finetuned   (backbone only; the head is kept)
Scored on the held-out dataset of a leave-one-dataset-out run, or on all of val otherwise.

python par/wise.py ../runs/lodo_base_PETA --alphas 0.5 0.6 0.7 0.8 0.9 1.0
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from upar.data import UPARDataset, build_transforms, load_split
from upar.metrics import challenge_metrics, fmt
from upar.models import build_model, cfg_from_checkpoint
from train import predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    run = Path(args.run)
    ck = torch.load(run / args.ckpt, map_location="cpu", weights_only=False)
    targs = ck["args"]
    cfg = cfg_from_checkpoint(ck)
    size = tuple(cfg["size"])
    holdout = targs.get("holdout")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    val = load_split("val")
    rows = None if holdout is None else np.where(val.ds.values == holdout)[0]
    ds = UPARDataset("val", build_transforms(size, False), rows=rows)
    dl = DataLoader(ds, 128, num_workers=args.workers, pin_memory=True)

    pre = build_model(cfg, pretrained=True).state_dict()
    ft = ck["model"]
    model = build_model(cfg, pretrained=False).cuda()
    res = {}
    for a in args.alphas:
        mixed = {k: (ft[k].float() if not k.startswith("backbone.") or k not in pre
                     else (1 - a) * pre[k].float() + a * ft[k].float()) for k in ft}
        model.load_state_dict(mixed)
        logits, idx = predict(model, dl, "cuda", dtype=dtype)
        m, _ = challenge_metrics(ds.labels[idx], 1 / (1 + np.exp(-logits)), ds.ds[idx])
        res[a] = m
        print(f"alpha={a:.2f} {fmt(m)}", flush=True)
        if a == 1.0:
            np.savez(run / "holdout_logits.npz", logits=logits, idx=idx)
    json.dump({str(k): v for k, v in res.items()}, open(run / "wise.json", "w"), indent=1)


if __name__ == "__main__":
    main()
