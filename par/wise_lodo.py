"""WiSE-FT on a leave-one-dataset-out run: held-out logits for backbone weights interpolated between the pretrained
model and the fine-tuned one, theta(a) = (1 - a) * theta_pretrained + a * theta_finetuned (backbone only; the head and
its text anchors are kept from the fine-tuned checkpoint). Same preprocessing and flip TTA as offset_proxy.py, so a = 1
reproduces held_val_logits_last.npz. Writes <run>/held_val_logits_wise<a>.npz for the pair evaluation done locally.

python wise_lodo.py ../runs/T_s1 --alphas 0.6 0.8 0.9 1.0
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from offset_proxy import predict_rows
from upar.data import load_split
from upar.models import build_model, cfg_from_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.6, 0.8, 0.9, 1.0])
    args = ap.parse_args()
    run = Path(args.run)
    ck = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
    cfg, held = cfg_from_checkpoint(ck), ck["args"]["holdout"]
    rows = np.where(load_split("val").ds.values == held)[0]
    pre = build_model(cfg, pretrained=True).state_dict()
    ft = ck["model"]
    for a in args.alphas:
        mixed = {k: ((1 - a) * pre[k].float() + a * v.float()).to(v.dtype)
                 if k.startswith("backbone.") and k in pre and v.is_floating_point() else v for k, v in ft.items()}
        lg, idx = predict_rows(cfg, {"model": mixed}, rows)
        np.savez(run / f"held_val_logits_wise{a:.2f}.npz", logits=lg, idx=idx)
        print(f"{run.name} alpha {a:.2f}: {len(idx)} held-out {held} images", flush=True)


if __name__ == "__main__":
    main()
