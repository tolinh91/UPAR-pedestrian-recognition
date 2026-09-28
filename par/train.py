"""Fine-tune a backbone + head on UPAR with a pluggable loss.

Components are chosen by name (see upar/losses.py and upar/models/heads.py):

  python train.py --out ../runs/base                                   # linear head, BCE + DeepMAR (submission v3)
  python train.py --loss asl --loss-kw gamma_neg=4 clip=0.05 --out ../runs/asl
  python train.py --loss logit_adj --loss-kw tau=0.5 smooth=0.05 --out ../runs/la
  python train.py --head query --head-kw n_layers=1 --out ../runs/query
  python train.py --holdout PETA --epochs 5 ...                        # leave-one-dataset-out proxy
  python train.py --list                                               # show registered losses / heads
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from timm.utils import ModelEmaV3
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from upar.data import UPARDataset, build_transforms, invalid_mask, load_split
from upar.losses import LOSSES, build_loss
from upar.metrics import challenge_metrics, fmt
from upar.models import HEADS, build_model
from upar.registry import parse_kv


def get_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("model")
    g.add_argument("--backbone", default="vit_base_patch16_dinov3.lvd1689m")
    g.add_argument("--size", type=int, nargs=2, default=[256, 128])
    g.add_argument("--head", default="linear", help=f"one of {HEADS.names()}")
    g.add_argument("--head-kw", nargs="*", default=[], metavar="K=V")
    g = ap.add_argument_group("loss")
    g.add_argument("--loss", default="bce", help=f"one of {LOSSES.names()}")
    g.add_argument("--loss-kw", nargs="*", default=["smooth=0.05", "weighting=deepmar"], metavar="K=V",
                   help="shared: smooth, weighting (none|deepmar|sqrt|inverse), max_weight; plus loss-specific")
    g = ap.add_argument_group("optimisation")
    g.add_argument("--epochs", type=int, default=10)
    g.add_argument("--bs", type=int, default=64)
    g.add_argument("--lr", type=float, default=5e-5, help="lr of the last backbone block")
    g.add_argument("--head-lr-mult", type=float, default=10.0)
    g.add_argument("--wd", type=float, default=0.05)
    g.add_argument("--layer-decay", type=float, default=0.75)
    g.add_argument("--warmup", type=float, default=0.5, help="warmup epochs")
    g.add_argument("--ema", type=float, default=0.9995)
    g.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16"],
                   help="auto: bf16 on Ampere+ (A100, L4), fp16 + GradScaler on older GPUs (T4, V100)")
    g.add_argument("--grad-ckpt", action="store_true")
    g = ap.add_argument_group("data")
    g.add_argument("--train-splits", nargs="+", default=["train"])
    g.add_argument("--holdout", default=None, help="dataset left out of training; val is scored on it only")
    g.add_argument("--ds-alpha", type=float, default=0.5, help="dataset sampling ~ N^alpha (0 = equal)")
    g.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 2))
    g.add_argument("--prefetch", type=int, default=4, help="batches prefetched per worker (lower = less RAM)")
    g = ap.add_argument_group("run")
    g.add_argument("--out", default="../runs/base")
    g.add_argument("--resume", action="store_true", help="continue from <out>/state.pt if it exists")
    g.add_argument("--max-iters", type=int, default=0, help="smoke test: stop each epoch after N iterations")
    g.add_argument("--list", action="store_true", help="print registered losses and heads and exit")
    args = ap.parse_args(argv)
    args.head_kw, args.loss_kw = parse_kv(args.head_kw), parse_kv(args.loss_kw)
    return args


def model_cfg(args):
    """What is needed to rebuild the network (stored in checkpoints, read by make_submission / wise)."""
    return {"backbone": args.backbone, "size": list(args.size), "head": args.head, "head_kw": args.head_kw}


# ---------------------------------------------------------------- data
def build_data(args):
    size = tuple(args.size)

    def rows(split, keep):
        if args.holdout is None:
            return None
        return np.where((load_split(split).ds.values != args.holdout) == keep)[0]

    inv = invalid_mask(load_split("train"))
    train_sets = [UPARDataset(s, build_transforms(size, True), inv, rows=rows(s, True)) for s in args.train_splits]
    weights = np.concatenate([d.sample_weights(args.ds_alpha) for d in train_sets])
    sampler = WeightedRandomSampler(weights, sum(len(d) for d in train_sets), replacement=True)
    train_dl = DataLoader(ConcatDataset(train_sets), args.bs, sampler=sampler, num_workers=args.workers,
                          pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=args.prefetch)

    val_rows = rows("val", False)
    if args.max_iters:
        val_rows = (np.arange(len(load_split("val"))) if val_rows is None else val_rows)[::50]
    val_ds = UPARDataset("val", build_transforms(size, False), rows=val_rows)
    val_dl = DataLoader(val_ds, 128, num_workers=args.workers, pin_memory=True)

    labels = np.concatenate([d.labels[d.idx] for d in train_sets])
    valid = np.concatenate([d.valid[d.idx] for d in train_sets])
    prior = (labels * valid).sum(0) / valid.sum(0).clip(1)  # fraction of positives per attribute
    return train_dl, val_ds, val_dl, prior


# ---------------------------------------------------------------- loops
@torch.no_grad()
def predict(model, loader, device="cuda", flip=True, dtype=torch.bfloat16):
    """Logits (hflip TTA averaged) and dataset row indices for every image of loader."""
    model.eval()
    logits, idx = [], []
    for x, _, _, j in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=dtype):
            z = model(x).float()
            if flip:
                z = (z + model(torch.flip(x, dims=[3])).float()) / 2
        logits.append(z.cpu()); idx.append(j)
    return torch.cat(logits).numpy(), torch.cat(idx).numpy()


class CosineWithWarmup:
    def __init__(self, opt, total, warmup):
        self.opt, self.total, self.warmup = opt, total, warmup
        for g in opt.param_groups:
            g.setdefault("base_lr", g["lr"])

    def set(self, step):
        if step < self.warmup:
            s = step / max(1, self.warmup)
        else:
            s = 0.5 * (1 + math.cos(math.pi * (step - self.warmup) / max(1, self.total - self.warmup)))
        for g in self.opt.param_groups:
            g["lr"] = g["base_lr"] * s


def train_one_epoch(ep, model, ema, loss_fn, opt, sched, scaler, loader, args, dtype, step):
    model.train(); t0 = time.time(); run = 0.0
    for it, (x, y, v, _) in enumerate(loader):
        if args.max_iters and it >= args.max_iters:
            break
        sched.set(step)
        x, y, v = x.cuda(non_blocking=True), y.cuda(non_blocking=True), v.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=dtype):
            logits = model(x)
        loss = loss_fn(logits.float(), y, v)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        ema.update(model, step=step); step += 1
        run += loss.item()
        if it % 200 == 0:
            print(f"ep {ep} it {it}/{len(loader)} loss {run / (it + 1):.4f} lr {opt.param_groups[-1]['lr']:.2e} "
                  f"{(it + 1) * args.bs / (time.time() - t0):.0f} img/s", flush=True)
    return step


def evaluate(ep, model, val_ds, val_dl, dtype, t0):
    logits, idx = predict(model, val_dl, dtype=dtype)
    m, per = challenge_metrics(val_ds.labels[idx], 1 / (1 + np.exp(-logits)), val_ds.ds[idx])
    print(f"== ep {ep} ({time.time() - t0:.0f}s) VAL {fmt(m)}", flush=True)
    for d, pm in per.items():
        print(f"   {d:11s} {fmt(pm)}", flush=True)
    return m, logits, idx


# ---------------------------------------------------------------- main
def main():
    args = get_args()
    if args.list:
        print("losses:", LOSSES.names()); print("heads:", HEADS.names()); return
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out / "args.json", "w"), indent=2)
    torch.backends.cuda.matmul.allow_tf32 = True
    if args.amp == "auto":
        args.amp = "bf16" if torch.cuda.get_device_capability()[0] >= 8 else "fp16"
    dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    print(f"GPU {torch.cuda.get_device_name(0)}, amp {args.amp}, workers {args.workers}, "
          f"head {args.head} {args.head_kw}, loss {args.loss} {args.loss_kw}", flush=True)

    train_dl, val_ds, val_dl, prior = build_data(args)
    model = build_model(model_cfg(args)).cuda()
    if args.grad_ckpt:
        model.backbone.set_grad_checkpointing(True)
    loss_fn = build_loss(args.loss, prior, **args.loss_kw).cuda()
    ema = ModelEmaV3(model, decay=args.ema)
    opt = torch.optim.AdamW(model.param_groups(args.lr, args.wd, args.layer_decay, args.head_lr_mult))
    sched = CosineWithWarmup(opt, args.epochs * len(train_dl), int(args.warmup * len(train_dl)))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp == "fp16")

    best, step, start = -1.0, 0, 0
    if args.resume and (out / "state.pt").exists():
        st = torch.load(out / "state.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); ema.module.load_state_dict(st["ema"])
        opt.load_state_dict(st["opt"]); scaler.load_state_dict(st["scaler"])
        best, step, start = st["best"], st["step"], st["epoch"] + 1
        print(f"resumed from epoch {st['epoch']} (best {best:.4f})", flush=True)

    for ep in range(start, args.epochs):
        t0 = time.time()
        step = train_one_epoch(ep, model, ema, loss_fn, opt, sched, scaler, train_dl, args, dtype, step)
        m, logits, idx = evaluate(ep, ema.module, val_ds, val_dl, dtype, t0)
        ck = {"model": ema.module.state_dict(), "model_cfg": model_cfg(args), "args": vars(args), "epoch": ep, "val": m}
        torch.save(ck, out / "last.pt")
        if m["Challenge_Avg"] > best:
            best = m["Challenge_Avg"]
            torch.save(ck, out / "best.pt")
            np.savez(out / "val_logits.npz", logits=logits, idx=idx)
        if args.resume:  # full state so an interrupted Colab session can continue
            torch.save({"model": model.state_dict(), "ema": ema.module.state_dict(), "opt": opt.state_dict(),
                        "scaler": scaler.state_dict(), "epoch": ep, "step": step, "best": best}, out / "state.pt.tmp")
            os.replace(out / "state.pt.tmp", out / "state.pt")
    print("best val Challenge_Avg", best)


if __name__ == "__main__":
    main()
