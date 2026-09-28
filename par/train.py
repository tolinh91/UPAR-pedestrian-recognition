"""Fine-tune DINOv3 for UPAR pedestrian attribute recognition.

python par/train.py --out runs/dinov3_b16 --epochs 10
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from timm.utils import ModelEmaV3
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from data import UPARDataset, attr_names, build_transforms, invalid_mask, load_split
from metrics import challenge_metrics, fmt
from model import PARModel


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="vit_base_patch16_dinov3.lvd1689m")
    ap.add_argument("--size", type=int, nargs=2, default=[256, 128])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-5, help="lr of the last backbone block; head gets 10x")
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--layer-decay", type=float, default=0.75)
    ap.add_argument("--warmup", type=float, default=0.5, help="warmup epochs")
    ap.add_argument("--label-smooth", type=float, default=0.05)
    ap.add_argument("--ds-alpha", type=float, default=0.5, help="dataset sampling ~ N^alpha (0 = equal)")
    ap.add_argument("--ema", type=float, default=0.9995)
    ap.add_argument("--train-splits", nargs="+", default=["train"])
    ap.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 2))
    ap.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16"],
                    help="auto: bf16 on Ampere+ (A100, L4), fp16 + GradScaler on older GPUs (T4, V100)")
    ap.add_argument("--max-iters", type=int, default=0, help="smoke test: stop each epoch after N iterations")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/state.pt if it exists")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--holdout", default=None, help="dataset left out of training; val is scored on it only")
    ap.add_argument("--out", default="runs/dinov3_b16")
    return ap.parse_args()


def pos_weights(labels, valid):
    """DeepMAR weighting: positives of rare attributes weigh more."""
    p = (labels * valid).sum(0) / valid.sum(0).clip(1)
    return torch.tensor(np.exp(1 - p), dtype=torch.float32), torch.tensor(np.exp(p), dtype=torch.float32)


@torch.no_grad()
def predict(model, loader, device, flip=True, dtype=torch.bfloat16):
    model.eval()
    probs, idx = [], []
    for x, _, _, j in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=dtype):
            logit = model(x).float()
            if flip:
                logit = (logit + model(torch.flip(x, dims=[3])).float()) / 2
        probs.append(logit.cpu()); idx.append(j)
    return torch.cat(probs).numpy(), torch.cat(idx).numpy()


def main():
    args = get_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out / "args.json", "w"), indent=2)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda"
    if args.amp == "auto":
        args.amp = "bf16" if torch.cuda.get_device_capability()[0] >= 8 else "fp16"
    dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp == "fp16")
    print(f"GPU {torch.cuda.get_device_name(0)}, amp {args.amp}, workers {args.workers}", flush=True)

    inv = invalid_mask(load_split("train"))
    size = tuple(args.size)
    def rows(split, keep):
        ds = load_split(split).ds.values
        return None if args.holdout is None else np.where((ds != args.holdout) == keep)[0]

    train_sets = [UPARDataset(s, build_transforms(size, True), inv, rows=rows(s, True)) for s in args.train_splits]
    train_ds = ConcatDataset(train_sets)
    weights = np.concatenate([d.sample_weights(args.ds_alpha) for d in train_sets])
    n_per_epoch = sum(len(d) for d in train_sets)
    sampler = WeightedRandomSampler(weights, n_per_epoch, replacement=True)
    train_dl = DataLoader(train_ds, args.bs, sampler=sampler, num_workers=args.workers, pin_memory=True,
                          drop_last=True, persistent_workers=True, prefetch_factor=4)
    val_rows = rows("val", False)
    if args.max_iters:
        val_rows = (np.arange(len(load_split("val"))) if val_rows is None else val_rows)[::50]
    val_ds = UPARDataset("val", build_transforms(size, False), rows=val_rows)
    val_dl = DataLoader(val_ds, 128, num_workers=args.workers, pin_memory=True)

    labels = np.concatenate([d.labels[d.idx] for d in train_sets])
    valid = np.concatenate([d.valid[d.idx] for d in train_sets])
    w_pos, w_neg = [w.to(device) for w in pos_weights(labels, valid)]

    model = PARModel(args.backbone, img_size=size).to(device)
    if args.grad_ckpt:
        model.backbone.set_grad_checkpointing(True)
    ema = ModelEmaV3(model, decay=args.ema)
    opt = torch.optim.AdamW(model.param_groups(args.lr, args.wd, args.layer_decay), betas=(0.9, 0.999))
    for g in opt.param_groups:
        g["base_lr"] = g["lr"]
    steps = args.epochs * len(train_dl); warm = int(args.warmup * len(train_dl))

    best, step, start = -1, 0, 0
    if args.resume and (out / "state.pt").exists():
        st = torch.load(out / "state.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); ema.module.load_state_dict(st["ema"])
        opt.load_state_dict(st["opt"]); scaler.load_state_dict(st["scaler"])
        best, step, start = st["best"], st["step"], st["epoch"] + 1
        print(f"resumed from epoch {st['epoch']} (best {best:.4f})", flush=True)
    for ep in range(start, args.epochs):
        model.train(); t0 = time.time(); run_loss = 0
        for it, (x, y, v, _) in enumerate(train_dl):
            if args.max_iters and it >= args.max_iters:
                break
            lr_scale = step / max(1, warm) if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / (steps - warm)))
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * lr_scale
            x, y, v = x.to(device, non_blocking=True), y.to(device), v.to(device)
            with torch.autocast("cuda", dtype=dtype):
                logit = model(x).float()
            ys = y * (1 - args.label_smooth) + args.label_smooth / 2
            w = torch.where(y > 0.5, w_pos, w_neg) * v
            loss = (F.binary_cross_entropy_with_logits(logit, ys, reduction="none") * w).sum() / v.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            ema.update(model, step=step); step += 1
            run_loss += loss.item()
            if it % 200 == 0:
                print(f"ep {ep} it {it}/{len(train_dl)} loss {run_loss / (it + 1):.4f} "
                      f"lr {opt.param_groups[-1]['lr']:.2e} {(it + 1) * args.bs / (time.time() - t0):.0f} img/s", flush=True)

        logits, idx = predict(ema.module, val_dl, device, dtype=dtype)
        prob = 1 / (1 + np.exp(-logits))
        m, per = challenge_metrics(val_ds.labels[idx], prob, val_ds.ds[idx])
        print(f"== ep {ep} ({time.time() - t0:.0f}s) VAL {fmt(m)}", flush=True)
        for d, pm in per.items():
            print(f"   {d:11s} {fmt(pm)}", flush=True)
        state = {"model": ema.module.state_dict(), "args": vars(args), "epoch": ep, "val": m}
        torch.save(state, out / "last.pt")
        if m["Challenge_Avg"] > best:
            best = m["Challenge_Avg"]
            torch.save(state, out / "best.pt")
            np.savez(out / "val_logits.npz", logits=logits, idx=idx)
        if args.resume:  # full training state so an interrupted Colab session can continue
            torch.save({"model": model.state_dict(), "ema": ema.module.state_dict(), "opt": opt.state_dict(),
                        "scaler": scaler.state_dict(), "epoch": ep, "step": step, "best": best}, out / "state.pt.tmp")
            os.replace(out / "state.pt.tmp", out / "state.pt")
    print("best val Challenge_Avg", best)


if __name__ == "__main__":
    main()
