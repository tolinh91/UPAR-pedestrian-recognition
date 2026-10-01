"""Which frozen backbone transfers best to an unseen dataset? (choose the backbone with evidence, not guesses)

For each backbone (all cost ~the same on CPU):
  1. extract frozen features once for every train+val image ([pooled ; mean patch tokens])
  2. leave-one-dataset-out: fit a linear probe on the train split of two datasets, score all images of the third
  3. report mean per-attribute AUC (threshold free) and Challenge_Avg (balanced decision, see below)
  4. time CPU inference with 6 threads (the evaluation server has 6 cores, no GPU)

Safety: pauses when CPU >= 103 C, aborts when free RAM < 1.5 GB.

python probe_backbones.py --out ../runs/probe
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import ConcatDataset, DataLoader

from upar.data import UPARDataset, attr_names, invalid_mask, load_split
from upar.metrics import challenge_metrics, fmt

BACKBONES = {
    "dinov3_b16_256x128": ("vit_base_patch16_dinov3.lvd1689m", (256, 128)),
    "dinov3_b16_288x112": ("vit_base_patch16_dinov3.lvd1689m", (288, 112)),
    "siglip2_b16_256x128": ("vit_base_patch16_siglip_256.v2_webli", (256, 128)),
    "pecore_b16_256x128": ("vit_pe_core_base_patch16_224.fb", (256, 128)),
    # round 2: same inference cost
    "eupe_b16_256x128": ("vit_base_patch16_dinov3_qkvb.eupe_lvd1689m", (256, 128)),  # distilled from PE + DINOv3
    "pespatial_b16_256x128": ("vit_pe_spatial_base_patch16_512.fb", (256, 128)),
    "tipsv2_b14_224x112": ("vit_base_patch14_reg1_tipsv2.webli", (224, 112)),  # patch 14: 16x8 = 128 tokens
}
DATASETS = ["Market1501", "PA100k", "PETA"]


# ---------------------------------------------------------------- machine safety
def cpu_temp():
    for h in Path("/sys/class/hwmon").glob("hwmon*"):
        if (h / "name").read_text().strip() == "k10temp":
            return int((h / "temp1_input").read_text()) / 1000
    return 0.0


def ram_free_mb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1024
    return 1e9


def guard():
    if ram_free_mb() < 1500:
        raise RuntimeError(f"free RAM {ram_free_mb():.0f} MB < 1500 MB, aborting to protect the machine")
    t = cpu_temp()
    if t >= 103:
        print(f"  [guard] CPU {t:.0f}C, cooling down", flush=True)
        t0 = time.time()
        while cpu_temp() > 95 and time.time() - t0 < 180:
            time.sleep(5)


# ---------------------------------------------------------------- features
def build(name, size):
    m = timm.create_model(name, pretrained=True, num_classes=0, img_size=size)
    cfg = timm.data.resolve_data_config({}, model=m)
    return m.eval(), cfg["mean"], cfg["std"]


@torch.no_grad()
def features(model, loader):
    n_prefix = getattr(model, "num_prefix_tokens", 1)
    feats, idx = [], []
    for it, (x, _, _, j) in enumerate(loader):
        if it % 50 == 0:
            guard()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tok = model.forward_features(x.cuda(non_blocking=True))
            pooled = model.forward_head(tok, pre_logits=True)
        f = torch.cat([pooled.float(), tok[:, n_prefix:].float().mean(1)], 1)
        feats.append(f.half().cpu()); idx.append(j)
    return torch.cat(feats), torch.cat(idx)


@torch.no_grad()
def cpu_ms_per_img(model, size, threads=6, bs=32, reps=3):
    torch.set_num_threads(threads)
    m = model.float().cpu()
    x = torch.randn(bs, 3, *size)
    m.forward_features(x)  # warm-up
    t = time.time()
    for _ in range(reps):
        m.forward_head(m.forward_features(x), pre_logits=True)
    torch.set_num_threads(16)
    return (time.time() - t) / (reps * bs) * 1000


# ---------------------------------------------------------------- linear probe
def fit_probe(X, Y, V, prior, epochs=300, wd=1e-4):
    """Logistic regression per attribute (masked BCE), full batch on GPU."""
    X, Y, V = X.cuda().float(), Y.cuda(), V.cuda()
    mu, sd = X.mean(0), X.std(0) + 1e-6
    lin = torch.nn.Linear(X.shape[1], Y.shape[1]).cuda()
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-2, weight_decay=wd)
    Xn = (X - mu) / sd
    for _ in range(epochs):
        loss = (F.binary_cross_entropy_with_logits(lin(Xn), Y, reduction="none") * V).sum() / V.sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return lambda Z: lin((Z.cuda().float() - mu) / sd).detach().cpu().numpy()


def mean_auc(y, s, valid):
    aucs = []
    for a in np.where(valid)[0]:
        pos, neg = s[y[:, a] == 1, a], s[y[:, a] == 0, a]
        if len(pos) == 0 or len(neg) == 0:
            continue
        r = np.argsort(np.argsort(np.concatenate([pos, neg])))  # rank-based AUC
        aucs.append((r[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))
    return float(np.mean(aucs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../runs/probe")
    ap.add_argument("--backbones", nargs="+", default=list(BACKBONES))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--tau", type=float, default=0.5, help="balanced decision: logit + tau * log((1-p)/p)")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    A = attr_names()
    tr, va = load_split("train"), load_split("val")
    inv = invalid_mask(tr)
    results = json.load(open(out / "results.json")) if (out / "results.json").exists() else {}
    for key in args.backbones:
        name, size = BACKBONES[key]
        f = out / f"{key}.pt"
        model, mean, std = build(name, size)
        ms = cpu_ms_per_img(model, size)
        if not f.exists():
            tf = T.Compose([T.Resize(size), T.ToTensor(), T.Normalize(mean, std)])
            sets = [UPARDataset(s, tf, inv) for s in ["train", "val"]]
            model = model.cuda()
            t0 = time.time()
            feats = [features(model, DataLoader(d, args.bs, num_workers=args.workers, pin_memory=True, prefetch_factor=2)) for d in sets]
            torch.save({"train": feats[0], "val": feats[1]}, f)
            print(f"{key}: features in {time.time() - t0:.0f}s, dim {feats[0][0].shape[1]}", flush=True)
        del model; torch.cuda.empty_cache()
        feats = torch.load(f)

        # stack train+val, remember split / dataset / labels / validity per row
        X = torch.cat([feats["train"][0], feats["val"][0]])
        rows = [(tr, feats["train"][1].numpy(), "train"), (va, feats["val"][1].numpy(), "val")]
        Y = np.concatenate([df[A].values[i] for df, i, _ in rows]).astype(np.float32)
        DS = np.concatenate([df.ds.values[i] for df, i, _ in rows])
        SPLIT = np.concatenate([[s] * len(i) for _, i, s in rows])
        Vm = np.ones_like(Y)
        for d, m in inv.items():
            Vm[DS == d] = ~m

        res = {"cpu_ms_per_img_6threads": ms}
        for held in DATASETS:
            fit_rows = (DS != held) & (SPLIT == "train")
            prior = (Y[fit_rows] * Vm[fit_rows]).sum(0) / Vm[fit_rows].sum(0).clip(1)
            predict = fit_probe(X[fit_rows], torch.tensor(Y[fit_rows]), torch.tensor(Vm[fit_rows]), prior)
            te = DS == held
            logit = predict(X[te]) + args.tau * np.log((1 - prior.clip(1e-3)) / prior.clip(1e-3))
            m, _ = challenge_metrics(Y[te], 1 / (1 + np.exp(-logit)), DS[te])
            m["mean_AUC"] = mean_auc(Y[te], logit, Y[te].any(0))
            res[held] = m
            print(f"  {key:22s} held-out {held:10s} AUC {m['mean_AUC']:.4f} {fmt(m)}", flush=True)
        res["mean_Avg"] = float(np.mean([res[d]["Challenge_Avg"] for d in DATASETS]))
        res["mean_AUC"] = float(np.mean([res[d]["mean_AUC"] for d in DATASETS]))
        results[key] = res
        json.dump(results, open(out / "results.json", "w"), indent=1)

    print("\n== summary (frozen features, linear probe, leave-one-dataset-out)")
    print(f"{'backbone':22s} {'CPU ms/img':>10s} {'mean AUC':>9s} {'mean Avg':>9s} " + " ".join(f"{d[:8]:>9s}" for d in DATASETS))
    for k, r in results.items():
        print(f"{k:22s} {r['cpu_ms_per_img_6threads']:>10.1f} {r['mean_AUC']:>9.4f} {r['mean_Avg']:>9.4f} "
              + " ".join(f"{r[d]['Challenge_Avg']:>9.4f}" for d in DATASETS))


if __name__ == "__main__":
    main()
