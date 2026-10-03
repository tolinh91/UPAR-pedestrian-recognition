"""UPAR Track 3 (top-down 2D pose, 17 PoseTrack18 slots): data, ViTPose(+) model, training and COCO-OKS evaluation.

Train on some UPAR datasets, evaluate on the held-out one (the hidden test comes from unseen domains):
    python pose.py --model usyd-community/vitpose-plus-base --holdout peta --epochs 10 --out runs/vpb_peta
Zero-shot check of a checkpoint (no training):
    python pose.py --model usyd-community/vitpose-plus-base --holdout peta --epochs 0 --out runs/vpb_zs
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset

DATA = Path(__file__).resolve().parent.parent / "data"
ANN = Path(__file__).resolve().parent.parent / "starter/data/annotations/specialization"
DATASETS = ("market", "pa100k", "peta")
NK = 17
FLIP_PAIRS = [(3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
# COCO sigmas with the eye values for head_bottom / head_top (PoseTrack18 convention in mmpose)
SIGMAS = np.array([.026, .025, .025, .035, .035, .079, .079, .072, .072, .062, .062, .107, .107, .087, .087, .089, .089])
IN_H, IN_W, HM_H, HM_W = 256, 192, 64, 48
PADDING = 1.25
MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255


def load_coco(name):
    d = json.load(open(ANN / f"train_{name}.json"))
    ims = {im["id"]: im for im in d["images"]}
    anns = [a for a in d["annotations"] if a.get("num_keypoints", 0) > 0 and a["image_id"] in ims]
    return d, ims, anns


def box_to_center_scale(box, aspect=IN_W / IN_H, padding=PADDING):
    x, y, w, h = [float(v) for v in box[:4]]
    c = np.array([x + w / 2, y + h / 2], np.float32)
    if w > aspect * h:
        h = w / aspect
    else:
        w = h * aspect
    return c, np.array([w, h], np.float32) * padding  # size of the crop window in pixels


def affine(c, s, rot=0.0, out=(IN_W, IN_H)):
    """Matrix mapping image pixels to the output window (UDP-style: pixel centres, size - 1)."""
    ow, oh = out
    sx, sy = (ow - 1) / s[0], (oh - 1) / s[1]
    a = math.radians(rot)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], np.float32)
    M = np.zeros((2, 3), np.float32)
    M[:, :2] = R * np.array([[sx], [sy]], np.float32)
    M[:, 2] = np.array([(ow - 1) / 2, (oh - 1) / 2], np.float32) - M[:, :2] @ c
    return M


class PoseDataset(Dataset):
    def __init__(self, names, train, sigma=2.0):
        self.items, self.train, self.sigma = [], train, sigma
        for n in names:
            _, ims, anns = load_coco(n)
            for a in anns:
                im = ims[a["image_id"]]
                self.items.append((str(DATA / im["file_name"]), np.array(a["keypoints"], np.float32).reshape(NK, 3),
                                   a["bbox"], a["id"], a["image_id"], n))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, kp, box, aid, iid, name = self.items[i]
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        c, s = box_to_center_scale(box)
        kp = kp.copy()
        rot = 0.0
        if self.train:
            s = s * np.clip(np.random.randn() * 0.25 + 1, 0.75, 1.25)
            c = c + np.random.uniform(-0.08, 0.08, 2) * s
            rot = float(np.clip(np.random.randn() * 30, -40, 40)) if np.random.rand() < 0.4 else 0.0
            if np.random.rand() < 0.5:  # horizontal flip of image and keypoint slots
                img = img[:, ::-1].copy()
                c = np.array([img.shape[1] - 1 - c[0], c[1]], np.float32)
                kp[:, 0] = img.shape[1] - 1 - kp[:, 0]
                for a, b in FLIP_PAIRS:
                    kp[[a, b]] = kp[[b, a]]
        M = affine(c, s, rot)
        crop = cv2.warpAffine(img, M, (IN_W, IN_H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        if self.train and np.random.rand() < 0.8:  # photometric jitter
            crop = np.clip(crop.astype(np.float32) * np.random.uniform(0.7, 1.3) + np.random.uniform(-25, 25), 0, 255)
        x = torch.from_numpy(((crop.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1).copy())
        # targets on the heatmap grid
        pts = np.c_[kp[:, :2], np.ones(NK)] @ M.T * np.array([(HM_W - 1) / (IN_W - 1), (HM_H - 1) / (IN_H - 1)])
        vis = (kp[:, 2] > 0).astype(np.float32)
        vis *= (pts[:, 0] >= 0) & (pts[:, 0] <= HM_W - 1) & (pts[:, 1] >= 0) & (pts[:, 1] <= HM_H - 1)
        yy, xx = np.mgrid[0:HM_H, 0:HM_W].astype(np.float32)
        hm = np.exp(-((xx[None] - pts[:, 0, None, None]) ** 2 + (yy[None] - pts[:, 1, None, None]) ** 2) / (2 * self.sigma ** 2))
        hm *= vis[:, None, None]
        return x, torch.from_numpy(hm.astype(np.float32)), torch.from_numpy(vis), torch.tensor(i)


class ViTPose(torch.nn.Module):
    """Hugging Face ViTPose / ViTPose+ (COCO expert of the MoE head) returning (B, 17, 64, 48) heatmaps."""

    def __init__(self, name):
        super().__init__()
        from transformers import VitPoseForPoseEstimation
        self.m = VitPoseForPoseEstimation.from_pretrained(name)
        self.plus = "plus" in name

    def forward(self, x):
        kw = {"dataset_index": torch.zeros(x.shape[0], dtype=torch.long, device=x.device)} if self.plus else {}
        return self.m(pixel_values=x, **kw).heatmaps


def flip_heatmaps(hm):
    hm = hm.flip(-1)
    idx = list(range(NK))
    for a, b in FLIP_PAIRS:
        idx[a], idx[b] = b, a
    return hm[:, idx]


def decode(hm, kernel=11):
    """Argmax + DARK refinement (Zhang et al. 2020) on (B, K, H, W) numpy heatmaps -> coords on the heatmap grid, max."""
    B, K, H, W = hm.shape
    flat = hm.reshape(B, K, -1)
    idx, mx = flat.argmax(-1), flat.max(-1)
    xy = np.stack([idx % W, idx // W], -1).astype(np.float32)
    hb = np.stack([np.stack([cv2.GaussianBlur(h, (kernel, kernel), 0) for h in b]) for b in hm])
    hb = np.log(np.clip(hb, 1e-3, 50))
    for b in range(B):
        for k in range(K):
            x, y = int(xy[b, k, 0]), int(xy[b, k, 1])
            if 1 < x < W - 2 and 1 < y < H - 2:
                h = hb[b, k]
                dx, dy = 0.5 * (h[y, x + 1] - h[y, x - 1]), 0.5 * (h[y + 1, x] - h[y - 1, x])
                dxx = 0.25 * (h[y, x + 2] - 2 * h[y, x] + h[y, x - 2])
                dyy = 0.25 * (h[y + 2, x] - 2 * h[y, x] + h[y - 2, x])
                dxy = 0.25 * (h[y + 1, x + 1] - h[y - 1, x + 1] - h[y + 1, x - 1] + h[y - 1, x - 1])
                det = dxx * dyy - dxy ** 2
                if det != 0:
                    xy[b, k] -= np.array([dyy * dx - dxy * dy, dxx * dy - dxy * dx]) / det
    return xy, mx


@torch.no_grad()
def predict(model, ds, bs=128, flip=True):
    model.eval()
    dl = DataLoader(ds, bs, num_workers=8)
    out = []
    for x, _, _, idx in dl:
        x = x.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hm = model(x).float()
            if flip:
                hm = (hm + flip_heatmaps(model(x.flip(-1)).float())) / 2
        xy, mx = decode(hm.cpu().numpy())
        for j, i in enumerate(idx.tolist()):
            path, kp, box, aid, iid, name = ds.items[i]
            c, s = box_to_center_scale(box)
            M = affine(c, s)
            Minv = cv2.invertAffineTransform(M)
            p = xy[j] * np.array([(IN_W - 1) / (HM_W - 1), (IN_H - 1) / (HM_H - 1)])
            p = np.c_[p, np.ones(NK)] @ Minv.T
            out.append({"image_id": iid, "category_id": 1, "score": float(mx[j].mean()),
                        "keypoints": np.c_[p, mx[j]].reshape(-1).round(3).tolist()})
    return out


def coco_eval(name, preds):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(ANN / f"train_{name}.json"))
        dt = gt.loadRes(preds)
        ev = COCOeval(gt, dt, "keypoints")
        ev.params.kpt_oks_sigmas = SIGMAS
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return {"AP": ev.stats[0], "AP50": ev.stats[1], "AP75": ev.stats[2], "AR": ev.stats[5]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="usyd-community/vitpose-plus-base")
    ap.add_argument("--holdout", default="peta", help="dataset left out of training and used for evaluation; 'none' = all")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--layer-decay", type=float, default=0.75)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-eval", type=int, default=0, help="evaluate on at most N held-out boxes (0 = all)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out / "args.json", "w"), indent=1)
    torch.backends.cuda.matmul.allow_tf32 = True

    model = ViTPose(args.model).cuda()
    train_names = [d for d in DATASETS if d != args.holdout]
    eval_names = [args.holdout] if args.holdout != "none" else []

    def evaluate(tag):
        res = {}
        for n in eval_names:
            ds = PoseDataset([n], train=False)
            if args.max_eval:
                ds.items = ds.items[:args.max_eval]
            preds = predict(model, ds)
            res[n] = coco_eval(n, preds) if not args.max_eval else coco_eval_subset(n, preds, {it[4] for it in ds.items})
        print(f"EVAL {tag} " + " ".join(f"{n} AP {r['AP']:.4f} AP50 {r['AP50']:.4f}" for n, r in res.items()), flush=True)
        return res

    hist = {"zero_shot": evaluate("zero-shot")} if eval_names else {}
    if args.epochs:
        tr = PoseDataset(train_names, train=True)
        dl = DataLoader(tr, args.bs, shuffle=True, num_workers=args.workers, drop_last=True, persistent_workers=True,
                        prefetch_factor=4, pin_memory=True)
        # layer-wise lr decay over the ViT blocks, head at full lr
        blocks = [n for n, _ in model.named_parameters() if ".layer." in n]
        n_layers = 1 + max(int(n.split(".layer.")[1].split(".")[0]) for n in blocks)
        groups = {}
        for n, p in model.named_parameters():
            if ".layer." in n:
                depth = int(n.split(".layer.")[1].split(".")[0]) + 1
            elif "backbone" in n or "embeddings" in n:
                depth = 0
            else:
                depth = n_layers + 1
            scale = args.layer_decay ** (n_layers + 1 - depth)
            wd = 0.0 if p.ndim == 1 or "position" in n else 0.1
            groups.setdefault((scale, wd), []).append(p)
        opt = torch.optim.AdamW([{"params": ps, "lr": args.lr * s, "base": args.lr * s, "weight_decay": wd}
                                 for (s, wd), ps in groups.items()], betas=(0.9, 0.999))
        total, warm, step, t0 = args.epochs * len(dl), 300, 0, time.time()
        for ep in range(args.epochs):
            model.train()
            for x, hm, vis, _ in dl:
                lr_f = min(1, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * step / total))
                for g in opt.param_groups:
                    g["lr"] = g["base"] * lr_f
                x, hm, vis = x.cuda(non_blocking=True), hm.cuda(non_blocking=True), vis.cuda(non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = model(x).float()
                loss = (((pred - hm) ** 2).mean((2, 3)) * vis).sum() / vis.sum().clamp(min=1) * 1000
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if step % 200 == 0:
                    print(f"ep {ep} it {step % len(dl)}/{len(dl)} loss {loss.item():.4f} lr {lr_f * args.lr:.2e} "
                          f"{(step + 1) * args.bs / (time.time() - t0):.0f} img/s", flush=True)
                step += 1
            if eval_names:
                hist[f"ep{ep}"] = evaluate(f"ep {ep}")
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": ep}, out / "last.pt")
    json.dump(hist, open(out / "eval.json", "w"), indent=1)
    print("done", flush=True)


def coco_eval_subset(name, preds, image_ids):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(ANN / f"train_{name}.json"))
        ev = COCOeval(gt, gt.loadRes(preds), "keypoints")
        ev.params.kpt_oks_sigmas = SIGMAS
        ev.params.imgIds = sorted(image_ids)
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return {"AP": ev.stats[0], "AP50": ev.stats[1], "AP75": ev.stats[2], "AR": ev.stats[5]}


if __name__ == "__main__":
    main()
