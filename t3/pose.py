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


SIMCC_K = 2  # SimCC bins per input pixel
SIMCC_SIGMA = (4.9, 5.66)  # label sigma in bins (x, y), RTMPose 256x192


def set_size(h, w, hm_div=4):
    global IN_H, IN_W, HM_H, HM_W
    IN_H, IN_W, HM_H, HM_W = h, w, h // hm_div, w // hm_div


def dark_kernel():
    return 11 if HM_H <= 64 else 17 if HM_H <= 96 else 23


def load_coco(name):
    d = json.load(open(ANN / f"train_{name}.json"))
    ims = {im["id"]: im for im in d["images"]}
    anns = [a for a in d["annotations"] if a.get("num_keypoints", 0) > 0 and a["image_id"] in ims]
    return d, ims, anns


def box_to_center_scale(box, aspect=None, padding=PADDING):
    aspect = IN_W / IN_H if aspect is None else aspect
    x, y, w, h = [float(v) for v in box[:4]]
    c = np.array([x + w / 2, y + h / 2], np.float32)
    if w > aspect * h:
        h = w / aspect
    else:
        w = h * aspect
    return c, np.array([w, h], np.float32) * padding  # size of the crop window in pixels


def affine(c, s, rot=0.0, out=None):
    """Matrix mapping image pixels to the output window (UDP-style: pixel centres, size - 1)."""
    ow, oh = out if out is not None else (IN_W, IN_H)
    sx, sy = (ow - 1) / s[0], (oh - 1) / s[1]
    a = math.radians(rot)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], np.float32)
    M = np.zeros((2, 3), np.float32)
    M[:, :2] = R * np.array([[sx], [sy]], np.float32)
    M[:, 2] = np.array([(ow - 1) / 2, (oh - 1) / 2], np.float32) - M[:, :2] @ c
    return M


class PoseDataset(Dataset):
    def __init__(self, names, train, sigma=None):
        # mmpose: sigma 2 for 64x48 heatmaps, 3 for 96x72
        self.items, self.train, self.sigma = [], train, sigma if sigma is not None else 2.0 * HM_H / 64
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
        pin = (np.c_[kp[:, :2], np.ones(NK)] @ M.T).astype(np.float32)  # keypoints in input-crop pixels
        return x, torch.from_numpy(hm.astype(np.float32)), torch.from_numpy(vis), torch.tensor(i), torch.from_numpy(pin)


def _patch_moe():
    """ViTPose+ evaluates all six expert layers on every token and masks five away; we only use the COCO expert
    (index 0), so compute it alone. Saves activation memory and time; identical output when all indices are 0.
    transformers >= 5 keeps the experts in VitPoseNaiveMoe, 4.x inside VitPoseBackboneMoeMLP."""
    from transformers.models.vitpose_backbone import modeling_vitpose_backbone as mb
    coco = lambda idx: torch.jit.is_tracing() or bool((idx == 0).all())
    if hasattr(mb, "VitPoseNaiveMoe"):
        cls = mb.VitPoseNaiveMoe
        if getattr(cls, "_coco_only", False):
            return
        orig = cls.forward

        def forward(self, hidden_state, indices):
            return self[0](hidden_state) if coco(indices) else orig(self, hidden_state, indices)
    else:
        cls = mb.VitPoseBackboneMoeMLP
        if getattr(cls, "_coco_only", False):
            return
        orig = cls.forward

        def forward(self, hidden_state, indices):
            if not coco(indices):
                return orig(self, hidden_state, indices)
            h = self.act(self.fc1(hidden_state))
            return torch.cat([self.fc2(h), self.experts[0](h)], dim=-1)
    cls.forward = forward
    cls._coco_only = True


class ViTPose(torch.nn.Module):
    """Hugging Face ViTPose / ViTPose+ (COCO expert of the MoE head) returning (B, 17, 64, 48) heatmaps."""

    def __init__(self, name, size=(256, 192), head="heatmap"):
        super().__init__()
        self.head_type = head
        from transformers import VitPoseForPoseEstimation
        self.m = VitPoseForPoseEstimation.from_pretrained(name)
        self.plus = "plus" in name
        if self.plus:
            _patch_moe()
            for n, p in self.m.named_parameters():  # only the COCO expert (index 0) is ever used
                if ".experts." in n and not n.split(".experts.")[1].startswith("0."):
                    p.requires_grad_(False)
        e = self.m.backbone.embeddings; old = tuple(e.patch_embeddings.image_size)
        if tuple(size) != old:  # bicubic interpolation of the position-embedding grid (no class token in use)
            ph, pw = e.patch_embeddings.patch_size if hasattr(e.patch_embeddings, "patch_size") else (16, 16)
            pe = e.position_embeddings.data; C = pe.shape[-1]
            grid = pe[:, 1:].reshape(1, old[0] // ph, old[1] // pw, C).permute(0, 3, 1, 2)
            grid = F.interpolate(grid, size=(size[0] // ph, size[1] // pw), mode="bicubic", align_corners=False)
            e.position_embeddings = torch.nn.Parameter(torch.cat([pe[:, :1], grid.flatten(2).transpose(1, 2)], 1))
            e.patch_embeddings.image_size = tuple(size)
            self.m.config.backbone_config.image_size = tuple(size)
        self.size = tuple(size)
        hd = self.m.head
        if head == "sr":  # one more 2x deconv block before the (pretrained) 1x1 conv: 128x96 heatmaps
            self.deconv3 = torch.nn.Sequential(torch.nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False),
                                               torch.nn.BatchNorm2d(256), torch.nn.ReLU())
        if head in ("simcc", "hm+simcc"):  # SimCC on the flattened per-joint maps of the pretrained decoder
            n = (size[0] // 4) * (size[1] // 4)
            self.sx = torch.nn.Linear(n, size[1] * SIMCC_K)
            self.sy = torch.nn.Linear(n, size[0] * SIMCC_K)

    def forward(self, x):
        kw = {"dataset_index": torch.zeros(x.shape[0], dtype=torch.long, device=x.device)} if self.plus else {}
        if self.head_type == "heatmap":
            return self.m(pixel_values=x, **kw).heatmaps
        f = self.m.backbone.forward_with_filtered_kwargs(x, **kw).feature_maps[-1]
        B = f.shape[0]
        f = f.permute(0, 2, 1).reshape(B, -1, self.size[0] // 16, self.size[1] // 16).contiguous()
        hd = self.m.head
        f = hd.relu1(hd.batchnorm1(hd.deconv1(f)))
        f = hd.relu2(hd.batchnorm2(hd.deconv2(f)))
        if self.head_type == "sr":
            return hd.conv(self.deconv3(f))
        maps = hd.conv(f)  # (B, 17, H/4, W/4)
        flat = maps.flatten(2)
        out = {"sx": self.sx(flat), "sy": self.sy(flat)}
        if self.head_type == "hm+simcc":
            out["hm"] = maps
        return out


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
    for x, _, _, idx, _ in dl:
        x = x.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hm = model(x).float()
            if flip:
                hm = (hm + flip_heatmaps(model(x.flip(-1)).float())) / 2
        xy, mx = decode(hm.cpu().numpy(), kernel=dark_kernel())
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


def simcc_decode(sx, sy, win=3):
    """(B, K, Wk), (B, K, Hk) logits -> (B, K, 2) input-pixel coords (argmax refined by the local expectation of the
    softmax within +-win bins) and (B, K) confidence (mean of the two peak probabilities)."""
    out, conf = [], []
    for lg in (sx, sy):
        p = lg.float().softmax(-1); n = p.shape[-1]
        pk, am = p.max(-1)
        offs = torch.arange(-win, win + 1, device=p.device)
        ix = (am[..., None] + offs).clamp(0, n - 1)
        w = p.gather(-1, ix)
        out.append((w * ix).sum(-1) / w.sum(-1).clamp(min=1e-9) / SIMCC_K); conf.append(pk)
    return torch.stack(out, -1), (conf[0] + conf[1]) / 2


def flip_simcc(o):
    """Outputs for a horizontally flipped crop -> original frame: x-logit vectors are reversed in pixel space."""
    idx = list(range(NK))
    for a, b in FLIP_PAIRS:
        idx[a], idx[b] = b, a
    r = {"sx": o["sx"][:, idx], "sy": o["sy"][:, idx]}
    if "hm" in o:
        r["hm"] = flip_heatmaps(o["hm"])
    return r


@torch.no_grad()
def predict_simcc(model, ds, bs=128):
    """Flip-tested SimCC (and, for hm+simcc, heatmap) predictions. Returns {variant: coco preds}."""
    model.eval()
    res = {"simcc": [], "heatmap": [], "average": []}
    for x, _, _, idx, _ in DataLoader(ds, bs, num_workers=8):
        x = x.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o, of = model(x), flip_simcc(model(x.flip(-1)))
        xy0, c0 = simcc_decode(o["sx"], o["sy"]); xy1, c1 = simcc_decode(of["sx"], of["sy"])
        xy1[..., 0] = (IN_W - 1) - xy1[..., 0]  # flipped crop -> original crop pixels
        xy_s, conf_s = ((xy0 + xy1) / 2).cpu().numpy(), ((c0 + c1) / 2).cpu().numpy()
        variants = {"simcc": (xy_s, conf_s)}
        if "hm" in o:
            hm = ((o["hm"].float() + of["hm"].float()) / 2).cpu().numpy()
            xh, mh = decode(hm, kernel=dark_kernel())
            xh = xh * np.array([(IN_W - 1) / (HM_W - 1), (IN_H - 1) / (HM_H - 1)])
            variants["heatmap"] = (xh, mh); variants["average"] = ((xh + xy_s) / 2, mh)
        for name, (xy, cf) in variants.items():
            for j, i in enumerate(idx.tolist()):
                path, kp, box, aid, iid, _ = ds.items[i]
                c, s = box_to_center_scale(box); Minv = cv2.invertAffineTransform(affine(c, s))
                p = np.c_[xy[j], np.ones(NK)] @ Minv.T
                res[name].append({"image_id": iid, "category_id": 1, "score": float(cf[j].mean()),
                                  "keypoints": np.c_[p, cf[j]].reshape(-1).round(3).tolist()})
    return {k: v for k, v in res.items() if v}


def simcc_loss(o, pin, vis):
    """Cross-entropy against normalised 1D Gaussian labels (equivalent to SimCC's KL loss up to a constant)."""
    tot = 0.0
    for key, coord, sig in (("sx", pin[..., 0], SIMCC_SIGMA[0]), ("sy", pin[..., 1], SIMCC_SIGMA[1])):
        lg = o[key].float(); n = lg.shape[-1]
        b = torch.arange(n, device=lg.device, dtype=torch.float32)
        t = torch.exp(-(b - coord[..., None] * SIMCC_K) ** 2 / (2 * sig ** 2))
        t = t / t.sum(-1, keepdim=True).clamp(min=1e-9)
        tot = tot + (-(t * lg.log_softmax(-1)).sum(-1) * vis).sum()
    return tot


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
    ap.add_argument("--size", default="256x192", help="input HxW; heatmaps are a quarter of it")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--head", default="heatmap", choices=["heatmap", "sr", "simcc", "hm+simcc"])
    ap.add_argument("--new-lr-mult", type=float, default=10.0, help="lr multiplier for newly initialised head layers")
    ap.add_argument("--eval-bs", type=int, default=128)
    ap.add_argument("--no-zero-shot", action="store_true")
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps (effective batch = bs * accum)")
    ap.add_argument("--save-preds", action="store_true", help="write held-out predictions of the last epoch")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    set_size(*map(int, args.size.split("x")), hm_div=2 if args.head == "sr" else 4)
    import os
    ddp = "LOCAL_RANK" in os.environ
    rank = int(os.environ.get("RANK", 0))
    if ddp:
        torch.distributed.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        json.dump(vars(args), open(out / "args.json", "w"), indent=1)
    torch.backends.cuda.matmul.allow_tf32 = True

    model = ViTPose(args.model, (IN_H, IN_W), head=args.head).cuda()
    if args.grad_ckpt:
        model.m.backbone.gradient_checkpointing_enable() if hasattr(model.m.backbone, "gradient_checkpointing_enable") else model.m.gradient_checkpointing_enable()
    train_names = [d for d in DATASETS if d != args.holdout]
    eval_names = [args.holdout] if args.holdout != "none" else []

    def evaluate(tag):
        res = {}
        for n in eval_names:
            ds = PoseDataset([n], train=False)
            if args.max_eval:
                ds.items = ds.items[:args.max_eval]
            if args.head in ("simcc", "hm+simcc"):
                pv = predict_simcc(model, ds, bs=args.eval_bs)
                for k, v in pv.items():
                    r = coco_eval(n, v); res[f"{n}/{k}"] = r
                    keep = [i for i in range(NK) if i not in (3, 4)]
                    v2 = [dict(d, score=float(np.mean(np.array(d["keypoints"]).reshape(NK, 3)[keep, 2]))) for d in v]
                    res[f"{n}/{k}/no_ears"] = coco_eval(n, v2)
                    if args.save_preds:
                        json.dump(v, open(out / f"preds_{n}_{k}_{tag.replace(' ', '')}.json", "w"))
                continue
            preds = predict(model, ds, bs=args.eval_bs)
            if args.save_preds:
                json.dump(preds, open(out / f"preds_{n}_{tag.replace(' ', '')}.json", "w"))
            res[n] = coco_eval(n, preds) if not args.max_eval else coco_eval_subset(n, preds, {it[4] for it in ds.items})
            keep = [i for i in range(NK) if i not in (3, 4)]
            p2 = [dict(d, score=float(np.mean(np.array(d["keypoints"]).reshape(NK, 3)[keep, 2]))) for d in preds]
            res[f"{n}/no_ears"] = coco_eval(n, p2)
        print(f"EVAL {tag} " + " ".join(f"{n} AP {r['AP']:.4f} AP50 {r['AP50']:.4f}" for n, r in res.items()), flush=True)
        return res

    hist = {"zero_shot": evaluate("zero-shot")} if eval_names and not args.no_zero_shot and rank == 0 else {}
    net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()]) if ddp else model
    if args.epochs:
        tr = PoseDataset(train_names, train=True)
        world = torch.distributed.get_world_size() if ddp else 1
        sampler = torch.utils.data.distributed.DistributedSampler(tr, shuffle=True, drop_last=True) if ddp else None
        dl = DataLoader(tr, args.bs // world, shuffle=sampler is None, sampler=sampler, num_workers=args.workers, drop_last=True,
                        persistent_workers=True, prefetch_factor=4, pin_memory=True)
        # layer-wise lr decay over the ViT blocks, head at full lr
        blocks = [n for n, _ in model.named_parameters() if ".layer." in n]
        n_layers = 1 + max(int(n.split(".layer.")[1].split(".")[0]) for n in blocks)
        groups = {}
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if n.startswith(("deconv3.", "sx.", "sy.")):
                groups.setdefault((args.new_lr_mult, 0.0 if p.ndim == 1 else 0.1), []).append(p)
                continue
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
            if sampler is not None:
                sampler.set_epoch(ep)
            for x, hm, vis, _, pin in dl:
                lr_f = min(1, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * step / total))
                for g in opt.param_groups:
                    g["lr"] = g["base"] * lr_f
                x, hm, vis = x.cuda(non_blocking=True), hm.cuda(non_blocking=True), vis.cuda(non_blocking=True)
                pin = pin.cuda(non_blocking=True)
                loss = 0.0
                chunks = list(zip(x.chunk(args.accum), hm.chunk(args.accum), vis.chunk(args.accum), pin.chunk(args.accum)))
                for ci, (xc, hc, vc, pc) in enumerate(chunks):
                    import contextlib
                    sync = net.no_sync() if ddp and ci < len(chunks) - 1 else contextlib.nullcontext()
                    with sync:
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            pred = net(xc)
                        if isinstance(pred, dict):
                            lc = simcc_loss(pred, pc, vc) / vis.sum().clamp(min=1)
                            if "hm" in pred:
                                lc = lc + (((pred["hm"].float() - hc) ** 2).mean((2, 3)) * vc).sum() / vis.sum().clamp(min=1) * 1000
                        else:
                            lc = (((pred.float() - hc) ** 2).mean((2, 3)) * vc).sum() / vis.sum().clamp(min=1) * 1000
                        lc.backward(); loss += lc.detach()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
                if step % 200 == 0 and rank == 0:
                    print(f"ep {ep} it {step % len(dl)}/{len(dl)} loss {loss.item():.4f} lr {lr_f * args.lr:.2e} "
                          f"{(step + 1) * args.bs / (time.time() - t0):.0f} img/s "
                          f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G", flush=True)
                step += 1
            if eval_names and rank == 0:
                hist[f"ep{ep}"] = evaluate(f"ep {ep}")
            if rank == 0:
                torch.save({"model": model.state_dict(), "args": vars(args), "epoch": ep}, out / "last.pt")
    if rank == 0:
        json.dump(hist, open(out / "eval.json", "w"), indent=1)
    print("done", flush=True)
    if ddp:
        torch.distributed.destroy_process_group()


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
