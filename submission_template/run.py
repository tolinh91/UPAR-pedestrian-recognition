"""UPAR 2027 Track 3 submission: ViTPose(+) fine-tuned on the UPAR pose data (17 PoseTrack18 slots).

The ingestion program calls predict_image(sample) once per image. Preprocessing, flip test and DARK decoding mirror
t3/pose.py (training and local evaluation) exactly. The evaluation image ships torch, numpy and Pillow only.
- GPU present: assets/model_gpu.torchscript (the large model). CPU only: assets/model_cpu.torchscript (small model),
  so the run stays inside the wall-clock limit either way.
- A time budget: past it, flip test is dropped; never fails the run on one bad image.
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(HERE, "assets", "config.json")))
IN_H, IN_W, HM_H, HM_W = CFG["in_h"], CFG["in_w"], CFG["hm_h"], CFG["hm_w"]
PADDING = CFG["padding"]
MEAN = np.array(CFG["mean"], np.float32) * 255
STD = np.array(CFG["std"], np.float32) * 255
FLIP_PAIRS = CFG["flip_pairs"]
NK = 17
BUDGET_S = float(os.environ.get("UPAR_BUDGET_S", CFG.get("budget_s", 6600)))
T_IMPORT = time.time()

_MODEL = None
_DEVICE = "cpu"
_FLIP = True
_S = {"n": 0}


def log(msg):
    print(f"[run.py] {msg}", flush=True)


def load_model():
    global _MODEL, _DEVICE, _FLIP
    if _MODEL is not None:
        return
    torch.set_grad_enabled(False)
    try:
        if torch.cuda.is_available():
            torch.zeros(1).cuda()
            _DEVICE = "cuda"
    except Exception as e:  # any CUDA problem: CPU
        log(f"cuda unusable: {e!r}")
    name = "model_gpu.torchscript" if _DEVICE == "cuda" else "model_cpu.torchscript"
    path = os.path.join(HERE, "assets", name)
    if not os.path.exists(path):
        path = os.path.join(HERE, "assets", "model_gpu.torchscript")
    if _DEVICE == "cpu":
        torch.set_num_threads(max(1, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 4))
        _FLIP = CFG.get("cpu_flip", False)
    _MODEL = torch.jit.load(path, map_location=_DEVICE).eval()
    log(f"device={_DEVICE} model={os.path.basename(path)} flip={_FLIP} budget={BUDGET_S:.0f}s")


def box_to_center_scale(box, aspect=IN_W / IN_H, padding=PADDING):
    x, y, w, h = [float(v) for v in box[:4]]
    c = np.array([x + w / 2, y + h / 2], np.float32)
    if w > aspect * h:
        h = w / aspect
    else:
        w = h * aspect
    return c, np.array([w, h], np.float32) * padding


def affine(c, s, out=(IN_W, IN_H)):
    ow, oh = out
    M = np.zeros((2, 3), np.float32)
    M[0, 0], M[1, 1] = (ow - 1) / s[0], (oh - 1) / s[1]
    M[:, 2] = np.array([(ow - 1) / 2, (oh - 1) / 2], np.float32) - M[:, :2] @ c
    return M


def warp_affine(image, M, out_h, out_w):
    """Bilinear warp with zero padding, same pixel convention as cv2.warpAffine (used in training)."""
    inv = np.linalg.inv(np.vstack([M, [0, 0, 1]]))[:2].astype(np.float32)
    ys, xs = np.meshgrid(np.arange(out_h, dtype=np.float32), np.arange(out_w, dtype=np.float32), indexing="ij")
    sx = inv[0, 0] * xs + inv[0, 1] * ys + inv[0, 2]
    sy = inv[1, 0] * xs + inv[1, 1] * ys + inv[1, 2]
    h, w = image.shape[:2]
    x0, y0 = np.floor(sx).astype(np.int64), np.floor(sy).astype(np.int64)
    wx, wy = (sx - x0)[..., None], (sy - y0)[..., None]
    # zero outside the image, per corner (as cv2 BORDER_CONSTANT)
    def px(yy, xx):
        ok = ((xx >= 0) & (xx < w) & (yy >= 0) & (yy < h))[..., None]
        return image[np.clip(yy, 0, h - 1), np.clip(xx, 0, w - 1)] * ok
    top = px(y0, x0) * (1 - wx) + px(y0, x0 + 1) * wx
    bot = px(y0 + 1, x0) * (1 - wx) + px(y0 + 1, x0 + 1) * wx
    return top * (1 - wy) + bot * wy


def _gauss_kernel(k=11):
    sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8  # cv2.GaussianBlur default for ksize k
    x = torch.arange(k, dtype=torch.float32) - (k - 1) / 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    return g / g.sum()


def decode(hm):
    """Argmax + DARK on (B, K, H, W) torch heatmaps -> (B, K, 2) coords on the heatmap grid, (B, K) maxima."""
    B, K, H, W = hm.shape
    flat = hm.reshape(B, K, -1)
    mx, idx = flat.max(-1)
    xy = torch.stack([idx % W, idx // W], -1).float()
    g = _gauss_kernel().to(hm)
    r = g.numel() // 2
    hb = F.pad(hm.reshape(B * K, 1, H, W), (r, r, r, r), mode="reflect")
    hb = F.conv2d(F.conv2d(hb, g.view(1, 1, 1, -1)), g.view(1, 1, -1, 1)).reshape(B, K, H, W)
    hb = hb.clamp(1e-3, 50).log()
    x, y = xy[..., 0].long(), xy[..., 1].long()
    ok = (x > 1) & (x < W - 2) & (y > 1) & (y < H - 2)
    xc, yc = x.clamp(2, W - 3), y.clamp(2, H - 3)
    hf = hb.reshape(B, K, -1)
    at = lambda dy, dx: hf.gather(2, ((yc + dy) * W + xc + dx).unsqueeze(-1)).squeeze(-1)
    c = at(0, 0)
    dx, dy = 0.5 * (at(0, 1) - at(0, -1)), 0.5 * (at(1, 0) - at(-1, 0))
    dxx, dyy = 0.25 * (at(0, 2) - 2 * c + at(0, -2)), 0.25 * (at(2, 0) - 2 * c + at(-2, 0))
    dxy = 0.25 * (at(1, 1) - at(-1, 1) - at(1, -1) + at(-1, -1))
    det = dxx * dyy - dxy ** 2
    good = ok & (det != 0)
    det = torch.where(good, det, torch.ones_like(det))
    off = torch.stack([(dyy * dx - dxy * dy) / det, (dxx * dy - dxy * dx) / det], -1)
    xy = xy - torch.where(good.unsqueeze(-1), off, torch.zeros_like(off))
    return xy, mx


def _flip_index():
    idx = list(range(NK))
    for a, b in FLIP_PAIRS:
        idx[a], idx[b] = b, a
    return idx


FLIP_IDX = _flip_index()


def _forward(x):
    if _DEVICE == "cuda" and CFG.get("gpu_half"):
        x = x.half()
    hm = _MODEL(x).float()
    if _FLIP:
        hm = (hm + _MODEL(x.flip(-1)).float().flip(-1)[:, FLIP_IDX]) / 2
    return hm


def _prior(box):
    x, y, w, h = [float(v) for v in box[:4]]
    return [[x + w / 2, y + h / 2, 0.0]] * NK


def predict_image(sample):
    global _FLIP
    load_model()
    boxes = sample.get("boxes") or []
    if not boxes:
        return {"keypoints": [], "scores": []}
    try:
        img = np.asarray(Image.open(sample["image_path"]).convert("RGB"), dtype=np.float32)
    except Exception as e:
        log(f"WARNING cannot read {sample.get('image_path')}: {e!r}")
        return {"keypoints": [_prior(b) for b in boxes], "scores": [0.01] * len(boxes)}
    kps, scores, crops, mats = [None] * len(boxes), [0.01] * len(boxes), [], []
    use = []
    for i, b in enumerate(boxes):
        if float(b[2]) <= 1 or float(b[3]) <= 1:
            kps[i] = _prior(b)
            continue
        c, s = box_to_center_scale(b)
        M = affine(c, s)
        crops.append(((warp_affine(img, M, IN_H, IN_W) - MEAN) / STD).transpose(2, 0, 1))
        mats.append(M)
        use.append(i)
    if use:
        x = torch.from_numpy(np.stack(crops).astype(np.float32)).to(_DEVICE)
        xy, mx = decode(_forward(x))
        xy, mx = xy.cpu().numpy(), mx.cpu().numpy()
        for j, i in enumerate(use):
            p = xy[j] * np.array([(IN_W - 1) / (HM_W - 1), (IN_H - 1) / (HM_H - 1)], np.float32)
            inv = np.linalg.inv(np.vstack([mats[j], [0, 0, 1]]))[:2]
            p = np.c_[p, np.ones(NK)] @ inv.T
            kps[i] = [[float(p[k, 0]), float(p[k, 1]), float(mx[j, k])] for k in range(NK)]
            scores[i] = float(mx[j].mean())
    _S["n"] += 1
    if _FLIP and _S["n"] % 500 == 0 and time.time() - T_IMPORT > BUDGET_S * 0.5:
        _FLIP = False  # half the budget gone: single pass from here (test size unknown in advance)
        log(f"{_S['n']} images after {time.time() - T_IMPORT:.0f}s: flip test disabled")
    return {"keypoints": kps, "scores": scores}
