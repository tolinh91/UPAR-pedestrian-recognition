"""Held-out AP of single checkpoints and of their heatmap average when they use different input sizes.
Heatmaps are resized to the finest grid (bilinear, align_corners=True, which keeps the UDP size-1 convention) before
averaging and DARK decoding. Reports the default score (mean of 17 maxima) and the score without the ear slots.

python ens_eval.py --holdout peta ../runs/l256_peta/last.pt:256x192 ../runs/l384_peta/last.pt:384x288
"""
import argparse
import itertools
import json

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pose

EARS = (3, 4)


@torch.no_grad()
def heatmaps(ck_path, size, ds_items, bs):
    pose.set_size(*size)
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    model = pose.ViTPose(ck["args"]["model"], size).cuda().eval()
    model.load_state_dict(ck["model"])
    ds = pose.PoseDataset([], train=False); ds.items = ds_items
    out = []
    for x, _, _, _, _ in DataLoader(ds, bs, num_workers=8):
        x = x.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = (model(x).float() + pose.flip_heatmaps(model(x.flip(-1)).float())) / 2
        out.append(h.half().cpu())
    del model; torch.cuda.empty_cache()
    return torch.cat(out)


def evaluate(name, items, hm, in_hw, fixed_score=None):
    pose.set_size(*in_hw)
    H, W = hm.shape[-2:]
    preds = {"default": [], "no_ears": []}
    if fixed_score is not None:
        preds["fixed"] = []
    keep = [k for k in range(17) if k not in EARS]
    for s in range(0, len(items), 256):
        xy, mx = pose.decode(hm[s:s + 256].float().numpy(), kernel=11 if H <= 64 else 17 if H <= 96 else 23)
        for j, it in enumerate(items[s:s + 256]):
            c, sc = pose.box_to_center_scale(it[2]); M = pose.affine(c, sc); Minv = cv2.invertAffineTransform(M)
            p = xy[j] * np.array([(in_hw[1] - 1) / (W - 1), (in_hw[0] - 1) / (H - 1)])
            p = np.c_[p, np.ones(17)] @ Minv.T
            kp = np.c_[p, mx[j]].reshape(-1).round(3).tolist()
            preds["default"].append({"image_id": it[4], "category_id": 1, "score": float(mx[j].mean()), "keypoints": kp})
            preds["no_ears"].append({"image_id": it[4], "category_id": 1, "score": float(mx[j][keep].mean()), "keypoints": kp})
            if fixed_score is not None:
                preds["fixed"].append({"image_id": it[4], "category_id": 1, "score": float(fixed_score[s + j]), "keypoints": kp})
    return {k: round(pose.coco_eval(name, v)["AP"], 4) for k, v in preds.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="peta")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--score-models", default="", help="e.g. 0,1: score every combo by the mean maximum of these models' average")
    ap.add_argument("models", nargs="+", help="checkpoint:HxW")
    a = ap.parse_args()
    items = pose.PoseDataset([a.holdout], train=False).items
    hms, sizes = [], []
    for m in a.models:
        path, size = m.split(":"); size = tuple(map(int, size.split("x")))
        hms.append(heatmaps(path, size, items, a.bs)); sizes.append(size)
    fixed = None
    if a.score_models:
        idx = [int(i) for i in a.score_models.split(",")]
        avg = sum(hms[i].float() for i in idx) / len(idx)
        fixed = avg.reshape(len(avg), 17, -1).max(-1).values.mean(1).numpy()
    res = {}
    for r in range(1, len(hms) + 1):
        for combo in itertools.combinations(range(len(hms)), r):
            big = max(combo, key=lambda i: sizes[i][0])
            Hh, Ww = hms[big].shape[-2:]
            avg = sum(F.interpolate(hms[i].float(), size=(Hh, Ww), mode="bilinear", align_corners=True)
                      if hms[i].shape[-2:] != (Hh, Ww) else hms[i].float() for i in combo) / len(combo)
            key = " + ".join(a.models[i].split("/")[-2] for i in combo)
            res[key] = evaluate(a.holdout, items, avg, sizes[big], fixed)
            print(key, res[key], flush=True)
    json.dump(res, open(f"../runs/ens_eval_{a.holdout}.json", "w"), indent=1)
