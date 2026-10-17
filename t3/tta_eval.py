"""Test-time augmentation on a held-out dataset: does running each box at several crop paddings / shifts help?

python tta_eval.py ../runs/l_lodo_peta peta
Each view is decoded separately (flip test inside), mapped back to image coordinates, and the views are combined by
a confidence-weighted mean of the coordinates. The instance score is the mean keypoint maximum without the two ear
channels; a second score subtracts the spread of the views (their disagreement, in units of the box size).
"""
import json, sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import pose as P

run, name = Path(sys.argv[1]), sys.argv[2]
args = json.load(open(run / "args.json"))
model = P.ViTPose(args["model"], (P.IN_H, P.IN_W), head="heatmap").cuda().eval()
model.load_state_dict(torch.load(run / "last.pt", map_location="cpu", weights_only=False)["model"])
base = P.PoseDataset([name], train=False)
KEEP = [k for k in range(P.NK) if k not in (3, 4)]


class View(Dataset):
    def __init__(self, padding, shift=(0.0, 0.0)):
        self.padding, self.shift = padding, np.array(shift, np.float32)

    def __len__(self):
        return len(base.items)

    def window(self, i):
        box = base.items[i][2]
        c, s = P.box_to_center_scale(box, padding=self.padding)
        return c + self.shift * s, s

    def __getitem__(self, i):
        img = cv2.cvtColor(cv2.imread(base.items[i][0]), cv2.COLOR_BGR2RGB)
        c, s = self.window(i)
        crop = cv2.warpAffine(img, P.affine(c, s), (P.IN_W, P.IN_H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        return torch.from_numpy(((crop.astype(np.float32) - P.MEAN) / P.STD).transpose(2, 0, 1).copy()), i


@torch.no_grad()
def run_view(v):
    kp, mx = np.zeros((len(v), P.NK, 2), np.float32), np.zeros((len(v), P.NK), np.float32)
    for x, idx in DataLoader(v, 128, num_workers=12):
        x = x.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hm = (model(x).float() + P.flip_heatmaps(model(x.flip(-1)).float())) / 2
        xy, m = P.decode(hm.cpu().numpy(), kernel=P.dark_kernel())
        for j, i in enumerate(idx.tolist()):
            c, s = v.window(i)
            Minv = cv2.invertAffineTransform(P.affine(c, s))
            p = xy[j] * np.array([(P.IN_W - 1) / (P.HM_W - 1), (P.IN_H - 1) / (P.HM_H - 1)])
            kp[i] = np.c_[p, np.ones(P.NK)] @ Minv.T; mx[i] = m[j]
    return kp, mx


def preds(kp, mx, score):
    return [{"image_id": int(it[4]), "category_id": 1, "score": float(sc),
             "keypoints": np.c_[k, m].reshape(-1).round(3).tolist()} for it, k, m, sc in zip(base.items, kp, mx, score)]


VIEWS = {"p1.25": (1.25, (0, 0)), "p1.15": (1.15, (0, 0)), "p1.35": (1.35, (0, 0)), "p1.45": (1.45, (0, 0)),
         "sx+": (1.25, (.03, 0)), "sx-": (1.25, (-.03, 0)), "sy+": (1.25, (0, .03)), "sy-": (1.25, (0, -.03))}
out = {k: run_view(View(*v)) for k, v in VIEWS.items()}
size = np.array([np.sqrt(max(it[2][2] * it[2][3], 1)) for it in base.items], np.float32)


def combine(keys):
    K = np.stack([out[k][0] for k in keys]); M = np.stack([out[k][1] for k in keys])
    w = np.clip(M, 1e-3, None)[..., None]
    kp = (K * w).sum(0) / w.sum(0)
    spread = np.sqrt(((K - kp[None]) ** 2).sum(-1)).mean(0)          # (N, 17) pixels
    return kp, M.mean(0), spread


res = {}
for tag, keys in {"single 1.25": ["p1.25"], "scales 1.15/1.25/1.35": ["p1.15", "p1.25", "p1.35"],
                  "scales 1.15..1.45": ["p1.15", "p1.25", "p1.35", "p1.45"],
                  "shifts": ["p1.25", "sx+", "sx-", "sy+", "sy-"], "all 8": list(VIEWS)}.items():
    kp, mx, spread = combine(keys)
    s0 = mx[:, KEEP].mean(1)
    res[tag] = P.coco_eval(name, preds(kp, mx, s0))["AP"]
    if len(keys) > 1:
        dis = (spread[:, KEEP].mean(1) / size)
        for lam in (2.0, 5.0, 10.0):
            res[f"{tag} | score - {lam}*spread"] = P.coco_eval(name, preds(kp, mx, s0 - lam * dis))["AP"]
    print(f"{tag:40s}", {k: round(v, 4) for k, v in res.items() if k.startswith(tag)}, flush=True)
json.dump(res, open(run / f"tta_{name}.json", "w"), indent=1)
