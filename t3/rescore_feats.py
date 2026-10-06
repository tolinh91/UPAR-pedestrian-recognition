"""Per-instance features + true OKS on a held-out dataset, for learning a better instance score (OKS AP ranks by it).

python rescore_feats.py ../runs/h_lodo_peta peta      -> ../runs/h_lodo_peta/feats_peta.npz
python rescore_feats.py ../runs/l_lodo_peta,../runs/h_lodo_peta peta   -> ../runs/h_lodo_peta/feats_peta_ens.npz
Features come from the same flip-test heatmaps the submission computes, so the rescorer can run inside run.py.
With several models the heatmaps are averaged (as in run.py) and 17 columns are appended: distance (heatmap cells)
between the first two models' decoded keypoints.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from pose import (HM_H, HM_W, IN_H, IN_W, NK, SIGMAS, PoseDataset, ViTPose, affine, box_to_center_scale, decode,
                  flip_heatmaps)


def heat_feats(hm):
    """(B, K, H, W) -> per-keypoint peak, mass/peak ratio and spatial std around the peak (heatmap cells)."""
    B, K, H, W = hm.shape
    h = np.clip(hm, 0, None)
    mx = h.reshape(B, K, -1).max(-1)
    mass = h.reshape(B, K, -1).sum(-1)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    w = h / np.clip(mass[..., None, None], 1e-6, None)
    mx_, my_ = (w * xx).sum((2, 3)), (w * yy).sum((2, 3))
    sd = np.sqrt(np.clip((w * ((xx - mx_[..., None, None]) ** 2 + (yy - my_[..., None, None]) ** 2)).sum((2, 3)), 0, None))
    return mx, mass / np.clip(mx, 1e-6, None), sd


def oks(pred, gt_kp, area):
    """COCO computeOks for one instance (pred, gt (K, 3) image coords)."""
    vis = gt_kp[:, 2] > 0
    if not vis.any():
        return np.nan
    var = (2 * SIGMAS) ** 2
    e = ((pred[:, 0] - gt_kp[:, 0]) ** 2 + (pred[:, 1] - gt_kp[:, 1]) ** 2) / var / (area + np.spacing(1)) / 2
    return float(np.exp(-e)[vis].mean())


@torch.no_grad()
def main(runs, name):
    runs = [Path(r) for r in runs.split(",")]
    import json
    models = []
    for run in runs:
        model = ViTPose(json.load(open(run / "args.json"))["model"])
        model.load_state_dict(torch.load(run / "last.pt", map_location="cpu", weights_only=False)["model"])
        models.append(model.cuda().eval())
    ds = PoseDataset([name], train=False)
    from pose import ANN
    areas = {a["id"]: a["area"] for a in json.load(open(ANN / f"train_{name}.json"))["annotations"]}
    F, Y, AID, IID, KP = [], [], [], [], []
    for x, _, _, idx in DataLoader(ds, 128, num_workers=5):
        x = x.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hs = [(m(x).float(), flip_heatmaps(m(x.flip(-1)).float())) for m in models]
        hs = [(a.cpu().numpy(), b.cpu().numpy()) for a, b in hs]
        h0, h1 = sum(a for a, _ in hs) / len(hs), sum(b for _, b in hs) / len(hs)
        if len(hs) > 1:
            dis = np.linalg.norm(decode((hs[0][0] + hs[0][1]) / 2)[0] - decode((hs[1][0] + hs[1][1]) / 2)[0], axis=-1)
        hm = (h0 + h1) / 2
        xy, mx = decode(hm)
        xy0, _ = decode(h0)
        xy1, _ = decode(h1)
        pk, ratio, sd = heat_feats(hm)
        flipd = np.linalg.norm(xy0 - xy1, axis=-1)  # heatmap cells
        for j, i in enumerate(idx.tolist()):
            path, kp, box, aid, iid, _ = ds.items[i]
            c, s = box_to_center_scale(box)
            Minv = cv2.invertAffineTransform(affine(c, s))
            p = xy[j] * np.array([(IN_W - 1) / (HM_W - 1), (IN_H - 1) / (HM_H - 1)])
            p = np.c_[p, np.ones(NK)] @ Minv.T
            bw, bh = float(box[2]), float(box[3])
            F.append(np.concatenate([mx[j], flipd[j], ratio[j], sd[j],
                                     [np.log(bw * bh + 1), bw / max(bh, 1), mx[j].mean(), flipd[j].mean()]]
                                    + ([dis[j]] if len(hs) > 1 else [])))
            Y.append(oks(p, kp, areas[aid]))
            AID.append(aid); IID.append(iid); KP.append(np.c_[p, mx[j]].reshape(-1))
    run = runs[-1]
    np.savez(run / f"feats_{name}{'_ens' if len(runs) > 1 else ''}.npz", X=np.array(F, np.float32), oks=np.array(Y, np.float32), ann_id=np.array(AID),
             image_id=np.array(IID), keypoints=np.array(KP, np.float32))
    y = np.array(Y)
    print(f"{run.name} {name}: {len(F)} instances, mean OKS {np.nanmean(y):.4f}, "
          f"corr(mean max, OKS) {np.corrcoef(np.array(F)[:, -2][~np.isnan(y)], y[~np.isnan(y)])[0, 1]:.3f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
