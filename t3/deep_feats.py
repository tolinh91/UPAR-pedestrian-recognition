"""Per-instance rescoring features of a held-out dataset, with the backbone's own tokens added (CCNet-style idea,
computed outside the network so the packaged model stays unchanged).

python deep_feats.py ../runs/l_lodo_peta peta  ->  ../runs/l_lodo_peta/deep_peta.npz
X:     the 72 heatmap columns of rescore_feats.py (maxima, flip disagreement, mass/peak, spread, box, means)
G:     mean of the last backbone tokens over the 16x12 grid (flip-averaged), C values
K:     mean of the tokens at the 17 predicted keypoints (nearest grid cell), C values
oks, ann_id, image_id, keypoints as in rescore_feats.py
"""
import json, sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from pose import (HM_H, HM_W, IN_H, IN_W, NK, ANN, PoseDataset, ViTPose, affine, box_to_center_scale, decode,
                  flip_heatmaps)
from rescore_feats import heat_feats, oks

run, name = Path(sys.argv[1]), sys.argv[2]
model = ViTPose(json.load(open(run / "args.json"))["model"])
model.load_state_dict(torch.load(run / "last.pt", map_location="cpu", weights_only=False)["model"])
model = model.cuda().eval()
ds = PoseDataset([name], train=False)
areas = {a["id"]: a["area"] for a in json.load(open(ANN / f"train_{name}.json"))["annotations"]}
kw = lambda x: {"dataset_index": torch.zeros(x.shape[0], dtype=torch.long, device=x.device)} if model.plus else {}


@torch.no_grad()
def tokens(x):
    f = model.m.backbone.forward_with_filtered_kwargs(x, **kw(x)).feature_maps[-1]      # (B, 192, C)
    return f.float()


Fx, G, K, Y, AID, IID, KP = [], [], [], [], [], [], []
with torch.no_grad():
    for x, _, _, idx, _ in DataLoader(ds, 128, num_workers=12):
        x = x.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h0 = model(x).float(); h1 = flip_heatmaps(model(x.flip(-1)).float())
            t0 = tokens(x); t1 = tokens(x.flip(-1))
        B, N, C = t0.shape
        gh, gw = IN_H // 16, IN_W // 16
        t1 = t1.reshape(B, gh, gw, C).flip(2).reshape(B, N, C)          # flipped view back onto the same grid
        tok = (t0 + t1) / 2
        h0, h1 = h0.cpu().numpy(), h1.cpu().numpy(); hm = (h0 + h1) / 2
        xy, mx = decode(hm); xy0, _ = decode(h0); xy1, _ = decode(h1)
        pk, ratio, sd = heat_feats(hm)
        flipd = np.linalg.norm(xy0 - xy1, axis=-1)
        cell = np.clip((xy[..., 1] * 4 / 16).astype(int), 0, gh - 1) * gw + np.clip((xy[..., 0] * 4 / 16).astype(int), 0, gw - 1)
        tk = torch.gather(tok, 1, torch.from_numpy(cell).cuda().long()[..., None].expand(-1, -1, C)).mean(1)
        G.append(tok.mean(1).cpu().numpy().astype(np.float16)); K.append(tk.cpu().numpy().astype(np.float16))
        for j, i in enumerate(idx.tolist()):
            path, kp, box, aid, iid, _ = ds.items[i]
            c, s = box_to_center_scale(box)
            Minv = cv2.invertAffineTransform(affine(c, s))
            p = xy[j] * np.array([(IN_W - 1) / (HM_W - 1), (IN_H - 1) / (HM_H - 1)])
            p = np.c_[p, np.ones(NK)] @ Minv.T
            bw, bh = float(box[2]), float(box[3])
            Fx.append(np.concatenate([mx[j], flipd[j], ratio[j], sd[j],
                                      [np.log(bw * bh + 1), bw / max(bh, 1), mx[j].mean(), flipd[j].mean()]]))
            Y.append(oks(p, kp, areas[aid])); AID.append(aid); IID.append(iid); KP.append(np.c_[p, mx[j]].reshape(-1))
np.savez(run / f"deep_{name}.npz", X=np.array(Fx, np.float32), G=np.concatenate(G), K=np.concatenate(K),
         oks=np.array(Y, np.float32), ann_id=np.array(AID), image_id=np.array(IID), keypoints=np.array(KP, np.float32))
print(run.name, name, len(Y), "instances, mean OKS", float(np.nanmean(Y)), flush=True)
