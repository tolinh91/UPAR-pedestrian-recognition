"""Language path for a non-CLIP member: a ridge map from its CLS token to the CLIP-L image embedding, so the CLIP text
anchors (query_text head) can score it. The map is fitted on pretrained features of training images (no labels), then
frozen as the head's `proj` buffer, exactly like CLIP's own projection for the CLIP member.

python hp_text_map.py upar/models/clip_l14_text_new.pt upar/models/clip_l14_text_new_hplus.pt [backbone]
Prints the zero-shot AUC of the anchors through the map next to CLIP's own, per val dataset.
"""
import sys

import numpy as np
import timm
import torch

from make_clip_text import auc
from upar.data import UPARDataset, attr_names, build_transforms, load_split

src, out = sys.argv[1], sys.argv[2]
name = sys.argv[3] if len(sys.argv) > 3 else "vit_huge_plus_patch16_dinov3.lvd1689m"
text = torch.load(src)
P = text["proj"].cuda(); Wt = (text["pos"] - text["neutral"][None]).cuda()
nrm = torch.nn.functional.normalize
hp = timm.create_model(name, pretrained=True, num_classes=0, img_size=(224, 112)).cuda().eval().to(torch.bfloat16)
cl = timm.create_model("vit_large_patch14_clip_224.openai", pretrained=True, num_classes=0,
                       img_size=(224, 112)).cuda().eval().to(torch.bfloat16)


def feats(split, rows):
    dh = UPARDataset(split, build_transforms((224, 112), False, mean=hp.pretrained_cfg["mean"], std=hp.pretrained_cfg["std"]), rows=rows)
    dc = UPARDataset(split, build_transforms((224, 112), False, mean=cl.pretrained_cfg["mean"], std=cl.pretrained_cfg["std"]), rows=rows)
    X, Y = [], []
    for i in range(0, len(rows), 32):
        b = range(i, min(i + 32, len(rows)))
        xh = torch.stack([dh[j][0] for j in b]).cuda().to(torch.bfloat16)
        xc = torch.stack([dc[j][0] for j in b]).cuda().to(torch.bfloat16)
        with torch.no_grad():
            X.append(hp.forward_features(xh)[:, 0].float().cpu())
            Y.append(nrm(cl.forward_features(xc)[:, 0].float() @ P, dim=-1).cpu())
    return torch.cat(X).double(), torch.cat(Y).double()


rng = np.random.default_rng(0)
tr_rows = np.sort(rng.choice(len(load_split("train")), 20000, replace=False))
X, Y = feats("train", tr_rows)
G = X.T @ X
best = None
for lam in (1e-4, 1e-3, 1e-2, 1e-1):  # ridge strength relative to the mean eigenvalue; picked on a split of the train rows
    k = 16000
    Gk = X[:k].T @ X[:k]
    W = torch.linalg.solve(Gk + lam * torch.trace(Gk) / Gk.shape[0] * torch.eye(Gk.shape[0], dtype=Gk.dtype), X[:k].T @ Y[:k])
    cos = nrm(X[k:] @ W, dim=-1).mul(Y[k:]).sum(-1).mean().item()
    print(f"lambda {lam:g}: held-out cosine to the CLIP embedding {cos:.4f}", flush=True)
    if best is None or cos > best[1]:
        best = (lam, cos)
lam = best[0]
W = torch.linalg.solve(G + lam * torch.trace(G) / G.shape[0] * torch.eye(G.shape[0], dtype=G.dtype), X.T @ Y).float()
torch.save({"proj": W, "pos": text["pos"], "neutral": text["neutral"], "prompts": text.get("prompts"), "map_from": name,
            "lambda": lam}, out)

va = load_split("val"); A = attr_names()
for ds in ("PA100k", "PETA", "Market1501"):
    rows = np.where(va.ds.values == ds)[0][:5000]
    Xv, Yv = feats("val", rows)
    y = va[A].values[rows]
    s_map = (nrm(Xv.float() @ W, dim=-1) @ Wt.cpu().T).numpy(); s_clip = (Yv.float() @ Wt.cpu().T).numpy()
    ok = [k for k in range(len(A)) if 0 < y[:, k].sum() < len(y)]
    print(f"{ds:10s} zero-shot AUC through the H+ map {np.mean([auc(y[:, k], s_map[:, k]) for k in ok]):.4f}"
          f" vs CLIP itself {np.mean([auc(y[:, k], s_clip[:, k]) for k in ok]):.4f}"
          f" | cosine {nrm(Xv.float() @ W, dim=-1).mul(Yv.float()).sum(-1).mean():.4f}", flush=True)
