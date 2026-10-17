"""MaskCLIP-style dense CLIP embeddings for region anchors: value embeddings of the last attention block (no q-k mixing,
no residual / MLP) -> final norm -> CLIP projection, pooled over the upper- or lower-body rows. Zero-shot AUC of the
colour attributes against the CLS anchor (prompts from zs_probe.NEW). python zs_probe_mask.py upar/models/clip_l14_text_new.pt"""
import sys

import numpy as np
import timm
import torch

from make_clip_text import auc
from upar.data import UPARDataset, attr_names, build_transforms, load_split
from zs_probe import LOWER, UPPER

d = torch.load(sys.argv[1]); W = (d["pos"] - d["neutral"][None]).cuda(); P = d["proj"].cuda()
tm = timm.create_model("vit_large_patch14_clip_224.openai", pretrained=True, num_classes=0, img_size=(224, 112)).cuda().eval()
tm = tm.to(torch.bfloat16)
cfg = tm.pretrained_cfg
ds = UPARDataset("val", build_transforms((224, 112), False, mean=cfg["mean"], std=cfg["std"]))
va = load_split("val"); A = attr_names(); blk = tm.blocks[-1]


@torch.no_grad()
def feats(x):
    t = tm.norm_pre(tm._pos_embed(tm.patch_embed(x)))
    for b in tm.blocks[:-1]:
        t = b(t)
    B, N, C = t.shape
    qkv = blk.attn.qkv(blk.norm1(t)).reshape(B, N, 3, C)
    dense = tm.norm(blk.attn.proj(qkv[:, :, 2])).float() @ P          # value path only
    cls = tm.norm(blk(t)).float()[:, 0] @ P
    g = dense[:, 1:].reshape(B, 16, 8, -1)
    n = torch.nn.functional.normalize
    return n(cls, dim=-1), n(n(g[:, 2:8], dim=-1).mean((1, 2)), dim=-1), n(n(g[:, 8:15], dim=-1).mean((1, 2)), dim=-1)


for name in ("PA100k", "PETA", "Market1501"):
    rows = np.where(va.ds.values == name)[0][:5000]
    E = [[], [], []]
    for i in range(0, len(rows), 32):
        x = torch.stack([ds[j][0] for j in rows[i:i + 32]]).cuda().to(torch.bfloat16)
        for k, v in enumerate(feats(x)):
            E[k].append(v.cpu())
    c, u, l = (torch.cat(e) for e in E)
    y = va[A].values[rows]
    s_cls, s_up, s_lo = (c @ W.cpu().T).numpy(), (u @ W.cpu().T).numpy(), (l @ W.cpu().T).numpy()
    ok = lambda k: 0 < y[:, k].sum() < len(y)
    for nm, ks, s in (("upper colours", UPPER, s_up), ("lower colours", LOWER, s_lo)):
        a_c = np.mean([auc(y[:, k], s_cls[:, k]) for k in ks if ok(k)]); a_r = np.mean([auc(y[:, k], s[:, k]) for k in ks if ok(k)])
        a_m = np.mean([auc(y[:, k], s_cls[:, k] + s[:, k]) for k in ks if ok(k)])
        print(f"{name:10s} {nm}: CLS {a_c:.4f}  MaskCLIP region {a_r:.4f}  CLS+region {a_m:.4f}", flush=True)
