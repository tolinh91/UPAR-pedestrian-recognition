"""How the joint query head of a dual run splits its attention between the two encoders, over all held-out images.

python routing_stats.py ../runs/DU32_PETA_s1 out.npz
For every held-out image, attention head h and attribute a: s = attention mass of the joint query on DINOv3 tokens
(CLS + 98 patches; the other 129 tokens are CLIP). Saves
  mean_share (8, 40)   mean of s over images
  frac_dino  (8, 40)   share of images where the pair reads mostly DINOv3 (s > 0.5)
  hist       (50,)     histogram of s over all (image, head, attribute) triples, bins on [0, 1]
  pure99     scalar    share of triples with s > 0.99 or s < 0.01
  n_dino     (N, 40)   per image and attribute: number of attention heads with s > 0.5
plus n_images and holdout. Same preprocessing as logit_extract.py (no flip, bf16 autocast).
"""
import argparse
from types import MethodType

import numpy as np
import torch
from torch.utils.data import DataLoader

from upar.data import UPARDataset, build_transforms, load_split
from upar.models import build_model, cfg_from_checkpoint

HOLD = {"PETA": "PETA", "Market1501": "Market1501", "PA100k": "PA100k"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("out")
    a = ap.parse_args()
    ck = torch.load(f"{a.run}/last.pt", map_location="cpu", weights_only=False)
    cfg = cfg_from_checkpoint(ck)
    hold = str(ck["args"]["holdout"])
    norm = {k: cfg[k] for k in ("mean", "std") if cfg.get(k)}
    va = load_split("val")
    rows = np.where(va.ds.values == HOLD[hold])[0].tolist()
    ds = UPARDataset("val", build_transforms(tuple(cfg["size"]), False, **norm), rows=rows)
    dl = DataLoader(ds, 128, num_workers=16, pin_memory=True)
    model = build_model(cfg, pretrained=False).cuda().eval()
    model.load_state_dict(ck["model"]); del ck
    grab = {}
    mha = model.joint.layers[0]["attn"]; orig = type(mha).forward

    def fwd(self, q, k, v, need_weights=False, attn_mask=None, orig=orig):
        out, w = orig(self, q, k, v, need_weights=True, attn_mask=attn_mask, average_attn_weights=False)
        grab["w"] = w                                         # (B, 8, 40, 228)
        return out, None
    mha.forward = MethodType(fwd, mha)

    n_d = None
    S, F, N = 0, 0, 0
    hist = np.zeros(50); pure = 0; total = 0; n_dino = []
    with torch.no_grad():
        for x, *_ in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(x.cuda(non_blocking=True))
            w = grab["w"].float()
            if n_d is None:                                   # DINOv3 tokens come first in the joint keys
                n_d = w.shape[-1] - 129
                assert n_d == 99, w.shape
            s = w[..., :n_d].sum(-1) / w.sum(-1)              # (B, 8, 40)
            S = S + s.sum(0).cpu().numpy(); F = F + (s > .5).float().sum(0).cpu().numpy(); N += len(s)
            hist += np.histogram(s.cpu().numpy().ravel(), bins=50, range=(0, 1))[0]
            pure += int(((s > .99) | (s < .01)).sum()); total += s.numel()
            n_dino.append((s > .5).sum(1).to(torch.uint8).cpu().numpy())
    res = dict(mean_share=S / N, frac_dino=F / N, hist=hist, pure99=pure / total, n_dino=np.concatenate(n_dino),
               n_images=N, holdout=hold)
    np.savez(a.out, **res)
    fd = res["frac_dino"]
    stable = np.mean((fd > .9) | (fd < .1))
    print(f"{a.run}: {N} held-out images ({hold}); pure (>99% one encoder) {100 * res['pure99']:.1f}% of triples; "
          f"pairs that read the same encoder on >90% of images {100 * stable:.1f}%; "
          f"heads on DINOv3 per attribute {res['n_dino'].mean():.2f} (range {res['n_dino'].min()}-{res['n_dino'].max()})", flush=True)


if __name__ == "__main__":
    main()
