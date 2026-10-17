"""How concentrated is the attention of each attribute query? (query heads of a T1 member or of the joint model)

python attn_stats.py RUN_OR_PT [CFG_JSON] [--n 512]
RUN_OR_PT: a run directory (last.pt) or a bf16 export (du_bf16.pt + its cfg json). For every query head of the model
(PARModel.head, or DualPARModel h / c / joint) and every attribute: mean over images and attention heads of the largest
attention weight and of the effective number of attended tokens exp(entropy). Images: the first N PETA val images.
"""
import argparse
import json
from types import MethodType

import numpy as np
import torch
from torch.utils.data import DataLoader

from upar.data import UPARDataset, attr_names, build_transforms, load_split
from upar.models import DualPARModel, build_model, cfg_from_checkpoint
from upar.models.heads import QueryHead

GROUPS = {"colours": lambda a: "-Color-" in a, "clothing type / length": lambda a: a.startswith(("UpperBody-Length", "LowerBody-Length", "LowerBody-Type")),
          "head / accessories": lambda a: a.startswith(("Accessory", "Hair")), "age / gender": lambda a: a.startswith(("Age", "Gender"))}


def capture(head):
    """Keep the attention weights (B, heads, A, tokens) of the head's cross-attention layers on head._attn."""
    head._attn = []
    for layer in head.layers:
        mha = layer["attn"]
        orig = type(mha).forward   # nn.MultiheadAttention or TempMultiheadAttention (per-query temperature)

        def fwd(self, q, k, v, need_weights=False, attn_mask=None, orig=orig):
            out, w = orig(self, q, k, v, need_weights=True, attn_mask=attn_mask, average_attn_weights=False)
            self._owner._attn.append(w.float())
            return out, None
        mha._owner = head
        mha.forward = MethodType(fwd, mha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("cfg", nargs="?")
    ap.add_argument("--n", type=int, default=512)
    a = ap.parse_args()
    if a.src.endswith(".pt"):
        cfg = json.load(open(a.cfg)); sd = torch.load(a.src, map_location="cpu", weights_only=True)
    else:
        ck = torch.load(f"{a.src}/last.pt", map_location="cpu", weights_only=False); cfg, sd = cfg_from_checkpoint(ck), ck["model"]
    model = build_model(cfg, pretrained=False); model.load_state_dict({k: v.float() for k, v in sd.items()}); model = model.cuda().eval()
    heads = {"joint": model.joint, "H+ head": model.h.head, "CLIP head": model.c.head} if isinstance(model, DualPARModel) \
        else {"head": model.head}
    heads = {k: h for k, h in heads.items() if isinstance(h, QueryHead)}
    for h in heads.values():
        capture(h)
    norm = {k: cfg[k] for k in ("mean", "std") if cfg.get(k)}
    va = load_split("val"); rows = np.where(va.ds.values == "PETA")[0][:a.n]
    dl = DataLoader(UPARDataset("val", build_transforms(tuple(cfg["size"]), False, **norm), rows=rows), 64, num_workers=8)
    stats = {k: {"max": [], "eff": []} for k in heads}
    with torch.no_grad():
        for x, *_ in dl:
            for h in heads.values():
                h._attn = []
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(x.cuda())
            for k, h in heads.items():
                w = h._attn[-1].clamp_min(1e-12)                           # (B, heads, A, tokens)
                stats[k]["max"].append(w.amax(-1).mean(1).cpu())           # (B, A)
                stats[k]["eff"].append((-(w * w.log()).sum(-1)).exp().mean(1).cpu())
    A = attr_names()
    n_tok = None
    print(f"{len(rows)} PETA val images; max = largest attention weight, eff = exp(entropy) = effective tokens attended")
    for k in heads:
        mx = torch.cat(stats[k]["max"]).mean(0).numpy(); ef = torch.cat(stats[k]["eff"]).mean(0).numpy()
        n_tok = n_tok or "?"
        line = " | ".join(f"{g}: max {np.mean([mx[i] for i, an in enumerate(A) if f(an)]):.2f} eff {np.mean([ef[i] for i, an in enumerate(A) if f(an)]):.1f}"
                          for g, f in GROUPS.items())
        print(f"{k:10s} {line}")
        top = np.argsort(-mx)[:5]
        print(f"{'':10s} most concentrated: " + ", ".join(f"{A[i]} {mx[i]:.2f}/{ef[i]:.1f}" for i in top))


if __name__ == "__main__":
    main()
