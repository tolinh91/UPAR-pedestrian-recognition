"""Numbers for the v25 slides that explain the text anchor, the loss and the fusion (laptop GPU, real v25 members).

1. text-anchor similarity: <normalize(P cls), w0_k> of the v25 CLIP member for chosen images and attributes
   (w0_k = t(sentence_k) - t("a person"), from the packaged weights; P = frozen CLIP projection);
2. training loss per attribute of each member on the example image (BCE, DeepMAR weights, label smoothing 0.05);
3. step-by-step fusion for chosen attributes.
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/media/tan/F/RWS2027/Upar-Challenge-Track-2/par")
from upar.data import UPARDataset, attr_names, build_transforms, load_split  # noqa: E402
import zipfile  # noqa: E402
from pathlib import Path  # noqa: E402
from upar.models import build_model  # noqa: E402

SUB = Path("/media/tan/F/RWS2027/Upar-Challenge-Track-1/submissions")
cfg = json.loads(zipfile.ZipFile(SUB / "t1_hp_clipTw70_gpu_v25.zip").read("assets/config.json"))
offset = np.array(cfg["offset"]) * cfg["offset_scale"]


def load_member(file, state_path):
    mcfg = dict([m for m in cfg["models"] if m["file"] == file][0]["model"], num_attrs=40)
    with torch.device("cuda"):
        model = build_model(mcfg, pretrained=False)
    state = torch.load(state_path, map_location="cpu", mmap=True, weights_only=True)
    model.load_state_dict({k: v.float() for k, v in state.items()})
    return model.eval().cuda(), mcfg

A = attr_names()
d = json.load(open("figs/numbers.json"))
cl, cl_cfg = load_member("m1.pt", SUB / "v25_parts/F_clipT_w70.pt")
h, w = cl_cfg["size"]
rows = [19023, 24738, 32671, 30465]
ds = UPARDataset("val", build_transforms((h, w), False, mean=cl_cfg["mean"], std=cl_cfg["std"]), rows=rows)
x = torch.stack([ds[i][0] for i in range(len(rows))]).cuda()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    t = cl.tokens(x).float()
head = cl.head
e = torch.nn.functional.normalize(t[:, 0] @ head.proj, dim=-1)          # image point in the CLIP image-text space
w0 = head.text_w0                                                         # sentence directions (40, 768)
sim = (e @ w0.T).cpu().numpy()                                            # cosine-difference to the neutral sentence
print("delta-cosine <e, t(sentence) - t(a person)> per image:")
for a in ("Accessory-Hat", "Accessory-Backpack", "Gender-Female", "Hair-Length-Long", "UpperBody-Color-White", "UpperBody-Color-Blue", "LowerBody-Color-Black"):
    k = A.index(a)
    print(f"  {a:26s} " + "  ".join(f"{r}:{sim[i, k]:+.4f}" for i, r in enumerate(rows)) + f"   |w0|={w0[k].norm():.3f}")
print("text-path logit pieces for 19023 Hat: s*<e,w0>", 100 * sim[0, A.index('Accessory-Hat')],
      "delta part", (100 * 0.1 * e[0] @ head.text_delta[A.index('Accessory-Hat')]).item(), "bias", head.text_b[A.index('Accessory-Hat')].item())

# training loss on the example (BCE + DeepMAR + smoothing 0.05), per member
va = load_split("val"); tr = load_split("train")
import pandas as pd
pi = pd.concat([tr, va])[A].values.mean(0).clip(1e-4, 1 - 1e-4)
wpos, wneg = np.exp(1 - pi), np.exp(pi)
y = np.array(d["images"]["19023"]["label"], float); tgt = y * 0.95 + 0.025
for name, z in (("H+", np.array(d["images"]["19023"]["hp_query"])),
                ("CLIP", np.array(d["images"]["19023"]["clip_query"]) + np.array(d["images"]["19023"]["clip_text"]))):
    bce = np.logaddexp(0, z) - tgt * z
    wl = np.where(y > 0.5, wpos, wneg) * bce
    o = np.argsort(-wl)[:6]
    print(f"loss {name}: mean {wl.mean():.3f}; largest:", ", ".join(f"{A[k]} {wl[k]:.2f} (z {z[k]:+.2f}, y {y[k]:.0f})" for k in o))
for a in ("Accessory-Hat", "Accessory-Backpack", "Gender-Female", "LowerBody-Type-Trousers&Shorts", "Accessory-Glasses-Sun", "Age-Adult"):
    k = A.index(a); print(f"  weights {a:30s} prior {pi[k]:.3f} w+ {wpos[k]:.2f} w- {wneg[k]:.2f}")
# fusion steps
for r in ("19023", "32671"):
    E = d["images"][r]
    for a in ("Gender-Female", "Accessory-Backpack", "Accessory-Hat", "Accessory-Glasses-Normal", "Hair-Length-Bald", "Age-Adult", "Age-Young"):
        k = A.index(a); zh = E["hp_query"][k]; zc = E["clip_query"][k] + E["clip_text"][k]; m = (zh + zc) / 2
        print(f"fusion {r} {a:26s} zH {zh:+.2f} zC {zc:+.2f} mean {m:+.2f} off {offset[k]:+.2f} total {m + offset[k]:+.2f} p {1/(1+np.exp(-(m+offset[k]))):.2f} label {E['label'][k]:.0f}")
