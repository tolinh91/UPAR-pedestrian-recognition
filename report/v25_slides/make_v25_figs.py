"""Figures for the T1 v25 slides from the real v25 members (laptop GPU, weights memory-mapped, models built on the GPU).

../../../Upar-Challenge-Track-2/.venv/bin/python make_v25_figs.py
Members: hp.pt (DINOv3 ViT-H+/16, query head) from the v21/v25 package, F_clipT_w70.pt (CLIP ViT-L/14, query head +
language anchor, WiSE 0.7). Writes, for the example val images: the model inputs with their patch grids, PCA colours of
the patch tokens, the query attention of chosen attributes in both members, and numbers.json (per-attribute logits of
both members, the CLIP query / text split, the offset of the decision rule, the fused probability and the label).
"""
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from matplotlib import cm
from PIL import Image, ImageDraw

sys.path.insert(0, "/media/tan/F/RWS2027/Upar-Challenge-Track-2/par")
from upar.data import UPARDataset, attr_names, build_transforms, image_cache, load_split  # noqa: E402
from upar.models import build_model  # noqa: E402

OUT = Path(__file__).parent / "figs"; OUT.mkdir(exist_ok=True)
SUB = Path("/media/tan/F/RWS2027/Upar-Challenge-Track-1/submissions")
ROWS = [19023, 24738, 32671, 30465]
MAIN = 19023
SHOW = ["Hair-Length-Long", "UpperBody-Length-Short", "UpperBody-Color-White", "LowerBody-Color-Grey",
        "Accessory-Backpack", "Gender-Female"]
A = attr_names(); va = load_split("val"); buf, off = image_cache("val")
torch.set_grad_enabled(False)

z = zipfile.ZipFile(SUB / "t1_hp_clipTw70_gpu_v25.zip")
cfg = json.loads(z.read("assets/config.json"))
hp_path = OUT.parent / "hp.pt"
if not hp_path.exists():
    hp_path.write_bytes(z.read("assets/hp.pt"))
offset = np.array(cfg["offset"]) * cfg["offset_scale"]


def load_member(file, state_path):
    mcfg = dict([m for m in cfg["models"] if m["file"] == file][0]["model"], num_attrs=40)
    with torch.device("cuda"):
        model = build_model(mcfg, pretrained=False)
    state = torch.load(state_path, map_location="cpu", mmap=True, weights_only=True)
    model.load_state_dict({k: v.float() for k, v in state.items()})
    return model.eval().cuda(), mcfg


def attention(model, t):
    """Cross-attention of every attribute query over CLS + patches (registers dropped), mean over the 8 heads."""
    head = model.head
    tt = torch.cat([t[:, :1], t[:, model.n_prefix:]], 1)
    kv = head.kv_norm(tt)
    q = head.queries.unsqueeze(0).expand(t.shape[0], -1, -1)
    layer = head.layers[0]
    _, w = layer["attn"](layer["q_norm"](q), kv, kv, need_weights=True, average_attn_weights=True)
    return w[:, :, 1:]                                                   # (B, A, patches)


def overlay(img, att, grid, alpha=0.8):
    a = att.reshape(grid).astype(np.float32)
    a = ((a - a.min()) / (a.max() - a.min() + 1e-9)) ** 0.5               # gamma 0.5: weaker patches stay visible
    a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize(img.size, Image.BICUBIC)) / 255.0
    heat = cm.inferno(0.25 + 0.75 * a)[..., :3]
    base = np.asarray(img).astype(np.float32) / 255.0 * 0.8               # slightly dimmed photo under the heat map
    out = base * (1 - alpha * a[..., None]) + heat * alpha * a[..., None]
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))


def pca_image(tokens, grid, size):
    X = tokens.float().cpu().numpy(); X = X - X.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    P = X @ Vt[:3].T
    P = (P - np.percentile(P, 2, 0)) / (np.percentile(P, 98, 0) - np.percentile(P, 2, 0) + 1e-9)
    im = Image.fromarray((np.clip(P, 0, 1).reshape(*grid, 3) * 255).astype(np.uint8))
    return im.resize(size, Image.NEAREST)


def grid_image(img, grid, color):
    im = img.copy(); d = ImageDraw.Draw(im); W, H = im.size; gh, gw = grid
    for i in range(1, gh):
        d.line([(0, i * H / gh), (W, i * H / gh)], fill=color, width=2)
    for j in range(1, gw):
        d.line([(j * W / gw, 0), (j * W / gw, H)], fill=color, width=2)
    return im


hp, hp_cfg = load_member("hp.pt", hp_path)
cl, cl_cfg = load_member("m1.pt", SUB / "v25_parts/F_clipT_w70.pt")
numbers = {"attrs": A, "offset": offset.tolist(), "images": {}}
for name, model, mcfg, color in (("hp", hp, hp_cfg, (47, 127, 181)), ("clip", cl, cl_cfg, (224, 122, 46))):
    h, w = mcfg["size"]; ps = 16 if name == "hp" else 14; grid = (h // ps, w // ps)
    ds = UPARDataset("val", build_transforms((h, w), False, mean=mcfg["mean"], std=mcfg["std"]), rows=ROWS)
    x = torch.stack([ds[i][0] for i in range(len(ROWS))]).cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        t = model.tokens(x).float()
        att = attention(model, t).float().cpu().numpy()
        zq = model.head._logits(t, model.n_prefix)[0].float()
        zt = model.head.text_logits(t).float() if hasattr(model.head, "text_logits") else torch.zeros_like(zq)
    for b, r in enumerate(ROWS):
        img = Image.open(io.BytesIO(buf[off[r]:off[r + 1]].tobytes())).convert("RGB").resize((224, 448), Image.BICUBIC)
        img.save(OUT / f"input_{r}.png")
        grid_image(img, grid, color).save(OUT / f"grid_{name}_{r}.png")
        pca_image(t[b, model.n_prefix:], grid, (224, 448)).save(OUT / f"pca_{name}_{r}.png")
        for a in SHOW:
            overlay(img, att[b, A.index(a)], grid).save(OUT / f"attn_{name}_{r}_{a}.png")
        e = numbers["images"].setdefault(str(r), {"dataset": va.ds.values[r], "label": va[A].values[r].tolist()})
        e[f"{name}_query"] = zq[b].cpu().tolist(); e[f"{name}_text"] = zt[b].cpu().tolist()
for r, e in numbers["images"].items():
    zh = np.array(e["hp_query"]); zc = np.array(e["clip_query"]) + np.array(e["clip_text"])
    zf = (zh + zc) / 2 + offset
    e["fused_prob"] = (1 / (1 + np.exp(-zf))).tolist()
    e["pred"] = [A[k] for k in np.where(zf > 0)[0]]
    e["gt"] = [A[k] for k in np.where(np.array(e["label"]) > 0)[0]]
    print(r, e["dataset"], "pred:", e["pred"], "| gt:", e["gt"])
json.dump(numbers, open(OUT / "numbers.json", "w"))
print("figures in", OUT, len(list(OUT.glob("*.png"))))
