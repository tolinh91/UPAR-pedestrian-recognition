"""Data for the architecture figure: one PETA image (unseen by both encoders), patch grids, PCA colours of the
final patch tokens, query attention and logits for four attributes, from the L-size members trained without PETA.
RAM-light: mmap checkpoints, models built on the GPU, one at a time."""
import gc, json, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, "/media/tan/F/RWS2027/Upar-Challenge-Track-2/par")
from upar.data import DATA, attr_names, build_transforms, load_split
from upar.models import build_model

OUT = Path(__file__).parent
CK = Path("/media/tan/F/RWS2027/remote_results/rtx5090_t1")
A = attr_names(); va = load_split("val"); tr = load_split("train")
p = tr[tr.ds != "PETA"][A].values.mean(0).clip(1e-3, 1 - 1e-3)
b = np.log((1 - p) / p) - (1 - 2 * p); TAU = 0.42
zD = np.load(CK / "runs/T_dinoL/held_val_logits_last.npz"); zC = np.load(CK / "runs/T_clipL/held_val_logits_last.npz")
idx = zD["idx"]; zm = (zD["logits"] + zC["logits"]) / 2 + TAU * b
y = va[A].values[idx]; pred = zm > 0
ATT = ["Gender-Female", "Hair-Length-Long", "Accessory-Backpack", "Accessory-Bag", "LowerBody-Type-Skirt&Dress",
       "Accessory-Hat", "UpperBody-Length-Short", "LowerBody-Color-Black"]
best = None
for i in range(len(idx)):
    if (pred[i] != (y[i] == 1)).sum() > 1:    # at most one wrong attribute overall
        continue
    pos = [a for a in ATT if y[i, A.index(a)] == 1]
    if len(pos) < 3 or "Gender-Female" not in pos:
        continue
    im = Image.open(DATA / va.image.values[idx[i]]).convert("L"); w, h = im.size
    bright = np.asarray(im).mean() / 255
    if bright < 0.38 or h < 160:
        continue
    score = len(pos) * 1000 + w * h / 100
    if best is None or score > best[0]:
        best = (score, i, pos)
_, i, pos = best
path = DATA / va.image.values[idx[i]]
show = pos[:3] + [a for a in ATT if a not in pos and y[i, A.index(a)] == 0][:1]   # three positives, one negative
print("image", path, Image.open(path).size, "show", show)
res = {"path": str(path), "attrs": show, "labels": [int(y[i, A.index(a)]) for a in show],
       "b": [float(b[A.index(a)]) for a in show], "tau": TAU}


def pca_rgb(tok):
    x = tok - tok.mean(0); u, s, vt = np.linalg.svd(x, full_matrices=False)
    c = x @ vt[:3].T; c = (c - c.min(0)) / (c.max(0) - c.min(0) + 1e-6)
    return c


for key, run in {"D": CK / "ckpt/T_dinoL", "C": CK / "ckpt/T_clipL"}.items():
    ck = torch.load(run / "last.pt", map_location="cpu", weights_only=False, mmap=True)
    cfg = ck["model_cfg"]
    with torch.device("cuda"):
        m = build_model(cfg, pretrained=False)
    m.load_state_dict(ck["model"]); m.eval()
    h, w = cfg["size"]; ps = 16 if "dinov3" in cfg["backbone"] else 14; gh, gw = h // ps, w // ps
    tf = build_transforms((h, w), False, mean=cfg.get("mean") or [0.485, 0.456, 0.406], std=cfg.get("std") or [0.229, 0.224, 0.225])
    with torch.no_grad():
        x = tf(Image.open(path).convert("RGB"))[None].cuda()
        t = m.tokens(x); n = m.n_prefix
        res[f"pca{key}"] = pca_rgb(t[0, n:].float().cpu().numpy()).reshape(gh, gw, 3).tolist()
        head = m.head; kv = head.kv_norm(torch.cat([t[:, :1], t[:, n:]], 1))
        layer = head.layers[0]; mha = layer["attn"]
        q = layer["q_norm"](head.queries)[None]
        Wq, Wk, _ = mha.in_proj_weight.chunk(3); bq, bk, _ = mha.in_proj_bias.chunk(3)
        Q = q @ Wq.T + bq; K = kv @ Wk.T + bk; nh = mha.num_heads; d = Q.shape[-1] // nh
        Q = Q.view(1, -1, nh, d).transpose(1, 2); K = K.view(1, -1, nh, d).transpose(1, 2)
        a = (Q @ K.transpose(-1, -2) / d ** 0.5).softmax(-1).mean(1)[0, :, 1:]
        res[f"att{key}"] = [(a[A.index(at)] / a[A.index(at)].sum()).reshape(gh, gw).cpu().numpy().tolist() for at in show]
        z = m.head(t, n)[0].float().cpu().numpy()
        res[f"z{key}"] = [float(z[A.index(at)]) for at in show]
        res[f"grid{key}"] = [gh, gw]; res[f"blocks{key}"] = len(m.backbone.blocks)
    print(key, cfg["backbone"], (h, w), "grid", (gh, gw), "blocks", res[f"blocks{key}"], "z", np.round(res[f"z{key}"], 2), "maxRSS %.0f MB" % (__import__("resource").getrusage(0).ru_maxrss / 1024), flush=True)
    del m, ck; gc.collect(); torch.cuda.empty_cache()
Image.open(path).convert("RGB").resize((112, 224), Image.BICUBIC).save(OUT / "arch_input.png")
json.dump(res, open(OUT / "arch_data.json", "w"))
