"""Where does the Track 1 model look for each attribute?

For a packaged submission (default: v16 = the v13 model, DINOv3 ViT-L 176x96 + query head), per image and attribute:
  * attention: the query head's cross-attention of attribute k over the 11x6 patch tokens (heads averaged; one layer,
    no mask, so it is recomputed exactly from the head's projection weights);
  * attribution: Integrated Gradients of the attribute's logit w.r.t. the input pixels (16 steps in bf16, mean-image baseline; 12 attributes),
    summed over colour channels and pooled to 22x12 cells. Positive = pushes the attribute up, negative = down.
Writes xai_data.json (for the interactive page) and xai_grid.png (static overview).

Run: ../../../Upar-Challenge-Track-2/.venv/bin/python explain.py <unzipped package dir>
"""
import base64, io, json, sys, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PKG = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
sys.path.insert(0, str(PKG / "vendor"))
from upar.models import build_model  # noqa: E402

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "UPAR-Challenge-2027" / "data"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

cfg = json.load(open(PKG / "assets" / "config.json"))
mcfg = dict(cfg["model"], num_attrs=len(cfg["attribute_names"]))
model = build_model(mcfg, pretrained=False)
model.load_state_dict({k: v.float() for k, v in torch.load(PKG / "assets/model.pt", map_location="cpu", weights_only=True).items()})
model.eval().to(DEV)
H, W = cfg["size"]
MEAN = torch.tensor(mcfg["mean"]).view(1, 3, 1, 1).to(DEV); STD = torch.tensor(mcfg["std"]).view(1, 3, 1, 1).to(DEV)
OFFSET = torch.tensor(cfg["offset"]).to(DEV) * cfg.get("offset_scale", 1.0)
NAMES = cfg["attribute_names"]
head = model.head
GH, GW = H // 16, W // 16

captured = {}
def hook(mod, inp, out):  # tokens entering the head, and n_prefix
    captured["tokens"], captured["n_prefix"] = inp[0], inp[1]
head.register_forward_hook(hook)


def attention(tokens, n_prefix):
    """(B, A, GH*GW) attention of every attribute query over the patch tokens, heads averaged (CLS weight dropped,
    remaining weights renormalised so the map shows where among the patches the query looks)."""
    t = torch.cat([tokens[:, :1], tokens[:, n_prefix:]], 1)
    kv = head.kv_norm(t)
    layer = head.layers[0]; mha = layer["attn"]
    q = layer["q_norm"](head.queries)[None].expand(t.shape[0], -1, -1)
    Wq, Wk, _ = mha.in_proj_weight.chunk(3); bq, bk, _ = mha.in_proj_bias.chunk(3)
    Q = q @ Wq.T + bq; K = kv @ Wk.T + bk
    nh = mha.num_heads; d = Q.shape[-1] // nh
    Q = Q.view(*Q.shape[:2], nh, d).transpose(1, 2); K = K.view(*K.shape[:2], nh, d).transpose(1, 2)
    a = (Q @ K.transpose(-1, -2) / d ** 0.5).softmax(-1).mean(1)  # (B, A, 1+N)
    cls = a[..., 0]; p = a[..., 1:]
    return p / p.sum(-1, keepdim=True), cls


def prep(path):
    img = Image.open(path).convert("RGB").resize((W, H), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(img, np.float32) / 255).permute(2, 0, 1)[None].to(DEV)
    return img, x


def integrated_gradients(x, ks, steps=16):
    """IG for several attributes: one bf16 batch of interpolated images, one backward per attribute.
    Returns {attribute: (H/8, W/8) map}, channel-summed and 8x8-pooled."""
    base = torch.zeros_like(x) + MEAN  # normalised baseline = 0 (mean colour image)
    alphas = torch.linspace(1 / steps, 1, steps, device=DEV).view(-1, 1, 1, 1)
    xs = (base + alphas * (x - base)).requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
        logits = model((xs - MEAN) / STD).float()
    out = {}
    for j, k in enumerate(ks):
        g, = torch.autograd.grad(logits[:, k].sum(), xs, retain_graph=j < len(ks) - 1)
        ig = ((x - base) * g.mean(0, keepdim=True)).sum(1)[0]
        out[k] = torch.nn.functional.avg_pool2d(ig[None, None], 8)[0, 0]
    return out


def pick_images():
    out = []
    for ds, sub, n in (("PETA", "images", 8), ("PA100k", "release_data/release_data", 8), ("Market1501", "bounding_box_train", 8)):
        files = sorted((DATA / ds / sub).glob("*"))
        rng = np.random.default_rng(1)
        out += [(ds, files[i]) for i in rng.choice(len(files), n, replace=False)]
    return out


IG_NAMES = ["Accessory-Hat", "Accessory-Backpack", "Accessory-Bag", "Accessory-Glasses-Normal", "Gender-Female",
            "Age-Adult", "Hair-Length-Long", "UpperBody-Length-Short", "LowerBody-Length-Short",
            "LowerBody-Type-Skirt&Dress", "UpperBody-Color-Black", "LowerBody-Color-Black"]
IG_ATTRS = [NAMES.index(n) for n in IG_NAMES if n in NAMES]
print("IG attributes:", [NAMES[k] for k in IG_ATTRS], flush=True)
records = []
for ds, path in pick_images():
    img, x = prep(path)
    with torch.no_grad():
        logit = model((x - MEAN) / STD)
        att, cls = attention(captured["tokens"], captured["n_prefix"])
    prob = torch.sigmoid(logit + OFFSET)[0]
    t0 = time.time()
    igd = integrated_gradients(x, IG_ATTRS)
    ig = [igd[k].float().cpu().numpy().round(5).tolist() if k in igd else None for k in range(len(NAMES))]
    buf = io.BytesIO(); img.save(buf, "PNG")
    records.append({"dataset": ds, "file": path.name, "image": base64.b64encode(buf.getvalue()).decode(),
                    "prob": prob.cpu().numpy().round(3).tolist(),
                    "attention": att[0].cpu().numpy().round(4).tolist(), "cls": cls[0].cpu().numpy().round(3).tolist(),
                    "ig": ig})
    print(f"{time.time() - t0:.1f}s", ds, path.name, "top:", [NAMES[i] for i in prob.argsort(descending=True)[:5].tolist()], flush=True)

json.dump({"names": NAMES, "size": [H, W], "grid": [GH, GW], "ig_grid": [H // 8, W // 8], "records": records},
          open(HERE / "xai_data.json", "w"))

# static overview: 6 images x (image, 4 attribute attention overlays)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
show = ["Accessory-Hat", "Accessory-Backpack", "UpperBody-Color-Black", "LowerBody-Color-Black", "Age-Adult", "Gender-Female"]
show = [s for s in show if s in NAMES]
recs = [records[i] for i in (0, 1, 8, 9, 16, 17)]
fig, axs = plt.subplots(len(recs), 1 + len(show), figsize=(1.25 * (1 + len(show)), 2.2 * len(recs)))
for r, rec in enumerate(recs):
    im = np.asarray(Image.open(io.BytesIO(base64.b64decode(rec["image"]))))
    axs[r, 0].imshow(im); axs[r, 0].set_title(rec["dataset"], fontsize=7)
    for c, name in enumerate(show):
        k = NAMES.index(name); a = np.array(rec["attention"][k]).reshape(GH, GW)
        a = np.kron(a, np.ones((16, 16)))
        ax = axs[r, c + 1]; ax.imshow(im); ax.imshow(a, cmap="inferno", alpha=.55)
        short = {"Accessory-Hat": "Hat", "Accessory-Backpack": "Backpack", "UpperBody-Color-Black": "Upper black",
                 "LowerBody-Color-Black": "Lower black", "Age-Adult": "Adult", "Gender-Female": "Female"}[name]
        ax.set_title(f"{short}\np={rec['prob'][k]:.2f}", fontsize=6.5)
for ax in axs.flat:
    ax.axis("off")
fig.tight_layout(); fig.savefig(HERE / "xai_grid.png", dpi=150); print("saved")
