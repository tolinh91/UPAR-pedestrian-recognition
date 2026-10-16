"""One image through the v31 joint model: everything the architecture page animates, as JSON (real values)."""
import base64
import io
import json
import sys
from types import MethodType

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/media/tan/F/RWS2027/Upar-Challenge-Track-2/par")
from upar.data import UPARDataset, attr_names, build_transforms, image_cache, load_split  # noqa: E402
from upar.models import build_model  # noqa: E402

ROW = 30465
F = "/media/tan/F/RWS2027/remote_results/final_du"
cfg = json.load(open(f"{F}/du_cfg.json"))
torch.set_default_dtype(torch.bfloat16)
with torch.device("cuda"):
    model = build_model(cfg, pretrained=False)
torch.set_default_dtype(torch.float32)
sd = torch.load(f"{F}/du_bf16.pt", map_location="cpu", mmap=True, weights_only=True)
print(model.load_state_dict(sd, strict=True)); del sd
model.eval()

att = {}
def capture(name, head):
    mha = head.layers[0]["attn"]; orig = type(mha).forward
    def fwd(self, q, k, v, need_weights=False, attn_mask=None, orig=orig):
        out, w = orig(self, q, k, v, need_weights=True, attn_mask=attn_mask, average_attn_weights=False)
        att[name] = w.float()[0].cpu().numpy()          # (heads, A, tokens)
        return out, None
    mha.forward = MethodType(fwd, mha)
for n, h in (("h", model.h.head), ("c", model.c.head), ("j", model.joint)):
    capture(n, h)

A = attr_names(); va = load_split("val")
ds = UPARDataset("val", build_transforms((224, 112), False, mean=[0, 0, 0], std=[1, 1, 1]), rows=[ROW])
x = ds[0][0][None].cuda().to(torch.bfloat16)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    th = model.h.tokens((x - model.mean_h) / model.std_h)
    tc = model.c.tokens((x - model.mean_c) / model.std_c)
    zh = model.h.head(th, model.h.n_prefix).float()
    zc = model.c.head(tc, model.c.n_prefix).float()
    zc_text = model.c.head.text_logits(tc).float()
    kv = torch.cat([model.ph(torch.cat([th[:, :1], th[:, model.h.n_prefix:]], 1)) + model.src[0],
                    model.pc(torch.cat([tc[:, :1], tc[:, model.c.n_prefix:]], 1)) + model.src[1]], 1)
    zj = model.joint(kv, 1).float()
    zfull = model(x).float()
z = {"h": zh[0].cpu().numpy(), "c": zc[0].cpu().numpy(), "c_text": zc_text[0].cpu().numpy(), "j": zj[0].cpu().numpy()}
mean3 = (z["h"] + z["c"] + z["j"]) / 3
assert np.allclose(mean3, zfull[0].cpu().numpy(), atol=0.05), np.abs(mean3 - zfull[0].cpu().numpy()).max()


def pca_rgb(t):
    t = t - t.mean(0)
    u, s, vt = np.linalg.svd(t, full_matrices=False)
    p = t @ vt[:3].T
    p = (p - p.min(0)) / (p.max(0) - p.min(0) + 1e-6)
    return (p * 255).round().astype(int).tolist(), (s[:3] ** 2 / (s ** 2).sum()).round(3).tolist()


ph_tok = th[0, model.h.n_prefix:].float().cpu().numpy()          # (98, 1280)
pc_tok = tc[0, 1:].float().cpu().numpy()                         # (128, 1024)
rgb_h, var_h = pca_rgb(ph_tok); rgb_c, var_c = pca_rgb(pc_tok)
buf, off = image_cache("val")
raw = Image.open(io.BytesIO(buf[off[ROW]:off[ROW + 1]].tobytes())).convert("RGB")
inp = Image.fromarray((ds[0][0].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8))
def b64(im):
    b = io.BytesIO(); im.save(b, format="PNG"); return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()

gt = va[A].values[ROW].astype(int).tolist()
r = lambda a, k=3: np.round(np.asarray(a, float), k).tolist()   # noqa: E731
out = {
    "image": va.image.values[ROW], "raw_size": list(raw.size), "raw": b64(raw), "input": b64(inp),
    "attrs": A, "gt": gt,
    "logits": {k: r(v) for k, v in z.items()}, "logit_final": r(mean3), "prob": r(1 / (1 + np.exp(-mean3))),
    # attention of each head's queries (mean over the 8 attention heads), over its tokens
    "att_h": r(att["h"].mean(0), 4),        # (40, 1 + 98)   CLS + 14x7 H+ patches (16 px)
    "att_c": r(att["c"].mean(0), 4),        # (40, 1 + 128)  CLS + 16x8 CLIP patches (14 px)
    "att_j": r(att["j"].mean(0), 4),        # (40, 1 + 98 + 1 + 128)
    "att_heads_h": r(att["h"][:, A.index("Accessory-Hat")], 4),   # the 8 attention heads of the hat query (H+)
    "att_heads_j": r(att["j"][:, A.index("Accessory-Hat")], 4),   # the 8 attention heads of the joint hat query
    "j_head_hshare": r(att["j"][:, :, :99].sum(-1), 4),           # (8 heads, 40 attrs): attention mass on H+ tokens
    "att_j_heads": r(att["j"].transpose(1, 0, 2), 3),             # (40 attrs, 8 heads, 228 tokens)
    "att_h_heads": r(att["h"].transpose(1, 0, 2), 3),             # (40, 8, 99)
    "att_c_heads": r(att["c"].transpose(1, 0, 2), 3),             # (40, 8, 129)
    "head_dims": {"h": int(model.h.head.queries.shape[1]), "c": int(model.c.head.queries.shape[1]), "j": int(model.joint.queries.shape[1])},
    "pca_h": rgb_h, "pca_c": rgb_c, "pca_var_h": var_h, "pca_var_c": var_c,
    "grid_h": [14, 7], "grid_c": [16, 8],
    "tok_shapes": {"h": list(th.shape), "c": list(tc.shape), "kv": list(kv.shape)},
    "params": {k: int(sum(p.numel() for p in m.parameters())) for k, m in
               (("h_backbone", model.h.backbone), ("c_backbone", model.c.backbone), ("h_head", model.h.head),
                ("c_head", model.c.head), ("joint", model.joint), ("ph", model.ph), ("pc", model.pc))},
    "blocks": {"h": len(model.h.backbone.blocks), "c": len(model.c.backbone.blocks)},
    "cfg": cfg,
}
json.dump(out, open(sys.argv[1], "w"))
print("written", sys.argv[1], "tokens", out["tok_shapes"], "pred positives",
      [A[i] for i in range(40) if mean3[i] > 0], "gt", [A[i] for i in range(40) if gt[i]])
