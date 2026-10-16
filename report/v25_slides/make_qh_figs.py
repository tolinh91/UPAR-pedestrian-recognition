"""Query-head illustration from the real v25 H+ member: attention of chosen queries, attention mass per body region vs
uniform pooling, and the most attended patches of the hat query (laptop GPU)."""
import io, json, sys, zipfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from matplotlib import cm
sys.path.insert(0, "/media/tan/F/RWS2027/Upar-Challenge-Track-2/par")
from upar.data import UPARDataset, attr_names, build_transforms, image_cache, load_split  # noqa: E402
from upar.models import build_model  # noqa: E402

OUT = Path("figs"); A = attr_names(); va = load_split("val"); buf, off = image_cache("val")
SUB = Path("/media/tan/F/RWS2027/Upar-Challenge-Track-1/submissions")
cfg = json.loads(zipfile.ZipFile(SUB / "t1_hp_clipTw70_gpu_v25.zip").read("assets/config.json"))
mcfg = dict([m for m in cfg["models"] if m["file"] == "hp.pt"][0]["model"], num_attrs=40)
with torch.device("cuda"):
    model = build_model(mcfg, pretrained=False)
model.load_state_dict({k: v.float() for k, v in torch.load("hp.pt", map_location="cpu", mmap=True, weights_only=True).items()})
model = model.eval().cuda(); torch.set_grad_enabled(False)
ROWS = [19023, 24738, 32671, 30465]; MAIN = 30465
ds = UPARDataset("val", build_transforms((224, 112), False, mean=mcfg["mean"], std=mcfg["std"]), rows=ROWS)
x = torch.stack([ds[i][0] for i in range(len(ROWS))]).cuda()
with torch.autocast("cuda", dtype=torch.bfloat16):
    t = model.tokens(x).float()
    head = model.head; tt = torch.cat([t[:, :1], t[:, model.n_prefix:]], 1); kv = head.kv_norm(tt)
    q = head.queries.unsqueeze(0).expand(len(ROWS), -1, -1); L = head.layers[0]
    _, w = L["attn"](L["q_norm"](q), kv, kv, need_weights=True, average_attn_weights=True)
att = w[:, :, 1:].float().cpu().numpy().reshape(len(ROWS), 40, 14, 7)     # patch attention (CLS share dropped)
att = att / att.sum((2, 3), keepdims=True)
regions = {"Accessory-Hat": (0, 3), "Hair-Length-Long": (0, 3), "UpperBody-Color-White": (2, 8), "Accessory-Backpack": (2, 9),
           "LowerBody-Color-Black": (7, 14), "LowerBody-Type-Trousers&Shorts": (7, 14)}
res = {}
for a, (r0, r1) in regions.items():
    k = A.index(a); m = att[:, k, r0:r1].sum((1, 2))
    res[a] = {"rows": [r0, r1], "uniform": (r1 - r0) / 14, "per_image": m.round(3).tolist(), "mean": float(m.mean())}
    print(f"{a:32s} rows {r0}-{r1 - 1}: attention mass {m.mean():.2f} (images {np.round(m, 2)}), uniform {(r1 - r0) / 14:.2f}")
b = ROWS.index(MAIN)
img = Image.open(io.BytesIO(buf[off[MAIN]:off[MAIN + 1]].tobytes())).convert("RGB").resize((112, 224), Image.BICUBIC)
big = img.resize((224, 448), Image.BICUBIC)
def overlay(a2d, alpha=0.8):
    a = (a2d - a2d.min()) / (a2d.max() - a2d.min() + 1e-9) if a2d.max() > a2d.min() else np.full_like(a2d, 0.35)
    a = a ** 0.5
    a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize(big.size, Image.BICUBIC)) / 255.0
    heat = cm.inferno(0.25 + 0.75 * a)[..., :3]; base = np.asarray(big).astype(np.float32) / 255 * 0.8
    return Image.fromarray((np.clip(base * (1 - alpha * a[..., None]) + heat * alpha * a[..., None], 0, 1) * 255).astype(np.uint8))
for a in ("Accessory-Hat", "Accessory-Backpack", "UpperBody-Color-White", "LowerBody-Color-Black"):
    overlay(att[b, A.index(a)]).save(OUT / f"qh_{MAIN}_{a}.png")
overlay(np.full((14, 7), 1 / 98)).save(OUT / f"qh_{MAIN}_uniform.png")
k = A.index("Accessory-Hat"); flat = att[b, k].ravel(); top = np.argsort(-flat)[:4]
box = big.copy(); d = ImageDraw.Draw(box)
for n, j in enumerate(top):
    r, c = divmod(int(j), 7)
    img.crop((c * 16, r * 16, c * 16 + 16, r * 16 + 16)).resize((96, 96), Image.NEAREST).save(OUT / f"qh_{MAIN}_top{n + 1}.png")
    d.rectangle([c * 32, r * 32, c * 32 + 31, r * 32 + 31], outline=(240, 154, 85), width=3)
    print(f"top {n + 1}: row {r} col {c} weight {flat[j] * 100:.1f}%")
box.save(OUT / f"qh_{MAIN}_hat_boxes.png")
d0 = json.load(open(OUT / "numbers.json"))["images"][str(MAIN)]
print("H+ hat logit", round(d0["hp_query"][k], 2), "label", d0["label"][k], "top4 share", round(float(flat[top].sum()), 3))
json.dump({"regions": res, "hat_top4": [float(flat[j]) for j in top], "hat_logit_hp": d0["hp_query"][k]}, open(OUT / "qh_numbers.json", "w"))
