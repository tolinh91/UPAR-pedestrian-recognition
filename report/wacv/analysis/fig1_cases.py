"""Method-figure data on the unseen domain (PETA val), from the L-size members trained without PETA.
../../../../Upar-Challenge-Track-2/.venv/bin/python fig1_cases.py  -> fig1_cases.json + crops (RAM-light: mmap, one model at a time)
Cases: agree (both encoders right and confident), ens (they disagree after the rule, the average is right),
rule (both raw logits say no, the label is yes, the rule turns the average to yes). One image per case."""
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
assert (zD["idx"] == zC["idx"]).all()
idx = zD["idx"]; zd, zc = zD["logits"].astype(np.float64), zC["logits"].astype(np.float64); zm = (zd + zc) / 2
y = va[A].values[idx]
ATTRS = ["Accessory-Hat", "Accessory-Backpack", "LowerBody-Type-Skirt&Dress", "Accessory-Glasses-Normal",
         "UpperBody-Color-Red", "Gender-Female", "Hair-Length-Long", "Accessory-Bag", "UpperBody-Color-Pink"]
used, cases = set(), []
for kind in ("agree", "ens", "rule"):
    best = None
    for name in ATTRS:
        if name in [c[1] for c in cases]:
            continue
        k = A.index(name); s = TAU * b[k]; yy = y[:, k] == 1
        dD, dC, dM = zd[:, k] + s > 0, zc[:, k] + s > 0, zm[:, k] + s > 0
        if kind == "agree":
            ok, score = yy & (zd[:, k] + s > 2) & (zc[:, k] + s > 2), np.minimum(zd[:, k], zc[:, k])
        elif kind == "ens":
            ok = yy & (dD != dC) & dM & (np.abs(zd[:, k] + s) > 0.75) & (np.abs(zc[:, k] + s) > 0.75) & (zm[:, k] + s > 0.5)
            score = np.minimum(zm[:, k] + s, np.abs(zd[:, k] - zc[:, k]) / 2)
        else:
            ok = yy & (zd[:, k] < -0.3) & (zc[:, k] < -0.3) & (zm[:, k] < -0.3) & (zm[:, k] + s > 0.3)
            score = np.minimum(zm[:, k] + s, -zm[:, k])
        for i in np.argsort(-np.where(ok, score, -1e9))[:20]:
            if ok[i] and i not in used:
                if best is None or score[i] > best[0]:
                    best = (score[i], name, int(i))
                break
    used.add(best[2]); cases.append((kind, best[1], best[2]))
print(cases, flush=True)

RUNS = {"D": CK / "ckpt/T_dinoL", "C": CK / "ckpt/T_clipL"}
res = [{"kind": kd, "attr": nm, "i": i, "path": str(DATA / va.image.values[idx[i]]), "label": int(y[i, A.index(nm)]),
        "b": float(b[A.index(nm)]), "zD": float(zd[i, A.index(nm)]), "zC": float(zc[i, A.index(nm)])} for kd, nm, i in cases]


def attention(m, x, k, grid):
    t = m.tokens(x); head = m.head; n = m.n_prefix
    kv = head.kv_norm(torch.cat([t[:, :1], t[:, n:]], 1))
    layer = head.layers[0]; mha = layer["attn"]
    q = layer["q_norm"](head.queries)[None]
    Wq, Wk, _ = mha.in_proj_weight.chunk(3); bq, bk, _ = mha.in_proj_bias.chunk(3)
    Q = q @ Wq.T + bq; K = kv @ Wk.T + bk; nh = mha.num_heads; d = Q.shape[-1] // nh
    Q = Q.view(1, -1, nh, d).transpose(1, 2); K = K.view(1, -1, nh, d).transpose(1, 2)
    a = (Q @ K.transpose(-1, -2) / d ** 0.5).softmax(-1).mean(1)[0]
    pt = a[k, 1:]
    return (pt / pt.sum()).reshape(grid).float().cpu().numpy()


for key, run in RUNS.items():
    ck = torch.load(run / "last.pt", map_location="cpu", weights_only=False, mmap=True)
    cfg = ck["model_cfg"]
    with torch.device("cuda"):
        m = build_model(cfg, pretrained=False)
    m.load_state_dict(ck["model"]); m.eval()
    h, w = cfg["size"]; ps = 16 if "dinov3" in cfg["backbone"] else 14
    tf = build_transforms((h, w), False, mean=cfg.get("mean") or [0.485, 0.456, 0.406], std=cfg.get("std") or [0.229, 0.224, 0.225])
    with torch.no_grad():
        for r in res:
            k = A.index(r["attr"]); x = tf(Image.open(r["path"]).convert("RGB"))[None].cuda()
            r[f"att{key}"] = attention(m, x, k, (h // ps, w // ps)).tolist()
            r[f"z{key}_check"] = float(m(x)[0, k])
    print(key, cfg["backbone"], (h, w), "max RSS so far %.0f MB" % (__import__("resource").getrusage(0).ru_maxrss / 1024), flush=True)
    del m, ck; gc.collect(); torch.cuda.empty_cache()
for r in res:
    Image.open(r["path"]).convert("RGB").resize((96, 192), Image.BICUBIC).save(OUT / f"fig1_{r['kind']}.png")
    print(r["kind"], r["attr"], "label", r["label"], "zD %.2f (%.2f) zC %.2f (%.2f) tau*b %.2f" % (r["zD"], r["zD_check"], r["zC"], r["zC_check"], TAU * r["b"]))
json.dump({"tau": TAU, "cases": res}, open(OUT / "fig1_cases.json", "w"))
