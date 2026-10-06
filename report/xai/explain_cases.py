"""Good and bad cases on an unseen domain, with the query head's attention and Integrated Gradients.

Model: runs/ft_PETA_dinov3_q (DINOv3 ViT-B 256x128 + one-layer query head, trained on PA100k + Market1501 only),
evaluated on PETA val images it has never seen. Decision rule: tau = 0.42 with priors of its own training data.
For each attribute in CASES we take the most confident correct positive ("good") and the worst error ("bad": the
missed positive with the lowest probability, or the false positive with the highest one), each from a different image.
Writes figs for both papers: ../paper_en/figs/xai_cases.pdf (English) and ../paper/figs/xai_cases.png (Vietnamese),
plus xai_cases.json with the selected cases.

Run from report/xai: ../../../Upar-Challenge-Track-2/.venv/bin/python explain_cases.py
"""
import json, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "par"))
from upar.data import DATA, attr_names, build_transforms, load_split  # noqa: E402
from upar.models import build_model  # noqa: E402

RUN = ROOT / "runs" / "ft_PETA_dinov3_q"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(RUN / "last.pt", map_location="cpu", weights_only=False)
model = build_model(ck["model_cfg"], pretrained=False); model.load_state_dict(ck["model"]); model.eval().to(DEV)
H, W = ck["model_cfg"]["size"]; GH, GW = H // 16, W // 16
tf = build_transforms((H, W), train=False)
A = attr_names(); va = load_split("val"); tr = load_split("train")

# held-out probabilities with the tau rule (priors of the two training datasets)
z = np.load(RUN / "held_val_logits_last.npz"); idx = z["idx"]
seen = tr[tr.ds != "PETA"][A].values.astype(float)
p = np.clip(np.nanmean(seen, 0), 1e-3, 1 - 1e-3)
b = np.log((1 - p) / p) - (1 - 2 * p)
prob = 1 / (1 + np.exp(-(z["logits"] + 0.42 * b)))
y = va[A].values[idx].astype(float)
paths = va.image.values[idx]

CASES = ["Accessory-Hat", "Accessory-Backpack", "LowerBody-Type-Skirt&Dress", "Gender-Female",
         "UpperBody-Color-Red", "Accessory-Glasses-Normal"]
used, good, bad = set(), [], []
for name in CASES:
    k = A.index(name); yk, pk = y[:, k], prob[:, k]
    for i in np.argsort(-pk):  # most confident true positive
        if yk[i] == 1 and pk[i] > .5 and i not in used:
            good.append((i, k, "TP")); used.add(i); break
    fn = [(pk[i], i, "FN") for i in np.argsort(pk) if yk[i] == 1 and pk[i] < .5][:1]
    fp = [(1 - pk[i], i, "FP") for i in np.argsort(-pk) if yk[i] == 0 and pk[i] > .5][:1]
    for _, i, kind in sorted(fn + fp):  # the more confident error first
        if i not in used:
            bad.append((i, k, kind)); used.add(i); break

captured = {}
model.head.register_forward_hook(lambda m, inp, out: captured.update(tokens=inp[0], n_prefix=inp[1]))


def attention(tokens, n_prefix):
    head = model.head; t = torch.cat([tokens[:, :1], tokens[:, n_prefix:]], 1); kv = head.kv_norm(t)
    layer = head.layers[0]; mha = layer["attn"]
    q = layer["q_norm"](head.queries)[None]
    Wq, Wk, _ = mha.in_proj_weight.chunk(3); bq, bk, _ = mha.in_proj_bias.chunk(3)
    Q = q @ Wq.T + bq; K = kv @ Wk.T + bk; nh = mha.num_heads; d = Q.shape[-1] // nh
    Q = Q.view(1, -1, nh, d).transpose(1, 2); K = K.view(1, -1, nh, d).transpose(1, 2)
    a = (Q @ K.transpose(-1, -2) / d ** 0.5).softmax(-1).mean(1)[0]
    pt = a[:, 1:]
    return (pt / pt.sum(-1, keepdim=True)).cpu().numpy(), a[:, 0].cpu().numpy()


def integrated_gradients(x, k, steps=24):
    alphas = torch.linspace(1 / steps, 1, steps, device=DEV).view(-1, 1, 1, 1)
    xs = (alphas * x).requires_grad_(True)  # baseline 0 in normalised space = mean-colour image
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
        logit = model(xs).float()[:, k].sum()
    g, = torch.autograd.grad(logit, xs)
    ig = (x * g.mean(0, keepdim=True)).sum(1)[0]
    return torch.nn.functional.avg_pool2d(ig[None, None], 8)[0, 0].float().cpu().numpy()


def explain(i, k):
    img = Image.open(DATA / paths[i]).convert("RGB")
    x = tf(img)[None].to(DEV)
    with torch.no_grad():
        model(x)
        att, cls = attention(captured["tokens"], captured["n_prefix"])
    return np.asarray(img.resize((W, H), Image.BILINEAR)), att[k].reshape(GH, GW), cls[k], integrated_gradients(x, k)


rows = {"good": [], "bad": []}
for tag, lst in (("good", good), ("bad", bad)):
    for i, k, kind in lst:
        im, att, cls, ig = explain(i, k)
        rows[tag].append(dict(i=int(i), k=k, name=A[k], kind=kind, label=int(y[i, k]), prob=float(prob[i, k]),
                              file=str(paths[i]), cls=float(cls), im=im, att=att, ig=ig))
        print(tag, kind, A[k], paths[i], f"p={prob[i, k]:.2f} y={int(y[i, k])} cls={cls:.2f}", flush=True)
json.dump({t: [{k: v for k, v in r.items() if k not in ("im", "att", "ig")} for r in rs] for t, rs in rows.items()},
          open(HERE / "xai_cases.json", "w"), indent=1)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

EN = {"Accessory-Hat": "hat", "Accessory-Backpack": "backpack", "LowerBody-Type-Skirt&Dress": "skirt/dress",
      "Gender-Female": "female", "UpperBody-Color-Red": "red top", "Accessory-Glasses-Normal": "glasses"}
VI = {"Accessory-Hat": "mũ", "Accessory-Backpack": "ba lô", "LowerBody-Type-Skirt&Dress": "váy",
      "Gender-Female": "nữ", "UpperBody-Color-Red": "áo đỏ", "Accessory-Glasses-Normal": "kính"}
TXT = {"en": dict(names=EN, good="correct (unseen PETA)", bad="errors (unseen PETA)", rows=["image", "attention", "IG"],
                  kind={"TP": "hit", "FN": "missed", "FP": "false alarm"}, lab="label"),
       "vi": dict(names=VI, good="đúng (PETA chưa thấy)", bad="sai (PETA chưa thấy)", rows=["ảnh", "attention", "IG"],
                  kind={"TP": "đúng", "FN": "bỏ sót", "FP": "báo nhầm"}, lab="nhãn")}


def figure(lang, out):
    T = TXT[lang]; n = max(len(rows["good"]), len(rows["bad"]))
    fig, axs = plt.subplots(6, n, figsize=(1.05 * n, 6 * 2.02), gridspec_kw={"hspace": .04, "wspace": .06})
    for b_i, tag in enumerate(("good", "bad")):
        for c in range(n):
            for r in range(3):
                axs[3 * b_i + r, c].axis("off")
            if c >= len(rows[tag]):
                continue
            R = rows[tag][c]
            ax0, ax1, ax2 = axs[3 * b_i, c], axs[3 * b_i + 1, c], axs[3 * b_i + 2, c]
            ax0.imshow(R["im"])
            ax0.set_title(f"{T['names'][R['name']]}\n{T['kind'][R['kind']]}\np={R['prob']:.2f}, {T['lab']} {R['label']}",
                          fontsize=6.2, linespacing=1.1, color="#1a7f4b" if tag == "good" else "#b42318")
            ax1.imshow(R["im"]); ax1.imshow(np.kron(R["att"], np.ones((16, 16))), cmap="inferno", alpha=.6)
            ax2.imshow(R["im"]); m = np.abs(R["ig"]).max() or 1
            ax2.imshow(np.kron(R["ig"], np.ones((8, 8))), cmap="RdBu_r", norm=TwoSlopeNorm(0, -m, m), alpha=.65)
        axs[3 * b_i, 0].text(-0.15, 1.42, T[tag], transform=axs[3 * b_i, 0].transAxes, fontsize=8, fontweight="bold",
                             color="#1a7f4b" if tag == "good" else "#b42318", ha="left")
    for r in range(6):
        axs[r, 0].text(-0.12, .5, T["rows"][r % 3], transform=axs[r, 0].transAxes, rotation=90, va="center", ha="right", fontsize=6.8)
    for c in range(n):  # space for the block titles above rows 0 and 3
        for r in (3, 4, 5):
            pos = axs[r, c].get_position(); axs[r, c].set_position([pos.x0, pos.y0 - .055, pos.width, pos.height])
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)


figure("en", ROOT / "report/paper_en/figs/xai_cases.pdf")
figure("vi", ROOT / "report/paper/figs/xai_cases.png")
print("saved")
