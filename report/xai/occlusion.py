"""Occlusion tests on an unseen domain: which body region does each attribute depend on?

Model: runs/ft_PETA_dinov3_q (DINOv3 ViT-B 256x128 + query head, trained without PETA), all 6,963 PETA val images.
1. Region masking: the crop is split into three horizontal bands (head 0-20%, upper body 20-55%, lower body 55-100%
   of the height). Each band is either masked (filled with the mean colour = 0 after normalisation) or kept alone
   (the other two masked). Per attribute we report the held-out AUC under each condition.
2. Sliding occlusion: for the 12 cases in xai_cases.json, a 32x32 mean-colour square slides with stride 16 and the
   drop of the attribute's logit is recorded (positive = the square hid evidence for the attribute).
Writes occlusion.json and the figures ../paper_en/figs/occlusion_{auc,maps}.pdf and ../paper/figs/occlusion_{auc,maps}.png.

Run from report/xai: ../../../Upar-Challenge-Track-2/.venv/bin/python occlusion.py
"""
import json, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "par"))
from upar.data import DATA, attr_names, build_transforms, load_split  # noqa: E402
from upar.models import build_model  # noqa: E402

RUN = ROOT / "runs" / "ft_PETA_dinov3_q"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(RUN / "last.pt", map_location="cpu", weights_only=False)
model = build_model(ck["model_cfg"], pretrained=False); model.load_state_dict(ck["model"]); model.eval().to(DEV)
H, W = ck["model_cfg"]["size"]
tf = build_transforms((H, W), train=False)
A = attr_names(); va = load_split("val")
idx = np.load(RUN / "held_val_logits_last.npz")["idx"]
paths = va.image.values[idx]; y = va[A].values[idx].astype(float)
BANDS = {"head": (0, .20), "upper": (.20, .55), "lower": (.55, 1.0)}


class DS(Dataset):
    def __len__(self):
        return len(paths)

    def __getitem__(self, i):
        return tf(Image.open(DATA / paths[i]).convert("RGB"))


def masked(x, keep=None, drop=None):
    x = x.clone()
    for name, (a, b) in BANDS.items():
        r0, r1 = int(a * H), int(b * H)
        if (drop is not None and name == drop) or (keep is not None and name != keep):
            x[:, :, r0:r1] = 0
    return x


CONDS = {"original": {}, **{f"mask {b}": {"drop": b} for b in BANDS}, **{f"only {b}": {"keep": b} for b in BANDS}}
logits = {c: [] for c in CONDS}
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
    for x in DataLoader(DS(), 64, num_workers=6):
        x = x.to(DEV)
        for c, kw in CONDS.items():
            logits[c].append(model(masked(x, **kw)).float().cpu())
logits = {c: torch.cat(v).numpy() for c, v in logits.items()}


def auc(s, t):
    pos, neg = s[t == 1], s[t == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = np.argsort(np.argsort(np.r_[pos, neg])) + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


ok = [k for k in range(len(A)) if 20 <= np.nansum(y[:, k]) <= len(y) - 20]  # attributes with >= 20 positives
res = {c: {A[k]: float(auc(logits[c][:, k], y[:, k])) for k in ok} for c in CONDS}
held = np.load(RUN / "held_val_logits_last.npz")["logits"]
print("sanity: mean AUC original", np.mean(list(res["original"].values())),
      "stored logits", np.mean([auc(held[:, k], y[:, k]) for k in ok]), flush=True)

GROUPS = {"hat": ["Accessory-Hat"], "glasses": ["Accessory-Glasses-Normal", "Accessory-Glasses-Sun"],
          "hair length": ["Hair-Length-Short", "Hair-Length-Long", "Hair-Length-Bald"],
          "age": ["Age-Young", "Age-Adult", "Age-Old"], "gender": ["Gender-Female"],
          "sleeve length": ["UpperBody-Length-Short"], "upper colour": [a for a in A if a.startswith("UpperBody-Color")],
          "backpack / bag": ["Accessory-Backpack", "Accessory-Bag"],
          "lower type": ["LowerBody-Type-Trousers&Shorts", "LowerBody-Type-Skirt&Dress"],
          "lower length": ["LowerBody-Length-Short"], "lower colour": [a for a in A if a.startswith("LowerBody-Color")]}
table = {g: {c: float(np.nanmean([res[c][a] for a in names if a in res[c]])) for c in CONDS} for g, names in GROUPS.items()}
for g, row in table.items():
    print(f"{g:15s} " + " ".join(f"{c}={v:.3f}" for c, v in row.items()), flush=True)

# sliding occlusion on the selected cases
cases = json.load(open(HERE / "xai_cases.json"))
S, STEP = 32, 16
ys_, xs_ = range(0, H - S + 1, STEP), range(0, W - S + 1, STEP)
maps = {}
for tag in ("good", "bad"):
    for c in cases[tag]:
        x = tf(Image.open(DATA / c["file"]).convert("RGB"))[None].to(DEV)
        batch = [x[0]]
        for r in ys_:
            for q in xs_:
                z = x[0].clone(); z[:, r:r + S, q:q + S] = 0; batch.append(z)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
            lg = model(torch.stack(batch)).float()[:, c["k"]].cpu().numpy()
        heat = np.zeros((H, W)); cnt = np.zeros((H, W))
        for j, (r, q) in enumerate((r, q) for r in ys_ for q in xs_):
            heat[r:r + S, q:q + S] += lg[0] - lg[j + 1]; cnt[r:r + S, q:q + S] += 1
        maps[f"{tag}:{c['file']}:{c['name']}"] = (heat / np.maximum(cnt, 1)).tolist()
json.dump({"bands": BANDS, "auc": res, "groups": table, "maps": maps}, open(HERE / "occlusion.json", "w"))

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

LAB = {"en": dict(groups={g: g for g in GROUPS}, cols=["mask head", "mask upper", "mask lower", "only head", "only upper", "only lower"],
                  cbar="AUC change vs. full image", good="correct", bad="errors"),
       "vi": dict(groups={"hat": "mũ", "glasses": "kính", "hair length": "độ dài tóc", "age": "tuổi", "gender": "giới tính",
                          "sleeve length": "áo ngắn tay", "upper colour": "màu áo", "backpack / bag": "ba lô / túi",
                          "lower type": "kiểu quần / váy", "lower length": "quần ngắn", "lower colour": "màu quần"},
                  cols=["che đầu", "che thân trên", "che thân dưới", "chỉ giữ đầu", "chỉ giữ thân trên", "chỉ giữ thân dưới"],
                  cbar="AUC thay đổi so với ảnh đủ", good="đúng", bad="sai")}
NAMES_ = {"en": {"Accessory-Hat": "hat", "Accessory-Backpack": "backpack", "LowerBody-Type-Skirt&Dress": "skirt/dress",
                 "Gender-Female": "female", "UpperBody-Color-Red": "red top", "Accessory-Glasses-Normal": "glasses"},
          "vi": {"Accessory-Hat": "mũ", "Accessory-Backpack": "ba lô", "LowerBody-Type-Skirt&Dress": "váy",
                 "Gender-Female": "nữ", "UpperBody-Color-Red": "áo đỏ", "Accessory-Glasses-Normal": "kính"}}
cond_keys = [c for c in CONDS if c != "original"]


def fig_auc(lang, out):
    L = LAB[lang]; g = list(GROUPS)
    M = np.array([[table[k][c] - table[k]["original"] for c in cond_keys] for k in g])
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    m = np.nanmax(np.abs(M))
    im = ax.imshow(M, cmap="RdBu", norm=TwoSlopeNorm(0, -m, max(m * .2, 1e-3)), aspect="auto")
    ax.set_xticks(range(len(cond_keys)), L["cols"], rotation=25, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(g)), [f"{L['groups'][k]} ({table[k]['original']:.2f})" for k in g], fontsize=7.5)
    for i in range(len(g)):
        for j in range(len(cond_keys)):
            ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=6.6, color="white" if abs(M[i, j]) > .55 * m else "black")
    ax.axvline(2.5, color="white", lw=2)
    cb = fig.colorbar(im, ax=ax, shrink=.8); cb.set_label(L["cbar"], fontsize=7); cb.ax.tick_params(labelsize=6.5)
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_maps(lang, out):
    L = LAB[lang]; n = max(len(cases["good"]), len(cases["bad"]))
    fig, axs = plt.subplots(2, n, figsize=(1.05 * n, 2 * 2.25), gridspec_kw={"hspace": .35, "wspace": .06})
    for b_i, tag in enumerate(("good", "bad")):
        for c_i in range(n):
            ax = axs[b_i, c_i]; ax.axis("off")
            if c_i >= len(cases[tag]):
                continue
            c = cases[tag][c_i]
            im = np.asarray(Image.open(DATA / c["file"]).convert("RGB").resize((W, H)))
            hm = np.array(maps[f"{tag}:{c['file']}:{c['name']}"]); m = np.abs(hm).max() or 1
            ax.imshow(im); ax.imshow(hm, cmap="RdBu_r", norm=TwoSlopeNorm(0, -m, m), alpha=.6)
            ax.set_title(f"{NAMES_[lang][c['name']]}\np={c['prob']:.2f}", fontsize=6.4,
                         color="#1a7f4b" if tag == "good" else "#b42318")
        axs[b_i, 0].text(-0.1, .5, L[tag], transform=axs[b_i, 0].transAxes, rotation=90, va="center", ha="right",
                         fontsize=7.5, fontweight="bold", color="#1a7f4b" if tag == "good" else "#b42318")
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)


fig_auc("en", ROOT / "report/paper_en/figs/occlusion_auc.pdf"); fig_auc("vi", ROOT / "report/paper/figs/occlusion_auc.png")
fig_maps("en", ROOT / "report/paper_en/figs/occlusion_maps.pdf"); fig_maps("vi", ROOT / "report/paper/figs/occlusion_maps.png")
print("saved")
