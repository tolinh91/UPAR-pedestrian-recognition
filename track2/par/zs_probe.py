"""Zero-shot probes for the language anchors of the CLIP member (no training).

python zs_probe.py OUT_PT      (needs open_clip_torch; writes the new prompt set as a --text-init file to OUT_PT)
Compares, per val dataset, the zero-shot AUC of the text directions w_k = t(prompt_k) - t("a person") when scored on
  cls     the projected CLS token (the current query_text anchor)
  region  colour attributes scored on the projected mean of the upper-body (rows 2-7 of the 16x8 patch grid) or
          lower-body (rows 8-14) patch tokens; the other attributes on the CLS token
for the current prompts (make_clip_text.PROMPTS) and a richer prompt ensemble (NEW below). The prompt choice should be
made on PA100k (a training dataset of both folds), not on the held-out folds.
"""
import sys

import numpy as np
import open_clip
import timm
import torch

from make_clip_text import PROMPTS as OLD, TEMPLATES, auc
from upar.data import UPARDataset, attr_names, build_transforms, load_split

COLOURS = ["black", "blue", "brown", "green", "grey", "orange", "pink", "purple", "red", "white", "yellow"]


def up(c):
    alt = ["gray"] if c == "grey" else []
    return [f"a person wearing a {x} {g}" for x in [c] + alt for g in ("top", "shirt", "jacket")] + \
        [f"a person in {c} upper-body clothing"]


def low(c):
    alt = ["gray"] if c == "grey" else []
    return [f"a person wearing {x} {g}" for x in [c] + alt for g in ("trousers", "pants")] + \
        [f"a person wearing a {c} skirt", f"a person in {c} lower-body clothing"]


NEW = ([["a teenager", "a young person", "a young adult in their twenties", "a student"],
        ["a middle-aged adult", "an adult in their thirties", "an adult in their forties"],
        ["an elderly person", "an old person with grey hair", "a senior citizen"],
        ["a woman", "a female pedestrian", "a lady"],
        ["a person with short hair", "a man with short hair"],
        ["a person with long hair", "a woman with long hair"],
        ["a bald person", "a bald man"],
        ["a person wearing a short-sleeved shirt", "a person wearing a t-shirt", "a person in short sleeves"]]
       + [up(c) for c in COLOURS]
       + [["a person wearing a patterned top", "a person wearing a striped shirt", "a person wearing a plaid shirt",
           "a person wearing a multicoloured top"]]
       + [["a person wearing shorts", "a person wearing a short skirt", "a person with bare legs"]]
       + [low(c) for c in COLOURS]
       + [["a person wearing patterned trousers", "a person wearing a patterned skirt", "a person wearing multicoloured pants"]]
       + [["a person wearing trousers", "a person wearing jeans", "a person wearing pants"],
          ["a person wearing a skirt", "a person wearing a dress"],
          ["a person carrying a backpack", "a person with a backpack on their back"],
          ["a person carrying a handbag", "a person carrying a shoulder bag", "a person carrying a bag",
           "a person holding a plastic bag"],
          ["a person wearing eyeglasses", "a person wearing glasses", "a person wearing spectacles"],
          ["a person wearing sunglasses", "a person wearing dark sunglasses"],
          ["a person wearing a hat", "a person wearing a cap", "a person wearing a beanie"]])
NEUTRAL_NEW = ["a person", "a pedestrian"]
UPPER, LOWER = list(range(8, 20)), list(range(21, 33))


@torch.no_grad()
def encode(model, tok, phrase_lists):
    out = []
    for phrases in phrase_lists:
        e = model.encode_text(tok([t.format(p) for p in phrases for t in TEMPLATES])).float()
        out.append(torch.nn.functional.normalize(torch.nn.functional.normalize(e, dim=-1).mean(0), dim=0))
    return torch.stack(out)


def main(out):
    A = attr_names(); assert len(NEW) == len(A) == len(OLD)
    clip, _, pre = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    clip.eval(); tok = open_clip.get_tokenizer("ViT-L-14")
    proj = clip.visual.proj.detach().float()
    sets = {"old": encode(clip, tok, [[p] for p in OLD]) - encode(clip, tok, [["a person"]])[0],
            "new": encode(clip, tok, NEW) - encode(clip, tok, [NEUTRAL_NEW])[0]}
    torch.save({"proj": proj, "pos": encode(clip, tok, NEW), "neutral": encode(clip, tok, [NEUTRAL_NEW])[0],
                "prompts": NEW}, out)
    del clip
    tm = timm.create_model("vit_large_patch14_clip_224.openai", pretrained=True, num_classes=0,
                           img_size=(224, 112)).cuda().eval().to(torch.bfloat16)
    norm = dict(mean=pre.transforms[-1].mean, std=pre.transforms[-1].std)
    ds = UPARDataset("val", build_transforms((224, 112), False, **norm))
    va = load_split("val"); P = proj.cuda()
    for name in ("PA100k", "PETA", "Market1501"):
        rows = np.where(va.ds.values == name)[0][:5000]
        E = {"cls": [], "up": [], "lo": []}
        for i in range(0, len(rows), 32):
            x = torch.stack([ds[j][0] for j in rows[i:i + 32]]).cuda().to(torch.bfloat16)
            with torch.no_grad():
                t = tm.forward_features(x).float()
            g = t[:, 1:].reshape(len(x), 16, 8, -1)
            for k, v in (("cls", t[:, 0]), ("up", g[:, 2:8].mean((1, 2))), ("lo", g[:, 8:15].mean((1, 2)))):
                E[k].append(torch.nn.functional.normalize(v @ P, dim=-1).cpu())
        E = {k: torch.cat(v) for k, v in E.items()}
        y = va[A].values[rows]
        ok = [k for k in range(len(A)) if 0 < y[:, k].sum() < len(y)]
        line = []
        for sname, W in sets.items():
            s_cls = (E["cls"] @ W.T).numpy()
            s_reg = s_cls.copy()
            s_reg[:, UPPER] = (E["up"] @ W[UPPER].T).numpy(); s_reg[:, LOWER] = (E["lo"] @ W[LOWER].T).numpy()
            for rname, s in (("cls", s_cls), ("region", s_reg)):
                a = {k: auc(y[:, k], s[:, k]) for k in ok}
                col = [a[k] for k in ok if k in UPPER + LOWER]
                line.append(f"{sname}/{rname}: all {np.mean(list(a.values())):.4f} colours {np.mean(col):.4f}")
        print(f"{name:10s} " + " | ".join(line), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
