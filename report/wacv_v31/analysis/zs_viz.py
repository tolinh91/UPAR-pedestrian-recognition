"""Zero-shot check of UniQuery's CLIP text score (no training, run on the remote GPU).

For the val images of PETA and Market-1501 (the two held-out datasets of the paper) and the first 3000 PA100k val
images (the subset make_clip_text.py logged), with the exact text file used to train v31 (clip_l14_text.pt):
  e      = normalize(CLS of timm CLIP ViT-L/14 at 224x112 @ frozen CLIP projection), as in training
  s_sub  = 100 <e, pos_k - neutral>    UniQuery text score at the start of training (delta = 0), without b_k
  s_raw  = 100 <e, pos_k>              the same score without subtracting the sentence 'a person'
  s_neu  = 100 <e, neutral>            similarity with 'a person' alone
Writes per-image scores and labels (zs_scores.npz) and per-attribute AUC (zs_auc.json).
"""
import json

import numpy as np
import pandas as pd
import timm
import torch
import torchvision.transforms as T
from PIL import Image

R = "/root/zs"
D = f"{R}/data"
MEAN, STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)   # OpenAI CLIP
tf = T.Compose([T.Resize((224, 112)), T.ToTensor(), T.Normalize(MEAN, STD)])           # = build_transforms eval


def auc(y, s):
    """Same rank AUC as make_clip_text.auc."""
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def main():
    t = torch.load(f"{R}/clip_l14_text.pt", map_location="cpu")
    proj, pos, neu = t["proj"].float().cuda(), t["pos"].float().cuda(), t["neutral"].float().cuda()
    df = pd.read_csv(f"{D}/annotations/task1/val/gt.csv")
    df.columns = ["image"] + list(df.columns[1:])
    df["ds"] = df.image.str.split("/").str[0]
    A = list(df.columns[1:41])
    tm = timm.create_model("vit_large_patch14_clip_224.openai", pretrained=True, num_classes=0,
                           img_size=(224, 112)).cuda().eval()
    out, arrays = {"attributes": A, "prompts": list(t["prompts"])}, {}
    for name, n in (("PETA", None), ("Market1501", None), ("PA100k", 3000)):
        g = df[df.ds == name]
        g = g if n is None else g.head(n)
        E = []
        with torch.no_grad():
            for i in range(0, len(g), 256):
                x = torch.stack([tf(Image.open(f"{D}/{p}").convert("RGB")) for p in g.image.values[i:i + 256]]).cuda()
                E.append(torch.nn.functional.normalize(tm.forward_features(x)[:, 0].float() @ proj, dim=-1))
        E = torch.cat(E)
        s_sub = (100 * E @ (pos - neu[None]).T).cpu().numpy()
        s_raw = (100 * E @ pos.T).cpu().numpy()
        s_neu = (100 * E @ neu).cpu().numpy()
        y = g[A].values.astype(np.int8)
        res = {}
        for k, a in enumerate(A):
            if 0 < y[:, k].sum() < len(y):
                res[a] = {"pos": int(y[:, k].sum()), "auc_sub": auc(y[:, k], s_sub[:, k]),
                          "auc_raw": auc(y[:, k], s_raw[:, k]),
                          "corr_raw_neu": float(np.corrcoef(s_raw[:, k], s_neu)[0, 1])}
        m = min(3000, len(g))
        first = [auc(y[:m, k], s_sub[:m, k]) for k in range(len(A)) if 0 < y[:m, k].sum() < m]
        out[name] = {"n": int(len(g)), "attrs": res,
                     "mean_auc_sub": float(np.mean([r["auc_sub"] for r in res.values()])),
                     "mean_auc_raw": float(np.mean([r["auc_raw"] for r in res.values()])),
                     "std_neu": float(s_neu.std()), "mean_std_raw": float(s_raw.std(0).mean()),
                     "mean_auc_sub_first3000": float(np.mean(first)), "n_attr_first3000": len(first)}
        arrays.update({f"{name}_s_sub": s_sub.astype(np.float32), f"{name}_s_raw": s_raw.astype(np.float32),
                       f"{name}_s_neu": s_neu.astype(np.float32), f"{name}_y": y,
                       f"{name}_image": np.array(g.image.tolist(), dtype="U")})
        print(f"{name}: n={len(g)} attrs={len(res)} AUC sub {out[name]['mean_auc_sub']:.4f} "
              f"raw {out[name]['mean_auc_raw']:.4f} | first-3000 sub {out[name]['mean_auc_sub_first3000']:.4f} "
              f"over {len(first)} | std(s_neu) {out[name]['std_neu']:.3f} mean std(s_raw) {out[name]['mean_std_raw']:.3f}",
              flush=True)
    np.savez_compressed(f"{R}/zs_scores.npz", **arrays)
    json.dump(out, open(f"{R}/zs_auc.json", "w"), indent=1)


if __name__ == "__main__":
    main()
