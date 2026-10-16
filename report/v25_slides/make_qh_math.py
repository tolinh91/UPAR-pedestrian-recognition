"""Numbers for the two query-head 'math' slides, from the real v25 H+ member on the queryhow example image (row 30465).
Runs on the training machine, CPU only: python make_qh_math.py V25_ZIP OUT_JSON  (from the training par/ directory)

The query head is recomputed by hand from its own weights, so each intermediate can be read and changed:
  kv = LayerNorm(CLS + 98 patch tokens);  h = LayerNorm(query);  per head (8 x 160 dims):
  score_i = <W_q h, W_k kv_i> / sqrt(160),  a = softmax(score),  out = sum_i a_i W_v kv_i;  then W_o, + query,
  + MLP, LayerNorm, logit = <feature, fc_weight[hat]> + fc_bias[hat].
Outputs: per-head raw scores and weights of the strongest hat head; head-mean weights per patch; the logit under the
real attention and under counterfactual attention (uniform over patches, top-4 patches only, lower body only); the hat
logit of the same query on a second image without a hat.
"""
import io
import json
import sys
import zipfile

import numpy as np
import torch

from upar.data import UPARDataset, attr_names, build_transforms
from upar.models import build_model

zpath, out = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(zpath)
cfg = json.loads(z.read("assets/config.json"))
mcfg = dict([m for m in cfg["models"] if m["file"] == "hp.pt"][0]["model"], num_attrs=40)
model = build_model(mcfg, pretrained=False)
model.load_state_dict({k: v.float() for k, v in torch.load(io.BytesIO(z.read("assets/hp.pt")), weights_only=True).items()})
model.eval(); torch.set_grad_enabled(False)
A = attr_names(); K = A.index("Accessory-Hat")
ROWS = [30465, 19023]                                   # hat example of the deck, and a second image (no hat)
ds = UPARDataset("val", build_transforms((224, 112), False, mean=mcfg["mean"], std=mcfg["std"]), rows=ROWS)
labels = {r: ds[i][1][K].item() for i, r in enumerate(ROWS)}
H = model.head; L = H.layers[0]; mha = L["attn"]; nh = mha.num_heads; dim = mha.embed_dim; hd = dim // nh
Wq, Wk, Wv = mha.in_proj_weight.split(dim); bq, bk, bv = mha.in_proj_bias.split(dim)


def head_forward(t, weights=None):
    """t: (1, tokens, dim) backbone tokens. weights: optional (99,) attention to use in every head instead of softmax."""
    tt = torch.cat([t[:, :1], t[:, model.n_prefix:]], 1)[0]           # CLS + 98 patches
    kv = H.kv_norm(tt)
    q0 = H.queries[K]
    h = L["q_norm"](q0)
    qh = (h @ Wq.T + bq).view(nh, hd); kh = (kv @ Wk.T + bk).view(-1, nh, hd).transpose(0, 1)
    vh = (kv @ Wv.T + bv).view(-1, nh, hd).transpose(0, 1)             # (nh, 99, hd)
    score = (kh @ qh[:, :, None]).squeeze(-1) / hd ** 0.5              # (nh, 99)
    a = torch.softmax(score, -1) if weights is None else weights[None].expand(nh, -1)
    o = (a[:, :, None] * vh).sum(1).reshape(dim)
    q1 = q0 + mha.out_proj(o)
    q2 = q1 + L["mlp"](L["mlp_norm"](q1))
    f = H.out_norm(q2)
    logit = float(f @ H.fc_weight[K] + H.fc_bias[K])
    return logit, score, a, f


res = {"labels": labels}
for i, r in enumerate(ROWS):
    t = model.tokens(ds[i][0][None])
    logit, score, a, f = head_forward(t)
    full = float(model.head(t, model.n_prefix)[0, K])                 # the module's own forward, as a check
    res[str(r)] = {"logit": logit, "logit_module": full}
    if i == 0:
        am = a.mean(0)                                                 # head-mean attention over CLS + patches
        patch = am[1:].numpy(); res["cls_share"] = float(am[0])
        res["patch_weight_mean"] = patch.tolist()                      # 98 = 14 rows x 7 cols
        hstar = int(torch.argmax(a[:, 1:].max(1).values))             # the most concentrated head
        res["head"] = hstar
        res["head_score"] = score[hstar, 1:].tolist(); res["head_weight"] = a[hstar, 1:].tolist()
        res["head_cls"] = {"score": float(score[hstar, 0]), "weight": float(a[hstar, 0])}
        res["per_head"] = [{"patch": int(torch.argmax(a[h, 1:])), "weight": float(a[h, 1:].max()), "cls": float(a[h, 0]),
                             "gap": float(torch.topk(score[h, 1:], 2).values.diff().abs())} for h in range(nh)]
        top = np.argsort(-patch)[:4]
        res["top4"] = [int(j) for j in top]
        u = torch.zeros(99); u[1:] = 1 / 98
        res["logit_uniform"] = head_forward(t, u)[0]
        w4 = torch.zeros(99); w4[1 + torch.tensor(top)] = torch.tensor(patch[top]) / float(patch[top].sum())
        res["logit_top4"] = head_forward(t, w4)[0]
        lo = torch.zeros(99); lo[1 + 7 * 7:] = 1 / (98 - 49)           # rows 7-13 = lower body
        res["logit_lower"] = head_forward(t, lo)[0]
        res["feature_norm"] = float(f.norm()); res["fc_norm"] = float(H.fc_weight[K].norm()); res["fc_bias"] = float(H.fc_bias[K])
        res["cos_feature_fc"] = float(torch.nn.functional.cosine_similarity(f, H.fc_weight[K], dim=0))
    print(r, "label hat", labels[r], "logit", round(logit, 3), "module", round(full, 3))
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items() if k.startswith(("logit", "head", "top4", "cls", "cos", "feature", "fc"))
       and k not in ("head_score", "head_weight")})
json.dump(res, open(out, "w"))
