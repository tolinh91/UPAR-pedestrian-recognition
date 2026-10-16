"""Real values for the query-head code walkthrough slides (v25 H+ member, hat query, example image row 30465).
CPU only, from the training par/ directory:  python qh_code_numbers.py V25_ZIP OUT_JSON

Follows heads.py QueryHead._logits line by line (CLS + patches -> kv_norm -> q_norm -> attention -> residual -> MLP ->
out_norm -> classifier) and keeps, for attention head 2 (index 1): the first 8 of the 160 query / key numbers of a few
patches, their scores and softmax weights; and for the classifier: the first 8 of the 1280 feature and weight numbers,
their products and the full sum.
"""
import io
import json
import sys
import zipfile

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
A = attr_names(); K = A.index("Accessory-Hat"); HEAD = 1
ds = UPARDataset("val", build_transforms((224, 112), False, mean=mcfg["mean"], std=mcfg["std"]), rows=[30465])
x = ds[0][0][None]
H = model.head; L = H.layers[0]; mha = L["attn"]; nh = mha.num_heads; dim = mha.embed_dim; hd = dim // nh
Wq, Wk, Wv = mha.in_proj_weight.split(dim); bq, bk, bv = mha.in_proj_bias.split(dim)

tokens = model.tokens(x)                                            # (1, 1 + 4 + 98, 1280)
t = torch.cat([tokens[:, :1], tokens[:, model.n_prefix:]], 1)       # heads.py line 200
kv = H.kv_norm(t)[0]                                                # line 203, (99, 1280)
q0 = H.queries[K]                                                   # line 231, the hat query (1280,)
h = L["q_norm"](q0)                                                 # line 236
qh = (h @ Wq.T + bq).view(nh, hd)                                   # inside line 237
kh = (kv @ Wk.T + bk).view(-1, nh, hd).transpose(0, 1)
vh = (kv @ Wv.T + bv).view(-1, nh, hd).transpose(0, 1)
score = (kh @ qh[:, :, None]).squeeze(-1) / hd ** 0.5               # (8, 99)
a = torch.softmax(score, -1)
o = (a[:, :, None] * vh).sum(1).reshape(dim)
attn_out = mha.out_proj(o)
q1 = q0 + attn_out                                                  # line 237
q2 = q1 + L["mlp"](L["mlp_norm"](q1))                               # line 238
f = H.out_norm(q2)                                                  # line 239
logit = f @ H.fc_weight[K] + H.fc_bias[K]                           # line 241
full = model.head(tokens, model.n_prefix)[0, K]

s1 = score[HEAD, 1:]; order = torch.argsort(-s1)
pick = {"top": int(order[0]), "second": int(order[1]), "shoulder": 2 * 7 + 5, "shoe": 13 * 7 + 3}  # patch index (row*7+col)
res = {"shapes": {"tokens": list(tokens.shape), "t": list(t.shape), "kv": list(kv.shape), "queries": list(H.queries.shape),
                  "qh": list(qh.shape), "kh": list(kh.shape), "score": list(score.shape), "f": list(f.shape)},
       "n_prefix": model.n_prefix, "head": HEAD + 1, "logit": float(logit), "logit_module": float(full),
       "bias": float(H.fc_bias[K]), "prob": float(torch.sigmoid(logit))}
res["q8"] = qh[HEAD, :8].tolist()
res["patches"] = {n: {"patch": j, "row": j // 7, "col": j % 7, "k8": kh[HEAD, 1 + j, :8].tolist(),
                      "score": float(score[HEAD, 1 + j]), "weight": float(a[HEAD, 1 + j]),
                      "dot8": float((qh[HEAD, :8] * kh[HEAD, 1 + j, :8]).sum() / hd ** 0.5)} for n, j in pick.items()}
res["cls"] = {"score": float(score[HEAD, 0]), "weight": float(a[HEAD, 0])}
res["f8"] = f[:8].tolist(); res["w8"] = H.fc_weight[K][:8].tolist()
prod = f * H.fc_weight[K]
res["prod8"] = prod[:8].tolist(); res["prod_sum"] = float(prod.sum())
res["prod_pos"] = float(prod[prod > 0].sum()); res["prod_neg"] = float(prod[prod < 0].sum())
res["q160"] = qh[HEAD].tolist()
for n, j in pick.items():
    res["patches"][n]["k160"] = kh[HEAD, 1 + j].tolist()
res["prod1280"] = prod.tolist()
res["norms"] = {"q0": float(q0.norm()), "attn_out": float(attn_out.norm()), "mlp_out": float((q2 - q1).norm()), "f": float(f.norm())}
json.dump(res, open(out, "w"), indent=1)
print({k: res[k] for k in ("shapes", "head", "logit", "logit_module", "bias", "prob", "prod_sum", "prod_pos", "prod_neg", "norms")})
print({n: (p["patch"], round(p["score"], 2), round(p["weight"], 4)) for n, p in res["patches"].items()}, res["cls"])
