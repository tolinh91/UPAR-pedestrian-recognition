"""Full matrices of the query head for the matrix-walkthrough slides (v25 H+, example image row 30465, all 40 queries).
CPU only, from the training par/ directory:  python qh_mats.py V25_ZIP OUT_NPZ

Same steps as heads.py QueryHead._logits, written as matrix products:
  T  = CLS + 98 patch tokens (99 x 1280), KV = LayerNorm(T)
  Qn = LayerNorm(40 queries) (40 x 1280)
  per head h (8 x 160 columns): Qh = Qn Wq_h + b, Kh = KV Wk_h + b, Vh = KV Wv_h + b
  S_h = Qh Kh^T / sqrt(160) (40 x 99),  A_h = softmax over the 99 tokens,  O_h = A_h Vh (40 x 160)
  O = [O_1 .. O_8] (40 x 1280) -> out_proj -> + queries -> + MLP -> LayerNorm = F (40 x 1280)
  logits = rowsum(F * W) + b (40)
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
ds = UPARDataset("val", build_transforms((224, 112), False, mean=mcfg["mean"], std=mcfg["std"]), rows=[30465])
x, y = ds[0][0][None], ds[0][1]
H = model.head; L = H.layers[0]; mha = L["attn"]; nh = mha.num_heads; dim = mha.embed_dim; hd = dim // nh
Wq, Wk, Wv = mha.in_proj_weight.split(dim); bq, bk, bv = mha.in_proj_bias.split(dim)

tokens = model.tokens(x)
T = torch.cat([tokens[:, :1], tokens[:, model.n_prefix:]], 1)[0]        # (99, 1280)
KV = H.kv_norm(T)
Qn = L["q_norm"](H.queries)                                             # (40, 1280)
Qh = (Qn @ Wq.T + bq).view(40, nh, hd).transpose(0, 1)                  # (8, 40, 160)
Kh = (KV @ Wk.T + bk).view(-1, nh, hd).transpose(0, 1)                  # (8, 99, 160)
Vh = (KV @ Wv.T + bv).view(-1, nh, hd).transpose(0, 1)                  # (8, 99, 160)
S = Qh @ Kh.transpose(1, 2) / hd ** 0.5                                 # (8, 40, 99)
Aw = torch.softmax(S, -1)
O = (Aw @ Vh).transpose(0, 1).reshape(40, dim)                          # (40, 1280)
q1 = H.queries + mha.out_proj(O)
q2 = q1 + L["mlp"](L["mlp_norm"](q1))
F = H.out_norm(q2)                                                      # (40, 1280)
W = H.fc_weight
logits = (F * W).sum(-1) + H.fc_bias
ref = model.head(tokens, model.n_prefix)[0]
print("max |logits - module|", float((logits - ref).abs().max()), "hat", float(logits[attr_names().index("Accessory-Hat")]))
f16 = lambda t: t.numpy().astype(np.float16)  # noqa: E731
np.savez_compressed(out, T=f16(T), KV=f16(KV), Q=f16(H.queries), Qn=f16(Qn), Qh=f16(Qh), Kh=f16(Kh), Vh=f16(Vh),
                    S=f16(S), A=f16(Aw), O=f16(O), F=f16(F), W=f16(W), logits=logits.numpy(), bias=H.fc_bias.numpy(),
                    labels=y.numpy(), attrs=np.array(attr_names()))
print(out, {k: tuple(v.shape) for k, v in dict(T=T, Qh=Qh, S=S, O=O, F=F).items()})
