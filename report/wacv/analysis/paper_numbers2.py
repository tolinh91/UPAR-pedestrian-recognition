"""Extra numbers for the 'fit less to the source' argument (PETA fold). -> numbers2.json
For each source-fitted alternative: what it gains in-domain (where it is fitted) and what it gains on held-out PETA."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/media/tan/F/RWS2027/Upar-Challenge-Track-2/par")
sys.path.insert(0, str(Path(__file__).parent))
from calibrate import fit_offsets
import paper_numbers as P   # reuses loaders (runs its own prints once)

K = P.K; H, C = K / "P_dinoHp224", K / "P_clipL224"
off = P.prior_off("PETA"); va, A = P.va, P.A
O = {}
# ranking drop: AUC in-domain vs held-out
for name, ds in (("H+", [H]), ("CLIP-L", [C]), ("pair", [H, C])):
    si, sz = P.ens(ds, "seen"); hi, hz = P.ens(ds)
    O.setdefault("auc", {})[name] = {"indomain": P.auc(si, sz), "peta": P.auc(hi, hz)}
# weights chosen in-domain
si, hs = P.load(H, "seen"); _, cs = P.load(C, "seen")
W = {}
for w in np.round(np.arange(0, 1.01, 0.1), 1):
    W[float(w)] = P.met(si, w * hs + (1 - w) * cs, "PETA")["Challenge_Avg"]
O["weights_indomain"] = W
# per-attribute offsets fitted on the in-domain validation data vs the tau rule
for name, ds in (("H+", [H]), ("CLIP-L", [C]), ("pair", [H, C])):
    si, sz = P.ens(ds, "seen"); hi, hz = P.ens(ds)
    gs, grs = va[A].values[si], va.ds.values[si]; gh, grh = va[A].values[hi], va.ds.values[hi]
    o, _ = fit_offsets(sz, gs, grs)
    r = {}
    for tag, zs_, zh_ in (("raw", sz, hz), ("tau", sz + 0.42 * off, hz + 0.42 * off), ("offsets", sz + o, hz + o), ("offsets_x0.6", sz + 0.6 * o, hz + 0.6 * o)):
        r[tag] = {"indomain": P.challenge_metrics(gs, P.sig(zs_), grs)[0]["Challenge_Avg"], "peta": P.challenge_metrics(gh, P.sig(zh_), grh)[0]["Challenge_Avg"]}
    O.setdefault("rules", {})[name] = r
    print(name, {k: (round(v["indomain"], 4), round(v["peta"], 4)) for k, v in r.items()}, flush=True)
print("auc", O["auc"]); print("weights in-domain", {k: round(v, 4) for k, v in W.items()})
json.dump(O, open(Path(__file__).with_name("numbers2.json"), "w"), indent=1)
