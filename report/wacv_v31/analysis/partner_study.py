"""Why CLIP next to DINOv3: every partner encoder X tested next to DINOv3 ViT-H+/16 (PETA fold, separate fine-tuning).

../../../../Upar-Challenge-Track-2/.venv/bin/python partner_study.py
All runs: weighted BCE (no logit adjustment), query head with 1 layer, 3 epochs, 8 frozen blocks, layer decay 0.75,
PETA held out, seed 1 (remote_results/kdhp/runs, recipe in kdhp/kdhp.sh). Decision logit > 0 (plain threshold 0.5).
Pair = mean of the two logits. Error overlap with DINOv3 = P(X errs | DINOv3 errs) over (image, labelled attribute) cells.
References: a second CLIP-L seed (C_s2) and the two DINOv3 / CLIP runs of Tab. two (remote_results/rare/runs).
Writes partner_study.json next to this file.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/media/tan/F/RWS2027/Upar-Challenge-Track-2/par")
from upar.data import attr_names, load_split  # noqa: E402
from upar.metrics import challenge_metrics  # noqa: E402

K = Path("/media/tan/F/RWS2027/remote_results/kdhp/runs")
RARE = Path("/media/tan/F/RWS2027/remote_results/rare/runs")
A = attr_names(); va = load_split("val"); GT = va[A].values; DS = va.ds.values
CAND = {"CLIP-L/14": ("P_clipL224", "224x112"), "AIMv2-L/14": ("Q_aimv2L224", "224x112"),
        "EVA02-CLIP-L/14": ("Q_eva02L224", "224x112"), "SigLIP2-So400m/14": ("Q_sig2So14", "224x112"),
        "MetaCLIP2-H/14": ("P_mc2H168", "168x84"), "PE-Core-L/14": ("PEc_L", "224x112"),
        "SigLIP2-So400m/16": ("Q_sig2So16", "256x128"), "MobileCLIP2-L/14": ("Q_mclip2L224", "224x112"),
        "SigLIP2-L/16-384": ("Q_sig2L384", "256x128"), "SigLIP2-L/16-256": ("Q_sig2L256", "256x128"),
        "DFN-CLIP-L/14": ("Q_dfnL224", "224x112"), "PE-Spatial-L/14": ("PEs_L", "224x112")}


def load(path):
    d = np.load(path / "held_val_logits_last.npz")
    o = np.argsort(d["idx"])
    return d["idx"][o], d["logits"][o].astype(np.float64)


def score(idx, z):
    return 100 * challenge_metrics(GT[idx], (z > 0).astype(float), DS[idx], thr=0.5)[0]["Challenge_Avg"]


idx, zd = load(K / "P_dinoHp224")
assert set(DS[idx]) == {"PETA"}
y = GT[idx]; valid = y >= 0
ed = ((zd > 0) != (y == 1)) & valid


def overlap(z):
    e = ((z > 0) != (y == 1)) & valid
    return 100 * e[ed].mean(), 100 * ed[e].mean()


out = {"DINOv3 alone": score(idx, zd), "partners": {}}
print(f"DINOv3 ViT-H+/16 alone {out['DINOv3 alone']:.2f}   error rate {100 * ed[valid].mean():.2f}%")
print(f"{'X':20s} {'input':8s} {'X alone':>8s} {'D + X':>7s} {'gain':>6s} {'P(X err|D err)':>15s} {'P(D err|X err)':>15s}")
for name, (run, size) in CAND.items():
    i, z = load(K / run); assert (i == idx).all()
    a, p = score(idx, z), score(idx, (zd + z) / 2)
    o1, o2 = overlap(z)
    out["partners"][name] = {"run": run, "input": size, "alone": a, "pair": p, "overlap_x_given_d": o1,
                             "overlap_d_given_x": o2}
    print(f"{name:20s} {size:8s} {a:8.2f} {p:7.2f} {p - max(a, out['DINOv3 alone']):6.2f} {o1:15.1f} {o2:15.1f}")
r = np.corrcoef([v["overlap_x_given_d"] for v in out["partners"].values()],
                [v["pair"] for v in out["partners"].values()])[0, 1]
print(f"corr(overlap with DINOv3, pair score) over the {len(CAND)} partners: {r:.2f}")
out["corr_overlap_pair"] = r

# same-encoder references on the same images
_, zc = load(K / "P_clipL224"); _, zc2 = load(K / "C_s2")
o1, o2 = overlap(zc2)
ec = ((zc > 0) != (y == 1)) & valid; ec2 = ((zc2 > 0) != (y == 1)) & valid
out["clip_seed2"] = {"alone": score(idx, zc2), "pair_with_clip": score(idx, (zc + zc2) / 2),
                     "overlap_clip2_given_clip1": 100 * ec2[ec].mean(), "pair_with_dinov3": score(idx, (zd + zc2) / 2)}
print("CLIP-L seed 2: alone {alone:.2f}, CLIP+CLIP {pair_with_clip:.2f}, DINOv3+CLIP-s2 {pair_with_dinov3:.2f}, "
      "P(CLIP s2 err | CLIP s1 err) {overlap_clip2_given_clip1:.1f}%".format(**out["clip_seed2"]))
for n in ("HB_s1", "HP_s1", "B_s1", "B_s2"):
    i, z = load(RARE / n)
    same = np.array_equal(i, idx) and np.allclose(z, zd if n.startswith("H") else zc, atol=1e-3)
    e = ((z > 0) != (y == 1)) & valid
    print(f"rare/{n}: alone {score(i, z):.2f}  identical to the kdhp {'DINOv3' if n.startswith('H') else 'CLIP'} run: "
          f"{same}  P(err | DINOv3 err) {100 * e[ed].mean():.1f}%")

# all 13 encoders (DINOv3 + the 12 partners, CLIP run P_clipL224): symmetric overlap = mean of the two conditionals
Z = {"DINOv3": zd} | {n: load(K / r)[1] for n, (r, _) in CAND.items()}
E = {n: ((z > 0) != (y == 1)) & valid for n, z in Z.items()}
names = list(Z)
sym = lambda a, b: 50 * (E[b][E[a]].mean() + E[a][E[b]].mean())
with_d = [sym("DINOv3", b) for b in names[1:]]
pairs = [(a, b) for i, a in enumerate(names[1:], 1) for b in names[i + 1:]]
without = {(a, b): (sym(a, b), score(idx, (Z[a] + Z[b]) / 2)) for a, b in pairs}
o = np.array([v[0] for v in without.values()]); s_ = np.array([v[1] for v in without.values()])
print(f"symmetric overlap: DINOv3 with each partner {min(with_d):.1f}-{max(with_d):.1f}% (mean {np.mean(with_d):.1f}); "
      f"between two partners {o.min():.1f}-{o.max():.1f}% (mean {o.mean():.1f}, median {np.median(o):.1f}, "
      f"{(o > max(with_d)).sum()} of {len(o)} above the DINOv3 maximum)")
top = max(without, key=lambda k: without[k][1])
print(f"pairs without DINOv3: highest {top} {without[top][1]:.2f}; pairs with DINOv3: "
      f"{min(v['pair'] for v in out['partners'].values()):.2f}-{max(v['pair'] for v in out['partners'].values()):.2f}; "
      f"{sum(v['pair'] > without[top][1] for v in out['partners'].values())} of 12 DINOv3 pairs above it")
out["sym_overlap_with_dinov3"] = with_d
out["pairs_without_dinov3"] = {f"{a} + {b}": {"sym_overlap": v[0], "pair": v[1]} for (a, b), v in without.items()}
json.dump(out, open(Path(__file__).with_name("partner_study.json"), "w"), indent=1)
