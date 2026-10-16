"""Every number of the WACV Track 1 paper, from saved held-out / in-domain logits (numpy only).
../../../../Upar-Challenge-Track-2/.venv/bin/python paper_numbers.py  -> numbers.json
PETA fold: models trained on Market1501 + PA100k train splits, scored on PETA val (unseen domain).
Market fold: trained on PA100k + PETA, scored on Market1501 val. In-domain: the PETA-fold models on Market1501 + PA100k val.
Decision rule: logits + tau * b_k, b_k = log((1-p)/p) - (1-2p), p = pooled prior of the fold's training labels."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/media/tan/F/RWS2027/Upar-Challenge-Track-2/par")
from calibrate import fit_offsets
from upar.data import attr_names, load_split
from upar.metrics import challenge_metrics

A = attr_names(); va = load_split("val"); tr = load_split("train")
K = Path("/media/tan/F/RWS2027/remote_results/kdhp/runs")
V = Path("/media/tan/F/RWS2027/Upar-Challenge-Track-2/runs/remote2")
R5 = Path("/media/tan/F/RWS2027/remote_results/rtx5090_t1/runs")
RV = Path("/media/tan/F/RWS2027/remote_results/vessl_t1/runs")
HOLD = {"PETA": "PETA", "Market": "Market1501"}
sig = lambda z: 1 / (1 + np.exp(-z))


def prior_off(fold):
    p = tr[tr.ds != HOLD[fold]][A].values.mean(0).clip(1e-3, 1 - 1e-3)
    return np.log((1 - p) / p) - (1 - 2 * p)


def load(d, kind="held"):
    z = np.load(Path(d) / f"{kind}_val_logits_last.npz")
    return z["idx"], z["logits"].astype(np.float64)


def met(idx, z, fold, tau=0.42):
    gt, grp = va[A].values[idx], va.ds.values[idx]
    m = challenge_metrics(gt, sig(z + tau * prior_off(fold)), grp)[0]
    return m


def auc(idx, z):
    gt = va[A].values[idx]; out = []
    for j in range(40):
        y = gt[:, j]
        if 0 < y.sum() < len(y):
            r = np.argsort(np.argsort(z[:, j])) + 1; P = y.sum(); N = len(y) - P
            out.append((r[y == 1].sum() - P * (P + 1) / 2) / (P * N))
    return float(np.mean(out))


def ens(dirs, kind="held"):
    zs = [load(d, kind) for d in dirs]
    assert all((x[0] == zs[0][0]).all() for x in zs)
    return zs[0][0], sum(x[1] for x in zs) / len(zs)


OUT = {}
# --- single encoders and pairs with H+ (PETA fold; in-domain Avg for the same models)
ENC = {"DINOv3 ViT-H+/16": K / "P_dinoHp224", "CLIP ViT-L/14": K / "P_clipL224", "MetaCLIP ViT-L/14": K / "M1_qg",
       "MetaCLIP2 ViT-H/14": K / "P_mc2H168", "EVA02-CLIP ViT-L/14": K / "Q_eva02L224", "PE-Core ViT-L/14": K / "PEc_L",
       "AIMv2 ViT-L/14": K / "Q_aimv2L224", "SigLIP2 So400m/14": K / "Q_sig2So14", "SigLIP2 ViT-L/16": K / "Q_sig2L384",
       "DFN-CLIP ViT-L/14": K / "Q_dfnL224"}
OUT["encoders"] = {}
for name, d in ENC.items():
    i, z = load(d); si, sz = load(d, "seen")
    row = {"peta_tau": met(i, z, "PETA")["Challenge_Avg"], "peta_auc": auc(i, z), "indomain_tau": met(si, sz, "PETA")["Challenge_Avg"]}
    if name != "DINOv3 ViT-H+/16":
        pi, pz = ens([K / "P_dinoHp224", d]); psi, psz = ens([K / "P_dinoHp224", d], "seen")
        row.update(pair_peta=met(pi, pz, "PETA")["Challenge_Avg"], pair_indomain=met(psi, psz, "PETA")["Challenge_Avg"])
    OUT["encoders"][name] = row
    print(name, {k: round(v, 4) for k, v in row.items()}, flush=True)

# --- ensemble design: third member, weights, distillation instead of ensembling
H, C = K / "P_dinoHp224", K / "P_clipL224"
OUT["third"] = {}
for name, d in {"MetaCLIP2 ViT-H/14": K / "P_mc2H168", "MetaCLIP ViT-L/14": K / "M1_qg", "SigLIP2 So400m/14": K / "Q_sig2So14", "AIMv2 ViT-L/14": K / "Q_aimv2L224"}.items():
    i, z = ens([H, C, d]); OUT["third"][name] = met(i, z, "PETA")["Challenge_Avg"]
print("third", OUT["third"])
OUT["weights"] = {}
hi, hz = load(H); _, cz = load(C); hmi, hmz = load(K / "Hp_M"); _, cmz = load(K / "C_M")
for w in (0.3, 0.4, 0.5, 0.6, 0.7, 1.0, 0.0):
    OUT["weights"][w] = {"PETA": met(hi, w * hz + (1 - w) * cz, "PETA")["Challenge_Avg"],
                         "Market": met(hmi, w * hmz + (1 - w) * cmz, "Market")["Challenge_Avg"]}
print("weights", OUT["weights"])
si, sz = load(K / "S_Hp_kd3_f8"); pi, pz = ens([K / "S_Hp_kd3_f8", C])
OUT["distill"] = {"student_peta": met(si, sz, "PETA")["Challenge_Avg"], "student_auc": auc(si, sz),
                  "student_plus_clip": met(pi, pz, "PETA")["Challenge_Avg"]}
print("distill", OUT["distill"])

# --- fine-tuning depth: (layer decay 0.75, freeze 8) vs (0.9, freeze 4)
DEPTH = {"H+": (K / "P_dinoHp224", K / "Hp_ld9", K / "Hp_M", K / "Hp_ld9_M"),
         "CLIP-L": (K / "P_clipL224", K / "C_ld9", K / "C_M", K / "C_ld9_M"),
         "SigLIP2-L": (K / "Q_sig2L384", K / "S_ld9", None, None)}
OUT["depth"] = {}
for name, (a, b, am, bm) in DEPTH.items():
    r = {}
    for tag, d, dm in (("conservative", a, am), ("deeper", b, bm)):
        i, z = load(d); si_, sz_ = load(d, "seen")
        r[tag] = {"indomain": met(si_, sz_, "PETA")["Challenge_Avg"], "peta": met(i, z, "PETA")["Challenge_Avg"], "peta_auc": auc(i, z)}
        if dm is not None:
            mi, mz = load(dm); r[tag]["market"] = met(mi, mz, "Market")["Challenge_Avg"]
    OUT["depth"][name] = r
    print("depth", name, r)
for tag, h, hm in (("conservative", H, K / "Hp_M"), ("deeper", K / "Hp_ld9", K / "Hp_ld9_M")):
    i, z = ens([h, C]); si_, sz_ = ens([h, C], "seen"); mi, mz = ens([hm, K / "C_M"])
    OUT["depth"].setdefault("ensemble", {})[tag] = {"peta": met(i, z, "PETA")["Challenge_Avg"],
                                                   "indomain": met(si_, sz_, "PETA")["Challenge_Avg"],
                                                   "market": met(mi, mz, "Market")["Challenge_Avg"]}
print("depth ensemble", OUT["depth"]["ensemble"])

# --- decision rule: tau curves, oracle offsets fitted on the held-out labels (upper bound of any threshold rule)
TAUS = [round(x, 2) for x in np.arange(0, 0.81, 0.05)]
OUT["tau"] = {}
for fold, members in (("PETA", {"H+": [H], "CLIP-L": [C], "H+ + CLIP-L": [H, C]}),
                      ("Market", {"H+": [K / "Hp_M"], "CLIP-L": [K / "C_M"], "H+ + CLIP-L": [K / "Hp_M", K / "C_M"]})):
    OUT["tau"][fold] = {}
    for name, ds in members.items():
        i, z = ens(ds)
        curve = {t: met(i, z, fold, t)["Challenge_Avg"] for t in TAUS}
        gt, grp = va[A].values[i], va.ds.values[i]
        orc, _ = fit_offsets(z, gt, grp)
        m42 = met(i, z, fold)
        OUT["tau"][fold][name] = {"curve": curve, "oracle": challenge_metrics(gt, sig(z + orc), grp)[0]["Challenge_Avg"],
                                  "mA": m42["mA"], "F1": m42["Instance_F1"], "P": m42["Instance_Prec"], "R": m42["Instance_Rec"], "auc": auc(i, z)}
        print("tau", fold, name, "raw", round(curve[0.0], 4), "0.42", round(m42["Challenge_Avg"], 4), "oracle", round(OUT["tau"][fold][name]["oracle"], 4), flush=True)

# --- noise: CLIP-L replicates (same recipe), and the pair with each replicate
reps = {"CLIP-L run A": K / "P_clipL224", "CLIP-L run B (seed 2)": K / "C_s2", "CLIP-L run C (QuickGELU)": K / "C_qg"}
OUT["noise"] = {}
for name, d in reps.items():
    i, z = load(d); pi, pz = ens([H, d])
    OUT["noise"][name] = {"single": met(i, z, "PETA")["Challenge_Avg"], "pair": met(pi, pz, "PETA")["Challenge_Avg"]}
vi, vz = load(RV / "P_clipL224")
_, hz2 = load(H)
OUT["noise"]["CLIP-L run D (other machine)"] = {"single": met(vi, vz, "PETA")["Challenge_Avg"], "pair": met(vi, (hz2 + vz) / 2, "PETA")["Challenge_Avg"]}
print("noise", OUT["noise"])

# --- proxy of the submitted ensembles (PETA fold) with the CLIP-L run A
V20 = [V / "kd2_a08_e10_PETA", V / "T_dinoL224_PETA", C, R5 / "T_mc2H"]
for d in V20:
    assert (Path(d) / "held_val_logits_last.npz").exists(), d
i, z = ens(V20); OUT["proxy"] = {"v20": met(i, z, "PETA")["Challenge_Avg"]}
i, z = ens([H, C]); OUT["proxy"]["v21"] = met(i, z, "PETA")["Challenge_Avg"]
i, z = ens([K / "Hp_ld9", C]); OUT["proxy"]["v22"] = met(i, z, "PETA")["Challenge_Avg"]
i, z = load(V / "kd2_a08_e10_PETA"); OUT["proxy"]["v17_student"] = met(i, z, "PETA")["Challenge_Avg"]
print("proxy", OUT["proxy"])
json.dump(OUT, open(Path(__file__).with_name("numbers.json"), "w"), indent=1)
