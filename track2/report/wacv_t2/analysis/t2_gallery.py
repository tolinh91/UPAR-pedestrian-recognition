"""Gallery-structure methods and group cross-entropy on both held-out folds (numpy on extracted logits and features).

../../../.venv/bin/python t2_gallery.py [--quick]       (writes t2_gallery.json)
Inputs (remote_results/t2k/out, from the 2026-10-18 runs; same recipe as the submitted models, one seed):
  HB_{fold}_s1_emb.npz   DINOv3 ViT-H+ model of the fold: logits, CLS and L2-normalised mean patch token of every val image
  PRE_hp_emb.npz         the pretrained DINOv3 ViT-H+ (never fine-tuned): CLS and mean patch token
  HG/TG_{fold}_s1.npz    the DINOv3 and CLIP models trained with group cross-entropy 0.5 (TG also WiSE-FT 0.7)
  saved CLIP logits      remote_results/rare/runs/{,M}T_s{1,2} (WiSE-FT 0.7), as in the paper tables
Ensemble = DINOv3 model (new run) + CLIP (text score, WiSE-FT 0.7); metrics averaged over the two CLIP seeds.
Methods (all label-free, applied the same way to every gallery):
  knn     Eq. 4 of the paper, k nearest by cosine of a feature
  prop    label propagation z <- (1-a) z0 + a W z, W = row-normalised kNN graph, 10 iterations
  mutual  Eq. 4 with mutual k-nearest neighbours only (keeps at least the nearest one)
  qe      query expansion: s'(q,g) = s(q,g) + beta * mean cosine between g and the top-m images of q under s
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/media/tan/F/RWS2027/Upar-Challenge-Track-2/par")
from eval_retrieval import load_task2  # noqa: E402
from upar.data import attr_names, load_split  # noqa: E402
from upar.retrieval import retrieval_metrics, score_loglik  # noqa: E402

O = Path("/media/tan/F/RWS2027/remote_results/t2k/out")
R = Path("/media/tan/F/RWS2027/remote_results/rare/runs")
A = attr_names(); va = load_split("val"); ids, Q = load_task2("val"); qid_of = dict(zip(ids.image, ids.qid))
FOLDS = {"PETA": ("", "PETA"), "Market": ("M", "Market1501")}
QUICK = "--quick" in sys.argv


def saved(run, pre):
    name, _, wise = run.partition("@")
    d = np.load(R / f"{pre}{name}" / (f"held_val_logits_wise{wise}.npz" if wise else "held_val_logits_last.npz"))
    return dict(zip(d["idx"], d["logits"].astype(np.float64)))


def newrun(name, key="logits_w1.00"):
    d = np.load(O / f"{name}.npz")
    return dict(zip(d["idx"], d[key].astype(np.float64)))


class Gallery:
    def __init__(self, hold, rows):
        self.rows = rows
        self.gt = va[A].values[rows].astype(np.float32)
        self.gq = np.array([qid_of[x] for x in va.image.values[rows]]); self.qids = np.unique(self.gq)
        self.Qm = Q[self.qids]

    def metrics(self, z=None, S=None):
        if S is None:
            S = score_loglik(self.Qm, 1 / (1 + np.exp(-z)))
        return retrieval_metrics(S, self.Qm, self.gt, self.gq, self.qids)


def knn_idx(E, k):
    S = E @ E.T; np.fill_diagonal(S, -np.inf)
    I = np.argpartition(-S, k, axis=1)[:, :k]
    o = np.argsort(-np.take_along_axis(S, I, 1), axis=1)
    return np.take_along_axis(I, o, 1), S


def smooth(z, I, a):
    return (1 - a) * z + a * z[I].mean(1)


def propagate(z, I, a, it=10):
    x = z.copy()
    for _ in range(it):
        x = (1 - a) * z + a * x[I].mean(1)
    return x


def mutual_smooth(z, I, a, k):
    Ik = I[:, :k]; n = len(z)
    rank_of = [set(r) for r in Ik]
    out = z.copy()
    for g in range(n):
        m = [j for j in Ik[g] if g in rank_of[j]] or [Ik[g, 0]]
        out[g] = (1 - a) * z[g] + a * z[m].mean(0)
    return out


def qe_scores(z, E, m, beta):
    S = score_loglik(G.Qm, 1 / (1 + np.exp(-z)))
    top = np.argpartition(-S, m, axis=1)[:, :m]
    C = np.stack([E[t].mean(0) for t in top]) @ E.T              # (Q, G) mean cosine to the top-m images
    return S + beta * C


C = np.load("/media/tan/F/RWS2027/remote_results/t2k/compact.npz")
out = {}
for fold, (pre, hold) in FOLDS.items():
    rows = C[f"{fold}_rows"]; assert (rows == np.where(va.ds.values == hold)[0]).all()
    G = Gallery(hold, rows)
    lg = {n: C[f"{fold}_{n}"].astype(np.float64) for n in ("HB", "HG", "TG", "TG1")}
    cd = [saved(f"T_s{s}@0.70", pre) for s in (1, 2)]; hd = saved("HB_s1", pre)
    clip = [np.stack([c[r] for r in rows]) for c in cd]
    hb_old = np.stack([hd[r] for r in rows])
    feats = [f for f in ("ft_mean", "ft_cls", "pre_mean", "pre_cls") if f"{fold}_{f}_nn" in C]
    NN = {f: C[f"{fold}_{f}_nn"].astype(np.int64) for f in feats}
    PC = {f: C[f"{fold}_{f}_pca64"].astype(np.float32) for f in feats if f"{fold}_{f}_pca64" in C}
    for f in PC:
        PC[f] /= np.linalg.norm(PC[f], axis=1, keepdims=True)
    res = out.setdefault(fold, {})

    def avg(fn, zs):
        ms = [fn(z) for z in zs]
        return {k: float(np.mean([m[k] for m in ms])) for k in ms[0] if k != "queries"}

    # ---- group cross-entropy (one seed each) and the run-to-run reference
    pairs = {"saved DINOv3 + CLIP(2 seeds)": [(hb_old + c) / 2 for c in clip],
             "new DINOv3 + CLIP(2 seeds)": [(lg["HB"] + c) / 2 for c in clip],
             "new DINOv3 + CLIP seed 1": [(lg["HB"] + clip[0]) / 2],
             "gce DINOv3 + gce CLIP": [(lg["HG"] + lg["TG"]) / 2],
             "gce DINOv3 + CLIP(2 seeds)": [(lg["HG"] + c) / 2 for c in clip],
             "new DINOv3 + gce CLIP": [(lg["HB"] + lg["TG"]) / 2],
             "DINOv3 alone (new)": [lg["HB"]], "gce DINOv3 alone": [lg["HG"]],
             "CLIP seed 1 alone": [clip[0]], "gce CLIP alone": [lg["TG"]]}
    for name, zs in pairs.items():
        res.setdefault("gce", {})[name] = avg(lambda z: G.metrics(z), zs)["mADM"]
    print(fold, "gce", {k: round(100 * v, 2) for k, v in res["gce"].items()}, flush=True)

    Z = pairs["new DINOv3 + CLIP(2 seeds)"]
    res["base"] = avg(lambda z: G.metrics(z), Z)
    for f in feats:
        res.setdefault("nn1_same_labels", {})[f] = float((G.gt[NN[f][:, 0]] == G.gt).all(1).mean())
    grid = [(10, .2), (20, .2), (50, .2), (20, .3), (50, .3)] if not QUICK else [(50, .2)]
    for f in feats:
        I = NN[f]
        for k, a in grid:
            res.setdefault("knn", {})[f"{f} k{k} a{a}"] = avg(lambda z: G.metrics(smooth(z, I[:, :k], a)), Z)["mADM"]
        if f in ("ft_mean", "pre_mean"):
            for k, a in ([(10, .2), (20, .2), (20, .3)] if not QUICK else [(20, .2)]):
                res.setdefault("prop", {})[f"{f} k{k} a{a}"] = avg(lambda z: G.metrics(propagate(z, I[:, :k], a)), Z)["mADM"]
            for k, a in ([(20, .3), (50, .3), (50, .4)] if not QUICK else [(50, .3)]):
                res.setdefault("mutual", {})[f"{f} k{k} a{a}"] = avg(lambda z: G.metrics(mutual_smooth(z, I, a, k)), Z)["mADM"]
        print(fold, f, "done", flush=True)
    I = NN["ft_mean"]
    for base_name, zs in (("raw", Z), ("knn", [smooth(z, I, .2) for z in Z])):
        for f in PC:
            for m in (1, 3, 5):
                for beta in ((1, 2, 4, 8) if not QUICK else (4,)):
                    res.setdefault("qe", {})[f"{base_name} {f} m{m} b{beta}"] = avg(
                        lambda z: G.metrics(S=qe_scores(z, PC[f], m, beta)), zs)["mADM"]
    print(fold, "qe done", flush=True)
json.dump(out, open(Path(__file__).with_name("t2_gallery.json"), "w"), indent=1)
for fold, r in out.items():
    print("==", fold, "base", round(100 * r["base"]["mADM"], 2), "nn1", {k: round(v, 3) for k, v in r["nn1_same_labels"].items()})
    for sec in ("knn", "prop", "mutual", "qe"):
        best = sorted(r[sec].items(), key=lambda kv: -kv[1])[:6]
        print(" ", sec, [(k, round(100 * (v - r["base"]["mADM"]), 2)) for k, v in best])
