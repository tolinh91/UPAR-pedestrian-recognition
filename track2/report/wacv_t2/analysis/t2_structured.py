"""Structured likelihood score for attribute retrieval (no retraining).

Independent score (paper Eq. ll): s = sum_k q_k log p_k + (1-q_k) log(1-p_k), p = sigmoid(z).
Structured score: for each one-of-K group G (age, hair length, lower-body type, upper colours, lower colours; the groups
of group CE), when the query has exactly one positive value k in G, the group's term is the categorical log-probability
log softmax(z_G / T)[k] instead of the product of independent sigmoids; other queries/attributes keep the independent
terms. Evaluated on both held-out folds with the gce ensemble (submitted members, seed 1) and the base ensemble, with
and without kNN smoothing (k=50, a=0.2, fine-tuned DINOv3 features).
../../../.venv/bin/python t2_structured.py"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/media/tan/F/RWS2027/Upar-Challenge-Track-2/par")
from eval_retrieval import load_task2  # noqa: E402
from upar.data import attr_names, load_split  # noqa: E402
from upar.retrieval import retrieval_metrics, score_loglik  # noqa: E402
A = attr_names(); va = load_split("val"); ids, Q = load_task2("val"); qid_of = dict(zip(ids.image, ids.qid))
C = np.load("/media/tan/F/RWS2027/remote_results/t2k/compact.npz"); R = Path("/media/tan/F/RWS2027/remote_results/rare/runs")
GROUPS = [[0, 1, 2], [4, 5, 6], [33, 34], list(range(8, 20)), list(range(21, 33))]
INGROUP = sorted(set(i for g in GROUPS for i in g)); FREE = [k for k in range(40) if k not in INGROUP]

def structured(Qm, z, T=1.0, eps=1e-4):
    p = np.clip(1 / (1 + np.exp(-z)), eps, 1 - eps); lp, lq = np.log(p), np.log1p(-p)
    S = Qm[:, FREE] @ (lp - lq)[:, FREE].T + lq[:, FREE].sum(1)[None, :]
    for g in GROUPS:
        qg = Qm[:, g]; one = qg.sum(1) == 1
        zg = z[:, g] / T; ls = zg - np.logaddexp.reduce(zg, axis=1, keepdims=True)        # (N, |G|) log softmax
        S[one] += ls[:, qg[one].argmax(1)].T                                            # categorical term
        S[~one] += qg[~one] @ (lp - lq)[:, g].T + lq[:, g].sum(1)[None, :]               # independent fallback
    return S

def saved(run, pre):
    name, _, wise = run.partition("@")
    d = np.load(R / f"{pre}{name}" / (f"held_val_logits_wise{wise}.npz" if wise else "held_val_logits_last.npz"))
    return dict(zip(d["idx"], d["logits"].astype(np.float64)))

out = {}
for fold, (pre, hold) in {"PETA": ("", "PETA"), "Market": ("M", "Market1501")}.items():
    rows = C[f"{fold}_rows"]
    gt = va[A].values[rows].astype(np.float32); gq = np.array([qid_of[x] for x in va.image.values[rows]]); qids = np.unique(gq); Qm = Q[qids]
    one_share = {f"G{i}": float((Qm[:, g].sum(1) == 1).mean()) for i, g in enumerate(GROUPS)}
    cl = saved("T_s1@0.70", pre); clip = np.stack([cl[r] for r in rows])
    Z = {"gce ensemble": (C[f"{fold}_HG"].astype(np.float64) + C[f"{fold}_TG"].astype(np.float64)) / 2,
         "base ensemble": (C[f"{fold}_HB"].astype(np.float64) + clip) / 2}
    I = C[f"{fold}_ft_mean_nn"].astype(np.int64)[:, :50]
    res = out.setdefault(fold, {"queries with one value per group": one_share})
    for zn, z in Z.items():
        for kn, zz in (("", z), (" + kNN", 0.8 * z + 0.2 * z[I].mean(1))):
            m_ind = retrieval_metrics(score_loglik(Qm, 1 / (1 + np.exp(-zz))), Qm, gt, gq, qids)
            res[f"{zn}{kn} | independent"] = {k: 100 * v for k, v in m_ind.items() if k != "queries"}
            for T in (1.0, 2.0):
                m_st = retrieval_metrics(structured(Qm, zz, T), Qm, gt, gq, qids)
                res[f"{zn}{kn} | structured T={T:g}"] = {k: 100 * v for k, v in m_st.items() if k != "queries"}
            print(fold, zn + kn, " | ".join(f"{k.split('| ')[1]}: " + " ".join(f"{m} {v:.2f}" for m, v in res[k].items() if m in ("mADM", "mAP", "Rank-1"))
                                          for k in res if k.startswith(zn + kn + " |")), flush=True)
    print(fold, "share of queries with exactly one value:", {k: round(v, 2) for k, v in one_share.items()})
json.dump(out, open(Path(__file__).with_name("t2_structured.json"), "w"), indent=1)
