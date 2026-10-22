"""Numbers of the Track 2 WACV paper, recomputed from saved held-out logits of the leave-one-dataset-out folds.

../../../.venv/bin/python t2_numbers.py            (writes t2_numbers.json next to this file)
Folds: PETA held out and Market-1501 held out; each held-out validation split is its own gallery with its own
queries (one query per distinct label vector, the organizers' task-2 annotations). Members (remote_results/rare/runs):
  HB  DINOv3 ViT-H+/16 with the recipe of the submitted hp.pt (query head, BCE + DeepMAR, smoothing 0.05)
  HP  the same encoder re-trained with the same recipe (second copy)
  B   CLIP ViT-L/14 with a plain query head (2 seeds)        T   CLIP with the text-anchor query head (2 seeds)
  Tw70  T after WiSE-FT with alpha 0.7 (the submitted CLIP member)
Ensembles average the member logits. CLIP seeds are averaged at the metric level. Scores: log-likelihood of the exact
query (score_loglik, as in the package) and the L1 distance of the UPAR baseline (score_l1).
Calibration: per-attribute Platt maps p = sigmoid(a z + b) fitted on the held-out labels (oracle) or on the validation
images of the two source datasets (seen_val_logits); NLL in nats per image over the attributes each dataset labels.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, "/media/tan/F/RWS2027/Upar-Challenge-Track-2/par")
from eval_retrieval import load_task2  # noqa: E402
from upar.data import attr_names, invalid_mask, load_split  # noqa: E402
from upar.retrieval import retrieval_metrics, score_l1, score_loglik  # noqa: E402

R = Path("/media/tan/F/RWS2027/remote_results/rare/runs")
A = attr_names(); va = load_split("val"); tr = load_split("train"); INV = invalid_mask(tr)
ids, Q = load_task2("val"); qid_of = dict(zip(ids.image, ids.qid))
FOLDS = {"PETA": ("", "PETA"), "Market": ("M", "Market1501")}
MEMBERS = {"HB": ["HB_s1"], "HP": ["HP_s1"], "B": ["B_s1", "B_s2"], "T": ["T_s1", "T_s2"], "Tw70": ["T_s1@0.70", "T_s2@0.70"]}


def load(run, pre, kind="held"):
    name, _, wise = run.partition("@")
    f = R / f"{pre}{name}" / (f"{kind}_val_logits_wise{wise}.npz" if wise else f"{kind}_val_logits_last.npz")
    d = np.load(f)
    return dict(zip(d["idx"], d["logits"].astype(np.float64)))


def stack(dicts):
    idx = np.array(sorted(set.intersection(*[set(d) for d in dicts])))
    return idx, np.mean([np.stack([d[i] for i in idx]) for d in dicts], 0)


def gallery(idx):
    gt = va[A].values[idx].astype(np.float32); gq = np.array([qid_of[x] for x in va.image.values[idx]])
    return gt, gq, np.unique(gq)


def madm(z, idx, score="ll", full=False):
    gt, gq, qids = gallery(idx); p = 1 / (1 + np.exp(-z))
    s = score_loglik(Q[qids], p) if score == "ll" else score_l1(Q[qids], p)
    m = retrieval_metrics(s, Q[qids], gt, gq, qids)
    return m if full else m["mADM"]


def valid(idx):
    v = np.ones((len(idx), len(A)), bool)
    ds = va.ds.values[idx]
    for d, m in INV.items():
        v[ds == d] = ~m
    return v


def nll(z, idx):
    y = va[A].values[idx].astype(np.float64); v = valid(idx); p = np.clip(1 / (1 + np.exp(-z)), 1e-7, 1 - 1e-7)
    return float(np.sum(-(y * np.log(p) + (1 - y) * np.log(1 - p)) * v) / len(idx))


def platt(z, idx):
    y = va[A].values[idx]; v = valid(idx); a = np.ones(len(A)); b = np.zeros(len(A))
    for k in range(len(A)):
        m = v[:, k]
        if m.sum() < 20 or len(np.unique(y[m, k])) < 2:
            continue
        lr = LogisticRegression(C=1e4, max_iter=1000).fit(z[m, k:k + 1], y[m, k])
        a[k], b[k] = lr.coef_[0, 0], lr.intercept_[0]
    return a, b


def runs_of(member, pre):
    return [load(r, pre) for r in MEMBERS[member]]


out = {"members": {}, "pairs": {}, "l1": {}, "calib": {}, "metrics6": {}}
for fold, (pre, hold) in FOLDS.items():
    # single members and pairs (CLIP seeds averaged at the metric level)
    for m in MEMBERS:
        vals = []
        for d in runs_of(m, pre):
            idx, z = stack([d]); vals.append(madm(z, idx))
        out["members"].setdefault(m, {})[fold] = float(np.mean(vals))
    for h in ("HB", "HP"):
        for c in ("B", "T", "Tw70"):
            vals, vals_l1 = [], []
            for dc in runs_of(c, pre):
                idx, z = stack([runs_of(h, pre)[0], dc]); vals.append(madm(z, idx))
                if (h, c) == ("HB", "Tw70"):
                    vals_l1.append(madm(z, idx, "l1"))
            out["pairs"].setdefault(f"{h}+{c}", {})[fold] = float(np.mean(vals))
            if vals_l1:
                out["l1"].setdefault("HB+Tw70", {})[fold] = float(np.mean(vals_l1))
    # L1 versus log-likelihood for the single members too, and all six metrics for the submitted-type pair
    for m in ("HB", "Tw70"):
        vals = []
        for d in runs_of(m, pre):
            idx, z = stack([d]); vals.append(madm(z, idx, "l1"))
        out["l1"].setdefault(m, {})[fold] = float(np.mean(vals))
    idx, z = stack([runs_of("HB", pre)[0], runs_of("Tw70", pre)[0]])
    out["metrics6"][fold] = {"ll": madm(z, idx, "ll", True), "l1": madm(z, idx, "l1", True)}
    # calibration of the pair HP + T (seed 1; HB has no saved source-domain logits on the Market fold): source = validation images of the two source datasets
    zh = {}
    for kind in ("held", "seen"):
        zh[kind] = stack([load("HP_s1", pre, kind), load("T_s1", pre, kind)])
    (ih, Zh), (iss, Zs) = zh["held"], zh["seen"]
    a_h, b_h = platt(Zh, ih); a_s, b_s = platt(Zs, iss)
    cal = {"nll_held": nll(Zh, ih), "nll_seen": nll(Zs, iss)}
    cal["cal_term_held"] = cal["nll_held"] - nll(a_h * Zh + b_h, ih)
    cal["cal_term_seen"] = cal["nll_seen"] - nll(a_s * Zs + b_s, iss)
    cal["madm_raw"] = madm(Zh, ih)
    cal["madm_oracle"] = madm(a_h * Zh + b_h, ih)
    cal["madm_source"] = madm(a_s * Zh + b_s, ih)
    for tag, (a, b) in {"src_scale_src_bias": (a_s, b_s), "src_scale_held_bias": (a_s, b_h),
                        "held_scale_src_bias": (a_h, b_s), "held_scale_held_bias": (a_h, b_h)}.items():
        cal[f"nll_{tag}"] = nll(a * Zh + b, ih)
        cal[f"madm_{tag}"] = madm(a * Zh + b, ih)
    for t in (0.9, 1.1, 1.3):
        cal[f"madm_T{t}"] = madm(Zh / t, ih)
    cal["scale_held"], cal["scale_seen"] = a_h.tolist(), a_s.tolist()
    cal["bias_held"], cal["bias_seen"] = b_h.tolist(), b_s.tolist()
    out["calib"][fold] = cal
    print(fold, "done", {k: round(v, 4) for k, v in cal.items() if not isinstance(v, list)}, flush=True)

json.dump(out, open(Path(__file__).with_suffix(".json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "calib"}, indent=1))
