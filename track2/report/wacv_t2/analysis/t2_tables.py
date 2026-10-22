"""Table numbers of the Track 2 paper that t2_numbers.py does not give, from the same saved held-out logits.

../../../.venv/bin/python t2_tables.py      (writes t2_tables.json)
six:    all six metrics of the submitted-type ensemble HB + Tw70 under the likelihood score and the L1 distance, each metric
        averaged over the two CLIP seeds (as the mADM columns of the encoder table)
dino2:  two trainings of the DINOv3 model (HB, HP) averaged, the control for "a second encoder of the same kind"
counts: gallery images and queries of each held-out gallery
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/media/tan/F/RWS2027/Upar-Challenge-Track-2/par")
from eval_retrieval import load_task2  # noqa: E402
from upar.data import attr_names, load_split  # noqa: E402
from upar.retrieval import retrieval_metrics, score_l1, score_loglik  # noqa: E402

R = Path("/media/tan/F/RWS2027/remote_results/rare/runs")
A = attr_names(); va = load_split("val")
ids, Q = load_task2("val"); qid_of = dict(zip(ids.image, ids.qid))
FOLDS = {"PETA": "", "Market": "M"}


def load(run, pre):
    name, _, wise = run.partition("@")
    d = np.load(R / f"{pre}{name}" / (f"held_val_logits_wise{wise}.npz" if wise else "held_val_logits_last.npz"))
    return dict(zip(d["idx"], d["logits"].astype(np.float64)))


def stack(dicts):
    idx = np.array(sorted(set.intersection(*[set(d) for d in dicts])))
    return idx, np.mean([np.stack([d[i] for i in idx]) for d in dicts], 0)


def metrics(z, idx, score):
    gt = va[A].values[idx].astype(np.float32); gq = np.array([qid_of[x] for x in va.image.values[idx]]); qids = np.unique(gq)
    p = 1 / (1 + np.exp(-z))
    return retrieval_metrics(score(Q[qids], p), Q[qids], gt, gq, qids)


out = {"six": {}, "dino2": {}, "counts": {}}
for fold, pre in FOLDS.items():
    hb = load("HB_s1", pre)
    for name, score in (("ll", score_loglik), ("l1", score_l1)):
        ms = [metrics(*stack([hb, load(t, pre)])[::-1], score) for t in ("T_s1@0.70", "T_s2@0.70")]
        out["six"].setdefault(fold, {})[name] = {k: float(np.mean([m[k] for m in ms])) for k in ms[0] if k != "queries"}
    idx, z = stack([hb, load("HP_s1", pre)])
    out["dino2"][fold] = metrics(z, idx, score_loglik)["mADM"]
    m = metrics(z, idx, score_loglik)
    out["counts"][fold] = {"images": int(len(idx)), "queries": int(m["queries"])}
json.dump(out, open(Path(__file__).with_name("t2_tables.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
