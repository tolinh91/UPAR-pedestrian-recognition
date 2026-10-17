"""Score a leave-one-dataset-out proxy run for both tracks and write <run>/scores.json.

Track 1: Challenge_Avg of last.pt after offsets fit on the seen domains (x0, x0.6, x1.0) and mean AUC on the held-out
dataset. Track 2: retrieval on the held-out dataset (gallery = its val images, queries = their query vectors), with
log-likelihood and L1 scoring of the raw probabilities (no offsets).
python score_run.py ../runs/<name>
"""
import json, sys
from pathlib import Path
import numpy as np
import offset_proxy
from eval_retrieval import load_task2
from probe_backbones import mean_auc
from upar.data import attr_names, load_split
from upar.retrieval import retrieval_metrics, score_l1, score_loglik


def main(run):
    run = Path(run)
    held = json.load(open(run / "args.json"))["holdout"]
    offset_proxy.main(str(run), "last.pt")
    t1 = json.load(open(run / "offset_proxy_last.json"))
    z = np.load(run / "held_val_logits_last.npz")
    va, A = load_split("val"), attr_names()
    gt = va[A].values[z["idx"]].astype(np.float32)
    prob = 1 / (1 + np.exp(-z["logits"]))
    ids, queries = load_task2("val")
    qid_of = dict(zip(ids.image, ids.qid))
    gq = np.array([qid_of[p] for p in va.image.values[z["idx"]]])
    q_ids = np.unique(gq)
    t2 = {name: retrieval_metrics(fn(queries[q_ids], prob), queries[q_ids], gt, gq, q_ids)
          for name, fn in (("loglik", score_loglik), ("l1", score_l1))}
    pick = lambda m: {k: round(m[k], 4) for k in ("Challenge_Avg", "mA", "Instance_F1")}
    res = {"holdout": held, "auc": round(float(mean_auc(gt, z["logits"], gt.any(0))), 4),
           "t1_off0.6": pick(t1["fitted offsets x 0.60"]), "t1_raw": pick(t1["fitted offsets x 0.00"]),
           "t1_off1.0": pick(t1["fitted offsets x 1.00"]),
           "t2_loglik": {k: round(v, 4) for k, v in t2["loglik"].items()}, "t2_l1": {k: round(v, 4) for k, v in t2["l1"].items()}}
    json.dump(res, open(run / "scores.json", "w"), indent=1)
    print("SCORES", run.name, held, "T1 Avg@0.6", res["t1_off0.6"]["Challenge_Avg"], "AUC", res["auc"],
          "T2 mADM", res["t2_loglik"]["mADM"], "mAP", res["t2_loglik"]["mAP"])


if __name__ == "__main__":
    main(sys.argv[1])
