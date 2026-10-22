"""Is kNN smoothing model-agnostic? Apply it, with PRETRAINED DINOv3 ViT-H+ features (never fine-tuned, so independent of
the attribute model) and the submitted setting k = 50, alpha = 0.2, to every development model with held-out logits on
the PETA and Market-1501 folds (runs/remote*/<run>/held_val_logits_last.npz; PA100k-fold models have no gallery features).
Reports, per fold, the mADM change of each model under the likelihood score.  ../../../.venv/bin/python t2_knn_general.py"""
import glob, json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/media/tan/F/RWS2027/Upar-Challenge-Track-2/par")
from eval_retrieval import load_task2  # noqa: E402
from upar.data import attr_names, load_split  # noqa: E402
from upar.retrieval import retrieval_metrics, score_loglik  # noqa: E402
A = attr_names(); va = load_split("val"); ids, Q = load_task2("val"); qid_of = dict(zip(ids.image, ids.qid))
C = np.load("/media/tan/F/RWS2027/remote_results/t2k/compact.npz")
HOLD = {"PETA": "PETA", "Market1501": "Market"}
gal = {}
for hold, fold in HOLD.items():
    rows = C[f"{fold}_rows"]; pos = {r: i for i, r in enumerate(rows)}
    gt = va[A].values[rows].astype(np.float32); gq = np.array([qid_of[x] for x in va.image.values[rows]]); qids = np.unique(gq)
    gal[hold] = dict(rows=rows, pos=pos, gt=gt, gq=gq, qids=qids, Qm=Q[qids], I=C[f"{fold}_pre_mean_nn"].astype(np.int64)[:, :50])
res = {}
for f in sorted(glob.glob("/media/tan/F/RWS2027/Upar-Challenge-Track-2/runs/remote*/*/held_val_logits_last.npz")):
    d = np.load(f); idx = d["idx"]; hold = va.ds.values[idx[0]]
    if hold not in gal:
        continue
    g = gal[hold]; zz = dict(zip(idx, d["logits"].astype(np.float64)))
    if not set(g["rows"]) <= set(zz):
        continue
    z = np.stack([zz[r] for r in g["rows"]])
    def madm(x):
        return 100 * retrieval_metrics(score_loglik(g["Qm"], 1 / (1 + np.exp(-x))), g["Qm"], g["gt"], g["gq"], g["qids"])["mADM"]
    base = madm(z); knn = madm(0.8 * z + 0.2 * z[g["I"]].mean(1))
    name = os.path.basename(os.path.dirname(f)); res[name] = {"held": hold, "base": base, "knn": knn}
    print(f"{hold:10s} {name:28s} {base:6.2f} -> {knn:6.2f} ({knn - base:+.2f})", flush=True)
summ = {}
for hold in gal:
    dl = [v["knn"] - v["base"] for v in res.values() if v["held"] == hold]
    summ[hold] = {"n": len(dl), "helps": int(sum(x > 0 for x in dl)), "min": min(dl), "median": float(np.median(dl)), "max": max(dl)}
print(json.dumps(summ, indent=1))
json.dump({"runs": res, "summary": summ}, open(Path(__file__).with_name("t2_knn_general.json"), "w"), indent=1)
