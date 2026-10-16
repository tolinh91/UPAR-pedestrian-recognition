"""Extra paper numbers from saved held-out logits (numpy only, no model runs).

../../../../Upar-Challenge-Track-2/.venv/bin/python ablations.py
(1) one encoder vs two copies of one encoder vs two different encoders, logit mean, decision logit > 0.
    Runs: remote_results/rare/runs/<V>_s<seed> (PETA held out) and M<V>_s<seed> (Market held out).
    LA recipe (logit_adj tau 0.42): CLIP TL_s1/s2, H+ HL_s1. BCE recipe (DeepMAR): CLIP B_s1/s2, H+ HB_s1 / HP_s1.
(2) per-attribute balanced accuracy of the three heads of a joint model (DTB, 2 seeds, logits16b with h/c/j parts).
Writes ablations.json next to this file.
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, "/media/tan/F/RWS2027/Upar-Challenge-Track-2/par")
from upar.data import attr_names, load_split  # noqa: E402
from upar.metrics import challenge_metrics  # noqa: E402

A = attr_names(); va = load_split("val"); GT = va[A].values; DS = va.ds.values
RARE = Path("/media/tan/F/RWS2027/remote_results/rare/runs")
JOINT = Path("/media/tan/F/RWS2027/remote_results/fusion/logits16b")
FOLDS = {"PETA": ("", "PETA"), "Market": ("M", "Market1501")}


def load_run(name):
    d = np.load(RARE / name / "held_val_logits_last.npz")
    return dict(zip(d["idx"], d["logits"].astype(np.float64)))


def avg_score(runs, hold):
    """Challenge_Avg (points) of the logit mean of runs (dicts idx -> logits) on the held-out rows."""
    idx = sorted(set.intersection(*[set(r) for r in runs]))
    z = np.mean([np.stack([r[i] for i in idx]) for r in runs], 0)
    m = challenge_metrics(GT[idx], (z > 0).astype(float), DS[idx], thr=0.5)[0]
    assert set(DS[idx]) == {hold}
    return 100 * m["Challenge_Avg"]


def one_vs_two():
    rec = {"LA": {"C": ["TL_s1", "TL_s2"], "H": ["HL_s1"]},
           "BCE": {"C": ["B_s1", "B_s2"], "H": ["HB_s1", "HP_s1"]}}
    out = {}
    for r, v in rec.items():
        for fold, (pre, hold) in FOLDS.items():
            L = {n: load_run(pre + n) for n in v["C"] + v["H"]}
            C, H = v["C"], v["H"]
            o = out.setdefault(r, {}).setdefault(fold, {})
            o["CLIP"] = np.mean([avg_score([L[c]], hold) for c in C])
            o["H+"] = np.mean([avg_score([L[h]], hold) for h in H])
            o["CLIP+CLIP"] = avg_score([L[c] for c in C], hold)
            if len(H) > 1:
                o["H+ + H+"] = avg_score([L[h] for h in H], hold)
            o["H+ + CLIP"] = np.mean([avg_score([L[h], L[c]], hold) for h in H for c in C])
    for r in out:
        out[r]["Mean"] = {k: np.mean([out[r][f][k] for f in FOLDS]) for k in out[r]["PETA"]}
    return out


def bal_acc(gt, z):
    """Per-attribute balanced accuracy at logit > 0; NaN where an attribute has no positive."""
    g = gt.astype(bool); p = z > 0
    tp = (g & p).sum(0); fn = (g & ~p).sum(0); tn = (~g & ~p).sum(0); fp = (~g & p).sum(0)
    ba = (tp / np.maximum(tp + fn, 1) + tn / np.maximum(tn + fp, 1)) / 2
    ba[g.sum(0) == 0] = np.nan
    return ba


def joint_heads():
    """Per head: Challenge_Avg on each fold/seed, and per-attribute wins of the joint head over the member heads."""
    out = {"avg": {}, "wins": [], "per_attr": {}}
    for fold, (_, hold) in FOLDS.items():
        for s in (1, 2):
            d = np.load(JOINT / f"DTB_{fold}_s{s}.npz")
            rows = d["idx"]; keep = DS[rows] == hold; rows = rows[keep]
            z = {k: d[f"{k}_w1.00"].astype(np.float64)[keep] for k in ("h", "c", "j")}
            z["hc"] = (z["h"] + z["c"]) / 2; z["hcj"] = (z["h"] + z["c"] + z["j"]) / 3
            for k, v in z.items():
                m = challenge_metrics(GT[rows], (v > 0).astype(float), DS[rows], thr=0.5)[0]
                out["avg"].setdefault(k, {})[f"{fold}_s{s}"] = 100 * m["Challenge_Avg"]
            ba = {k: bal_acc(GT[rows], z[k]) for k in ("h", "c", "j")}
            ok = ~np.isnan(ba["j"])
            out["wins"].append({"fold": fold, "seed": s, "n_attr": int(ok.sum()),
                                "j_gt_both": int((ba["j"][ok] > np.maximum(ba["h"], ba["c"])[ok]).sum()),
                                "j_lt_both": int((ba["j"][ok] < np.minimum(ba["h"], ba["c"])[ok]).sum())})
            for k in ("h", "c", "j"):
                out["per_attr"].setdefault(k, {})[f"{fold}_s{s}"] = [None if np.isnan(x) else float(x) for x in ba[k]]
    return out


if __name__ == "__main__":
    res = {"one_vs_two": one_vs_two(), "joint_heads": joint_heads(), "attrs": A}
    for r, v in res["one_vs_two"].items():
        print(r)
        for k in v["PETA"]:
            print(f"  {k:12s} " + "  ".join(f"{f} {v[f][k]:.2f}" for f in ("PETA", "Market", "Mean")))
    print("joint model DTB, Challenge_Avg per head")
    for k, v in res["joint_heads"]["avg"].items():
        print(f"  {k:4s} " + "  ".join(f"{f} {x:.2f}" for f, x in v.items()))
    for w in res["joint_heads"]["wins"]:
        print("  wins", w)
    json.dump(res, open(Path(__file__).with_name("ablations.json"), "w"), indent=1, default=float)
