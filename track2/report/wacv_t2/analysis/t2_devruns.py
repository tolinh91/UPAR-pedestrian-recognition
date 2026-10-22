"""Robustness check over the models of development (report_review/analysis/t12_runs.json, held-out logits of 49 models:
42 PETA-fold, 5 Market-fold, 2 PA100k-fold; DINOv3, CLIP-family, EVA-02, SigLIP2, ConvNeXt and other encoders).
Prints, per held-out dataset, the mADM gain of log-likelihood over L1 ranking and of Platt recalibration fitted on the
source datasets (in points), and how many models the latter helps.  python3 t2_devruns.py -> t2_devruns.json
"""
import json
from collections import defaultdict
from pathlib import Path

d = json.load(open("/media/tan/F/RWS2027/report_review/analysis/t12_runs.json"))
out = defaultdict(lambda: {"n": 0, "ll_minus_l1": [], "source_recal": [], "oracle_recal": []})
for run, v in d.items():
    if v.get("madm_source_recal") is None:
        continue
    o = out[v["held"]]; o["n"] += 1
    o["ll_minus_l1"].append(100 * (v["madm"] - v["madm_l1"]))
    o["source_recal"].append(100 * (v["madm_source_recal"] - v["madm"]))
    o["oracle_recal"].append(100 * (v["madm_oracle_recal"] - v["madm"]))
res = {h: {"n": o["n"], "ll_minus_l1_min": min(o["ll_minus_l1"]), "ll_minus_l1_max": max(o["ll_minus_l1"]),
           "source_recal_helps": sum(x > 0 for x in o["source_recal"]),
           "source_recal_mean": sum(o["source_recal"]) / o["n"], "oracle_recal_mean": sum(o["oracle_recal"]) / o["n"]}
       for h, o in out.items()}
json.dump(res, open(Path(__file__).with_name("t2_devruns.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
