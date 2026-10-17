"""Compare Track 3 LODO runs: AP / AP50 (no-ear score, from the run logs) and the gross-error anatomy of the last epoch.

python lodo_report.py RUNS_DIR LOGS_DIR NAME...     (NAME = run name, e.g. L_peta_base; the fold is the 2nd field)
Per run: no-ear AP and AP50 of the last epoch, rate of persons with OKS < 0.5, the part of them that a left-right swap
of the prediction fixes (inversions), and the mean OKS. Predictions come from pose.py --save-preds, in the order of the
held-out annotations (one prediction per annotated person).
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

ANN = Path(__file__).resolve().parent.parent / "starter/data/annotations/specialization"
SIG = np.array([.026, .025, .025, .035, .035, .079, .079, .072, .072, .062, .062, .107, .107, .087, .087, .089, .089])
FLIP = [(3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]


def oks(p, g, v, area):
    e = ((p - g) ** 2).sum(-1) / (2 * max(area, 1) * (2 * SIG) ** 2)
    m = v > 0
    return float(np.exp(-e)[m].mean()) if m.any() else np.nan


def anatomy(fold, preds):
    d = json.load(open(ANN / f"train_{fold}.json"))
    ims = {im["id"] for im in d["images"]}
    anns = [a for a in d["annotations"] if a.get("num_keypoints", 0) > 0 and a["image_id"] in ims]
    assert len(anns) == len(preds), (len(anns), len(preds))
    o, osw = [], []
    for a, p in zip(anns, preds):
        assert a["image_id"] == p["image_id"]
        g = np.array(a["keypoints"], float).reshape(17, 3); q = np.array(p["keypoints"], float).reshape(17, 3)[:, :2]
        o.append(oks(q, g[:, :2], g[:, 2], a["area"]))
        qs = q.copy()
        for x, y in FLIP:
            qs[[x, y]] = qs[[y, x]]
        osw.append(oks(qs, g[:, :2], g[:, 2], a["area"]))
    o, osw = np.array(o), np.array(osw)
    fail = o < 0.5
    return {"fail%": 100 * fail.mean(), "inv%": 100 * (fail & (osw >= 0.5)).mean(), "meanOKS": np.nanmean(o)}


def main():
    runs, logs = Path(sys.argv[1]), Path(sys.argv[2])
    print(f"{'run':18s} {'AP':>7s} {'AP50':>7s} | {'OKS<.5 %':>8s} {'inverted %':>10s} {'mean OKS':>8s}")
    for name in sys.argv[3:]:
        fold = name.split("_")[1]
        ev = [l for l in open(logs / f"{name}.log") if l.startswith("EVAL")]
        if not ev:
            print(f"{name:18s} (no eval yet)"); continue
        m = re.search(rf"{fold}/no_ears AP ([0-9.]+) AP50 ([0-9.]+)", ev[-1])
        last = sorted((runs / name).glob(f"preds_{fold}_ep*.json"))[-1]
        a = anatomy(fold, json.load(open(last)))
        print(f"{name:18s} {float(m.group(1)):7.4f} {float(m.group(2)):7.4f} | {a['fail%']:8.2f} {a['inv%']:10.2f} {a['meanOKS']:8.4f}")


if __name__ == "__main__":
    main()
