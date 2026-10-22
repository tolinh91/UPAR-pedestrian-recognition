"""Quick checks for a T2 v14 on the group-CE ensemble (compact.npz fold logits, numpy only)."""
import json, sys
from pathlib import Path
import numpy as np
src = open(Path(__file__).with_name("t2_gallery.py")).read().split("C = np.load(")[0]
sys.argv = ["x"]; exec(src)
C = np.load("/media/tan/F/RWS2027/remote_results/t2k/compact.npz")
stu = dict(zip(*[np.load("/media/tan/F/RWS2027/Upar-Challenge-Track-2/runs/remote2/kd2_a08_e10_PETA/held_val_logits_last.npz")[k] for k in ("idx", "logits")]))
for fold, (pre, hold) in FOLDS.items():
    rows = C[f"{fold}_rows"]; G = Gallery(hold, rows)
    hg, tg = C[f"{fold}_HG"].astype(np.float64), C[f"{fold}_TG"].astype(np.float64)
    I = C[f"{fold}_ft_mean_nn"].astype(np.int64)
    md = lambda z: 100 * G.metrics(z)["mADM"]  # noqa: E731
    r = {"gce pair": md((hg + tg) / 2)}
    if fold == "PETA":
        st = np.stack([stu[x] for x in rows]).astype(np.float64)
        r["gce pair + old student (1/3)"] = md((hg + tg + st) / 3)
        r["gce pair + student, knn k50 a.2"] = md(smooth((hg + tg + st) / 3, I, .2))
    for wh in (0.4, 0.6, 0.7):
        r[f"weights DINOv3 {wh}"] = md(wh * hg + (1 - wh) * tg)
    z = (hg + tg) / 2
    for k, a in ((50, .2), (20, .2), (50, .3), (20, .3), (50, .4), (10, .2)):
        r[f"knn k{k} a{a}"] = md(smooth(z, I[:, :k], a))
    print(fold, json.dumps({k: round(v, 2) for k, v in r.items()}), flush=True)
