"""Can an image-level facing classifier fix left/right inversions? Leave-one-dataset-out, CPU only.

python facing_probe.py "../runs/l_lodo_{name}"
Facing label from the ground-truth keypoints: the person faces the camera when their left shoulder / hip lies to the
right of the right one in the image (x_left - x_right > 0), faces away when it is clearly negative. A logistic probe on
the large model's tokens (deep_<name>.npz: G = mean token, K = mean token at the predicted keypoints) is trained on
two datasets (held-out model features) and applied to the third. Where the probe is confident and the predicted
skeleton implies the opposite facing, left and right are swapped. Reported: swap precision and AP before / after.
"""
import contextlib, copy, io, json, sys
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from pose import ANN, SIGMAS
from rescore_feats import oks

NAMES = ("peta", "pa100k", "market")
pattern = sys.argv[1]
SWAP = list(range(17))
for a, b in [(3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]:
    SWAP[a], SWAP[b] = b, a
KEEP = [k for k in range(17) if k not in (3, 4)]


def facing(kp, vis, width):
    """+1 facing the camera, -1 facing away, 0 unclear: mean of x_left - x_right over shoulders and hips."""
    d = []
    for l, r in ((5, 6), (11, 12)):
        if vis[l] and vis[r]:
            d.append(kp[l, 0] - kp[r, 0])
    if not d:
        return 0
    m = np.mean(d) / max(width, 1.0)
    return 1 if m > 0.08 else -1 if m < -0.08 else 0


def load(name):
    z = dict(np.load(Path(pattern.format(name=name)) / f"deep_{name}.npz"))
    js = json.load(open(ANN / f"train_{name}.json"))
    anns = {a["id"]: a for a in js["annotations"]}
    gt_f, pr_f, gts, areas = [], [], [], []
    for aid, kpf in zip(z["ann_id"], z["keypoints"]):
        a = anns[int(aid)]; g = np.array(a["keypoints"], float).reshape(17, 3); p = kpf.reshape(17, 3)
        w = a["bbox"][2]
        gt_f.append(facing(g, g[:, 2] > 0, w)); pr_f.append(facing(p, np.ones(17, bool), w))
        gts.append(g); areas.append(a["area"])
    z.update(gt_f=np.array(gt_f), pr_f=np.array(pr_f), gt=np.array(gts), area=np.array(areas))
    return z


def ap(name, d, kps, score):
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(ANN / f"train_{name}.json"))
        preds = [{"image_id": int(i), "category_id": 1, "score": float(s), "keypoints": k.reshape(-1).tolist()}
                 for i, s, k in zip(d["image_id"], score, kps)]
        ev = COCOeval(gt, gt.loadRes(copy.deepcopy(preds)), "keypoints"); ev.params.kpt_oks_sigmas = SIGMAS
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return ev.stats[0]


D = {n: load(n) for n in NAMES}
for n, d in D.items():
    inv = np.array([oks(k.reshape(17, 3)[SWAP], g, a) - o > 0.2 for k, g, a, o in zip(d["keypoints"], d["gt"], d["area"], d["oks"])])
    d["inv"] = inv
    print(f"{n}: facing labels front {np.mean(d['gt_f'] == 1):.2f} back {np.mean(d['gt_f'] == -1):.2f} unclear {np.mean(d['gt_f'] == 0):.2f}"
          f" | inversions {inv.sum()} ({inv.mean():.2%}) | predicted facing disagrees with GT on {np.mean((d['pr_f'] * d['gt_f']) < 0):.2%}", flush=True)

for held in NAMES:
    tr = [D[n] for n in NAMES if n != held]; te = D[held]
    X = lambda d: np.c_[d["G"].astype(np.float32), d["K"].astype(np.float32)]
    Xtr = np.concatenate([X(d) for d in tr]); ytr = np.concatenate([d["gt_f"] for d in tr])
    m = ytr != 0
    sc = StandardScaler().fit(Xtr[m])
    clf = LogisticRegression(C=0.05, max_iter=2000).fit(sc.transform(Xtr[m]), ytr[m] > 0)
    pf = clf.predict_proba(sc.transform(X(te)))[:, 1]                    # P(facing the camera)
    clear = te["gt_f"] != 0
    acc = np.mean((pf[clear] > .5) == (te["gt_f"][clear] > 0))
    kp = te["keypoints"].reshape(-1, 17, 3); score = kp[:, KEEP, 2].mean(1)
    base = ap(held, te, kp, score)
    out = [f"{held}: probe facing accuracy {acc:.3f} (n={clear.sum()}), AP no-ear score {base:.4f}"]
    for t in (0.9, 0.95, 0.98, 0.99):
        conf_front, conf_back = pf > t, pf < 1 - t
        swap = ((conf_front & (te["pr_f"] < 0)) | (conf_back & (te["pr_f"] > 0)))
        kp2 = kp.copy(); kp2[swap] = kp2[swap][:, SWAP]
        prec = te["inv"][swap].mean() if swap.any() else float("nan")
        out.append(f"  t={t}: swap {swap.sum()} (true inversions {te['inv'][swap].sum()}, precision {prec:.2f}, "
                   f"recall {te['inv'][swap].sum() / max(te['inv'].sum(), 1):.2f}) AP {ap(held, te, kp2, score):.4f}")
    print("\n".join(out), flush=True)
