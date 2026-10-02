"""Error analysis of a leave-one-dataset-out proxy run on its held-out dataset (both tracks).

python error_analysis.py <run dir> [--out ../report/figs]
Reads seen_val_logits_last.npz / held_val_logits_last.npz written by offset_proxy.py.
"""
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image
from calibrate import fit_offsets
from eval_retrieval import load_task2
from upar.data import DATA, attr_names, load_split
from upar.retrieval import score_loglik

COLORS = ["Black", "Blue", "Brown", "Green", "Grey", "Orange", "Pink", "Purple", "Red", "White", "Yellow", "Other"]


def auc(y, s):
    """Mann-Whitney AUC (ties averaged)."""
    order = np.argsort(s); r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    r = np.bincount(inv, weights=r)[inv] / cnt[inv]
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--out", default="../report/figs")
    args = ap.parse_args(); run = Path(args.run); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    A = attr_names(); va, tr = load_split("val"), load_split("train"); GT, DS = va[A].values, va.ds.values
    held = json.load(open(run / "args.json"))["holdout"] if (run / "args.json").exists() else "PETA"
    s, h = np.load(run / "seen_val_logits_last.npz"), np.load(run / "held_val_logits_last.npz")
    off, _ = fit_offsets(s["logits"], GT[s["idx"]], DS[s["idx"]])
    gt, z = GT[h["idx"]].astype(int), h["logits"]; pred = (z + 0.6 * off) > 0
    prev_src = tr[tr.ds != held][A].values.mean(0)
    rep = {}

    print(f"\n=== 1. Label imbalance ({held}, {len(gt)} images) ===")
    rows = []
    for k, a in enumerate(A):
        y = gt[:, k]
        if y.sum() < 5:
            continue
        tp, fn = (pred[:, k] & (y == 1)).sum(), (~pred[:, k] & (y == 1)).sum()
        tn, fp = (~pred[:, k] & (y == 0)).sum(), (pred[:, k] & (y == 0)).sum()
        rows.append(dict(attr=a, k=k, prev_src=prev_src[k], prev=y.mean(), auc=auc(y, z[:, k]), recall=tp / (tp + fn),
                         spec=tn / (tn + fp), fn=int(fn), fp=int(fp), n_pos=int(y.sum())))
    rows.sort(key=lambda r: r["prev_src"])
    prev = np.array([r["prev_src"] for r in rows])
    for key in ("auc", "recall", "spec"):
        print(f"  Spearman(source prevalence, {key}) = {spearman(prev, np.array([r[key] for r in rows])):+.2f}")
    q = np.quantile(prev, [1 / 3, 2 / 3])
    for lo, hi, nm in ((0, q[0], "rare"), (q[0], q[1], "middle"), (q[1], 1, "common")):
        m = [r for r in rows if lo <= r["prev_src"] <= hi]
        print(f"  {nm:7s} third ({len(m)} attrs, {100*min(r['prev_src'] for r in m):.1f}-{100*max(r['prev_src'] for r in m):.1f}%): "
              f"AUC {np.mean([r['auc'] for r in m]):.3f}  recall {np.mean([r['recall'] for r in m]):.3f}  "
              f"specificity {np.mean([r['spec'] for r in m]):.3f}")
    print("  rarest 8:", ", ".join(f"{r['attr']} (src {100*r['prev_src']:.1f}%, AUC {r['auc']:.2f}, recall {r['recall']:.2f}, n={r['n_pos']})" for r in rows[:8]))
    rep["imbalance"] = rows

    print("\n=== 2. Where the errors are ===")
    tot = sum(r["fn"] + r["fp"] for r in rows)
    for r in sorted(rows, key=lambda r: -(r["fn"] + r["fp"]))[:10]:
        print(f"  {r['attr']:28s} errors {r['fn'] + r['fp']:5d} ({100*(r['fn'] + r['fp'])/tot:4.1f}%)  missed {r['fn']:4d}  false alarms {r['fp']:4d}")

    print("\n=== 3. Colour confusions (images with exactly one true colour, argmax of the 12 colour logits) ===")
    conf = {}
    for part, start in (("Upper", 8), ("Lower", 21)):
        idx = list(range(start, start + 12)); g = gt[:, idx]; one = g.sum(1) == 1
        t, pmax = g[one].argmax(1), z[one][:, idx].argmax(1)
        acc = (t == pmax).mean()
        M = np.zeros((12, 12), int); np.add.at(M, (t, pmax), 1)
        conf[part] = M.tolist()
        off_diag = [(M[i, j], COLORS[i], COLORS[j]) for i in range(12) for j in range(12) if i != j]
        top = sorted(off_diag, reverse=True)[:6]
        print(f"  {part} body: top-1 colour accuracy {acc:.3f} on {one.sum()} images | most frequent confusions (true -> predicted): "
              + ", ".join(f"{a}->{b} {n}" for n, a, b in top))
    rep["colour_confusion"] = conf

    print("\n=== 4. Image resolution ===")
    paths = va.image.values[h["idx"]]
    hgt = np.array([Image.open(DATA / p).size[1] for p in paths])
    inst_f1 = 2 * (pred & (gt == 1)).sum(1) / np.maximum(pred.sum(1) + gt.sum(1), 1)
    for lo, hi in ((0, 80), (80, 120), (120, 160), (160, 250), (250, 10000)):
        m = (hgt >= lo) & (hgt < hi)
        if m.sum():
            print(f"  height {lo:4d}-{hi if hi < 10000 else 'max':>4}: {m.sum():5d} images, mean instance F1 {inst_f1[m].mean():.3f}, "
                  f"errors/image {((pred != (gt == 1)).sum(1))[m].mean():.2f}")

    print("\n=== 5. Track 2 failures (log-likelihood ranking, raw probabilities) ===")
    ids, queries = load_task2("val"); qid_of = dict(zip(ids.image, ids.qid))
    gq = np.array([qid_of[p] for p in paths]); q_ids = np.unique(gq)
    prob = 1 / (1 + np.exp(-z)); sim = score_loglik(queries[q_ids], prob)
    first, top1_diff = [], np.zeros(40)
    for i, qi in enumerate(q_ids):
        order = np.argsort(-sim[i]); match = gq[order] == qi
        r1 = np.nonzero(match)[0][0] + 1; first.append(r1)
        if r1 > 1:                                   # attributes on which the top-ranked wrong image differs from the query
            top1_diff += (gt[order[0]] != queries[qi]).astype(float)
    first = np.array(first)
    print(f"  {len(q_ids)} queries: first correct match at rank 1 for {np.mean(first == 1):.1%}, within 10 for {np.mean(first <= 10):.1%}, "
          f"beyond 50 for {np.mean(first > 50):.1%}")
    n_fail = (first > 1).sum()
    order = np.argsort(-top1_diff)[:10]
    print(f"  in the {n_fail} queries whose top result is wrong, the top result differs from the query on "
          f"{top1_diff.sum() / n_fail:.2f} attributes on average, most often: " +
          ", ".join(f"{A[k]} {100*top1_diff[k]/n_fail:.0f}%" for k in order))
    rep["t2_first_rank"] = first.tolist()

    print("\n=== 6. Most confident errors: figure ===")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    margin = np.where(pred != (gt == 1), np.abs(z + 0.6 * off), 0)
    worst = np.argsort(-margin.max(1))[:8]
    fig, axs = plt.subplots(1, 8, figsize=(13, 3.6))
    for ax, i in zip(axs, worst):
        ax.imshow(Image.open(DATA / paths[i]).convert("RGB")); ax.axis("off")
        k = int(margin[i].argmax())
        ax.set_title(f"{A[k].replace('UpperBody-', 'Up ').replace('LowerBody-', 'Low ').replace('Accessory-', '')}\n"
                     f"thật {gt[i, k]}, đoán {int(pred[i, k])} (p={1/(1+np.exp(-(z[i, k] + 0.6*off[k]))):.2f})", fontsize=7)
    fig.tight_layout(); fig.savefig(out / f"errors_{held}.png", dpi=160); plt.close(fig)
    json.dump(rep, open(run / "error_analysis.json", "w"), default=float)
    print(f"  saved {out / f'errors_{held}.png'}")


if __name__ == "__main__":
    main()
