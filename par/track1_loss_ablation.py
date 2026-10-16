"""Matched, low-cost Track 1 loss ablation on the PETA held-out domain.

From the Track 1 directory:
  python par/track1_loss_ablation.py prepare
  python par/track1_loss_ablation.py run
  python par/track1_loss_ablation.py score

The three runs share the same training images, pretrained checkpoint, seed,
optimizer, augmentation and one-epoch budget. All scoring uses the same
source-domain prior and preselected threshold rule, tau=0.42. This is a
screening experiment, not an unbiased estimate of leaderboard improvement.
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "track1_loss_ablation_seed2027_rows.npy"
INIT = ROOT / "runs/ft_PETA_dinov3_q/best.pt"
OUTPUT = ROOT / "runs/track1_loss_ablation_2027"
EXPERIMENTS = {
    "bce_control": [],
    "soft_metric": ["--soft-metric", "0.1", "--soft-metric-tau", "0.42"],
    "attribute_rank": ["--attr-rank", "0.1"],
}


def frames():
    base = ROOT / "UPAR-Challenge-2027/data/annotations/task1"
    return pd.read_csv(base / "train/gt.csv"), pd.read_csv(base / "val/gt.csv")


def source_domain(row_names):
    return np.array([str(path).split("/")[0] for path in row_names])


def prepare(max_per_domain):
    train, _ = frames()
    domain = source_domain(train.iloc[:, 0].values)
    rng = np.random.default_rng(2027)
    chosen = []
    for name in ("Market1501", "PA100k"):
        available = np.flatnonzero(domain == name)
        if len(available) == 0:
            raise ValueError(f"Missing source domain: {name}")
        picked = rng.choice(available, size=min(len(available), max_per_domain), replace=False)
        chosen.append(picked)
        print(f"{name}: {len(picked)} sampled from {len(available)}")
    rows = np.sort(np.concatenate(chosen))
    np.save(ROWS, rows)
    print(f"Saved {len(rows)} source-domain row indices to {ROWS}")


def run(epochs, bs, workers):
    if not INIT.exists():
        raise FileNotFoundError(f"Initial checkpoint missing: {INIT}")
    if not ROWS.exists():
        raise FileNotFoundError(f"First run: python {__file__} prepare")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, add_flags in EXPERIMENTS.items():
        out = OUTPUT / name
        if (out / "best.pt").exists() and (out / "val_logits.npz").exists():
            print(f"Skip completed experiment: {name}", flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, str(ROOT / "par/train.py"),
            "--backbone", "vit_base_patch16_dinov3.lvd1689m",
            "--size", "256", "128",
            "--head", "query", "--head-kw", "n_layers=1",
            "--loss", "bce", "--loss-kw", "smooth=0.05", "weighting=deepmar",
            "--holdout", "PETA", "--train-rows", str(ROWS),
            "--init-checkpoint", str(INIT),
            "--seed", "2027", "--epochs", str(epochs), "--bs", str(bs),
            "--workers", str(workers), "--prefetch", "2",
            "--lr", "0.000005", "--ema", "0.99",
            "--out", str(out),
            *add_flags,
        ]
        print(f"Starting {name}: {' '.join(command)}", flush=True)
        with (out / "train.log").open("w") as logfile:
            process = subprocess.run(command, cwd=ROOT / "par", stdout=logfile, stderr=subprocess.STDOUT)
        if process.returncode:
            raise RuntimeError(f"{name} failed (exit {process.returncode}); see {out / 'train.log'}")
        print(f"Completed {name}; log: {out / 'train.log'}", flush=True)
    score()


def score():
    from upar.metrics import challenge_metrics

    train, val = frames()
    attrs = list(val.columns[1:41])
    train_domain = source_domain(train.iloc[:, 0].values)
    val_domain = source_domain(val.iloc[:, 0].values)
    p = train.loc[train_domain != "PETA", attrs].mean().to_numpy().clip(1e-3, 1 - 1e-3)
    theory = np.log((1 - p) / p) - (1 - 2 * p)
    results = []
    for name, path in [("initial", ROOT / "runs/ft_PETA_dinov3_q/val_logits.npz"), *[
        (key, OUTPUT / key / "val_logits.npz") for key in EXPERIMENTS
    ]]:
        if not path.exists():
            print(f"Missing logits, skip {name}: {path}")
            continue
        with np.load(path) as item:
            z, idx = item["logits"], item["idx"]
        if not np.all(val_domain[idx] == "PETA"):
            raise ValueError(f"{name} includes non-PETA validation examples")
        if z.shape != (len(idx), len(attrs)):
            raise ValueError(f"{name} has unexpected logit shape {z.shape}")
        gt = val[attrs].values[idx]
        logits = np.clip(z + 0.42 * theory, -50, 50)
        metrics, _ = challenge_metrics(gt, 1 / (1 + np.exp(-logits)), val_domain[idx])
        results.append({"run": name, "rows": len(idx), **metrics})
        print(f"{name:16} Challenge_Avg={metrics['Challenge_Avg']:.6f} "
              f"mA={metrics['mA']:.6f} F1={metrics['Instance_F1']:.6f}")
    if results:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / "results.json").write_text(json.dumps(results, indent=2))
        with (OUTPUT / "results.csv").open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved comparison to {OUTPUT / 'results.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "score"))
    parser.add_argument("--max-per-domain", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.max_per_domain)
    elif args.action == "run":
        run(args.epochs, args.bs, args.workers)
    else:
        score()


if __name__ == "__main__":
    main()
