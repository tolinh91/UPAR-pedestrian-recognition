"""Mimic the ingestion program on val images: import run.py, call load_model / predict_batch, score.

python par/test_submission.py ../submissions/dinov3_b16_v1 --n 2000
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from upar.data import DATA, attr_names, load_split  # noqa: E402
from upar.metrics import challenge_metrics, fmt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sub")
    ap.add_argument("--n", type=int, default=0, help="random subset of val (0 = all)")
    args = ap.parse_args()

    sub = Path(args.sub).resolve()
    spec = importlib.util.spec_from_file_location("run", sub / "run.py")
    run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)

    val = load_split("val")
    if args.n:
        val = val.sample(args.n, random_state=0)
    names = attr_names()
    samples = [{"index": i, "image": p, "image_path": str(DATA / p), "attribute_names": names}
               for i, p in enumerate(val.image)]
    t0 = time.time(); run.load_model(); t1 = time.time()
    bs = getattr(run, "BATCH_SIZE", 32)
    out = []
    for i in range(0, len(samples), bs):
        out += run.predict_batch(samples[i:i + bs])
    t2 = time.time()
    prob = np.array(out)
    assert prob.shape == (len(samples), 40) and np.isfinite(prob).all() and prob.min() >= 0 and prob.max() <= 1
    print(f"load {t1 - t0:.1f}s, predict {t2 - t1:.1f}s ({(t2 - t1) / len(samples) * 1000:.2f} ms/img)")
    m, per = challenge_metrics(val[names].values, prob, val.ds.values)
    print("VAL", fmt(m))
    for d, pm in per.items():
        print(f"  {d:11s} {fmt(pm)}")


if __name__ == "__main__":
    main()
