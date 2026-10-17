"""H+ re-ranking embeddings (L2-normalised mean patch token, exactly as the T2 package computes them) for the PETA and
Market1501 val images, for the kNN slides of the T2 v12 deck. Runs on the training machine from /workspace/T1/par.

python t2_emb.py PKG_DIR OUT_NPZ
PKG_DIR: an unpacked GPU package whose hp.pt is the T2 v12 H+ member (the v25 T1 package has the identical file) with
run.py = run_rerank.py. Writes feats (N, 1280) float16, rows (val row ids) and ds.
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import torch

from upar.data import image_cache, load_split

pkg, out = Path(sys.argv[1]), sys.argv[2]
va = load_split("val")
rows = np.concatenate([np.where(va.ds.values == d)[0] for d in ("PETA", "Market1501")])
img_dir = Path("/workspace/T1/pkgtest/img"); img_dir.mkdir(parents=True, exist_ok=True)
buf, off = image_cache("val")
paths = []
for j in rows:
    p = img_dir / f"{j}.jpg"
    if not p.exists():
        p.write_bytes(buf[off[j]:off[j + 1]].tobytes())
    paths.append(str(p))
for name in [k for k in list(sys.modules) if k == "run" or k.startswith(("upar", "timm"))]:
    del sys.modules[name]
sys.path.insert(0, str(pkg))
run = importlib.import_module("run")
run.load_model()
j = [m["file"] for m in run._CFG["models"]].index("hp.pt")
feats = []
with torch.no_grad():
    for i in range(0, len(paths), 128):
        _, f = run._batch_logits([{"image_path": p} for p in paths[i:i + 128]], feat_member=j)
        feats.append(f.half().cpu().numpy())
feats = np.concatenate(feats)
np.savez(out, feats=feats, rows=rows, ds=va.ds.values[rows])
print(out, feats.shape, "norm", float(np.linalg.norm(feats[:5].astype(np.float32), axis=1).mean()))
