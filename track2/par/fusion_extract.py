"""Cache the per-attribute query embeddings q and the final logits of one trained member on the whole val split,
for the fusion pilot (fusion_pilot.py). Same inference as the T1 package: no flip, bf16 autocast, raw logits.

python fusion_extract.py ../runs/HL32_PETA_s1 /dev/shm/feats/HL32_PETA_s1
  -> <out>_q.npy   (N_val, 40, D) float16   q after the head's out_norm (the vector its classifier reads)
     <out>.npz     logits (N_val, 40) float32 (query_text: query logit + language-anchor logit, as deployed),
                   qlogit (N_val, 40) float32 (query part only), idx (N_val,), holdout, run
Checks: logits == q . fc_weight + fc_bias (+ text logit) up to bf16 rounding.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from upar.data import UPARDataset, build_transforms
from upar.models import build_model, cfg_from_checkpoint


@torch.no_grad()
def main(run, out):
    run, out = Path(run), Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ck = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
    cfg = cfg_from_checkpoint(ck)
    model = build_model(cfg, pretrained=False).cuda().eval()
    model.load_state_dict(ck["model"])
    head = model.head
    norm = {k: cfg[k] for k in ("mean", "std") if cfg.get(k)}
    ds = UPARDataset("val", build_transforms(tuple(cfg["size"]), False, **norm))
    dl = DataLoader(ds, 256, num_workers=16, pin_memory=True)
    N, A, D = len(ds), head.queries.shape[0], head.queries.shape[1]
    Q = np.lib.format.open_memmap(str(out) + "_q.npy", mode="w+", dtype=np.float16, shape=(N, A, D))
    Z, QZ, IDX = np.zeros((N, A), np.float32), np.zeros((N, A), np.float32), np.zeros(N, np.int64)
    n, err = 0, 0.0
    for x, _, _, j, _ in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z, _, q = model(x.cuda(non_blocking=True), return_aux=True)
        z, q = z.float(), q.float()
        qz = (q * head.fc_weight.float()).sum(-1) + head.fc_bias.float()
        if not hasattr(head, "text_logits"):
            err = max(err, (z - qz).abs().max().item())
        b = len(j)
        Q[n:n + b] = q.cpu().numpy().astype(np.float16)
        Z[n:n + b], QZ[n:n + b], IDX[n:n + b] = z.cpu().numpy(), qz.cpu().numpy(), j.numpy()
        n += b
    Q.flush()
    assert n == N and (IDX == np.arange(N)).all()
    np.savez(str(out) + ".npz", logits=Z, qlogit=QZ, idx=IDX, holdout=str(ck["args"]["holdout"]), run=str(run))
    print(f"{run.name}: {N} images, q {Q.shape}, max |logit - q.w - b| (query head only) {err:.4f}, "
          f"text part mean |.| {np.abs(Z - QZ).mean():.3f}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
