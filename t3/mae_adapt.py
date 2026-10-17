"""Human-centric MAE continued pretraining of the ViTPose+ backbone (the Sapiens recipe, on allowed data only).

python mae_adapt.py --model usyd-community/vitpose-plus-large --datasets PA100k Market1501 --epochs 8 \
       --out ../runs/mae_l_noPETA
Images: the person crops of the challenge's own training data (Track 1 gt.csv lists, train + val), never the test
domain; for a leave-PETA-out check PETA is excluded too. 75% of the 16x12 patches are masked, the backbone sees the
rest (COCO expert of the MoE), a 4-layer decoder reconstructs the per-patch-normalised pixels of the masked patches.
Writes backbone.pt (state dict of model.m.backbone) for pose.py --init-backbone.
"""
import argparse, math, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import pose

UPAR = Path("/workspace/work/UPAR/data")
H, W, P = 256, 192, 16


class Crops(Dataset):
    def __init__(self, datasets):
        rows = pd.concat([pd.read_csv(UPAR / f"annotations/task1/{s}/gt.csv", usecols=[0]) for s in ("train", "val")])
        paths = rows.iloc[:, 0].astype(str)
        self.paths = [p for p in paths if p.split("/")[0] in datasets]
        self.mean = torch.tensor(pose.MEAN / 255).view(3, 1, 1); self.std = torch.tensor(pose.STD / 255).view(3, 1, 1)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(UPAR / self.paths[i]).convert("RGB")
        w, h = im.size
        s = np.random.uniform(0.7, 1.0)                                   # random resized crop, aspect of the input
        ch, cw = int(h * s), int(min(w, h * s * W / H))
        y0, x0 = np.random.randint(0, h - ch + 1), np.random.randint(0, max(w - cw, 0) + 1)
        im = im.crop((x0, y0, x0 + cw, y0 + ch))
        canvas = Image.new("RGB", (int(ch * W / H), ch))                  # letterbox to 3:4 like the pose crops
        canvas.paste(im, ((canvas.size[0] - im.size[0]) // 2, 0))
        x = torch.from_numpy(np.asarray(canvas.resize((W, H), Image.BILINEAR), np.float32) / 255).permute(2, 0, 1)
        if np.random.rand() < 0.5:
            x = x.flip(-1)
        return (x - self.mean) / self.std


class MAE(nn.Module):
    def __init__(self, backbone, dim, ddim=512, depth=4, mask=0.75):
        super().__init__()
        self.bb, self.mask = backbone, mask
        self.n = (H // P) * (W // P)
        self.enc2dec = nn.Linear(dim, ddim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, ddim))
        self.dpos = nn.Parameter(torch.randn(1, self.n, ddim) * 0.02)
        self.dec = nn.TransformerEncoder(nn.TransformerEncoderLayer(ddim, 8, ddim * 4, 0.0, activation="gelu",
                                                                    batch_first=True, norm_first=True), depth)
        self.pred = nn.Linear(ddim, P * P * 3)

    def forward(self, x):
        B = x.shape[0]
        emb = self.bb.embeddings(x)                                       # (B, N, C) with positions
        keep = int(self.n * (1 - self.mask))
        order = torch.rand(B, self.n, device=x.device).argsort(1)
        vis, hid = order[:, :keep], order[:, keep:]
        tok = torch.gather(emb, 1, vis[..., None].expand(-1, -1, emb.shape[-1]))
        idx = torch.zeros(B, dtype=torch.long, device=x.device)
        h = self.bb.encoder(tok, dataset_index=idx).last_hidden_state
        h = self.enc2dec(self.bb.layernorm(h))
        full = self.mask_token.expand(B, self.n, -1).to(h.dtype).clone()
        full.scatter_(1, vis[..., None].expand(-1, -1, h.shape[-1]), h)
        out = self.pred(self.dec(full + self.dpos))                        # (B, N, P*P*3)
        tgt = x.unfold(2, P, P).unfold(3, P, P).permute(0, 2, 3, 1, 4, 5).reshape(B, self.n, -1)
        tgt = (tgt - tgt.mean(-1, keepdim=True)) / (tgt.var(-1, keepdim=True) + 1e-6).sqrt()
        sel = lambda t: torch.gather(t, 1, hid[..., None].expand(-1, -1, t.shape[-1]))
        return F.mse_loss(sel(out).float(), sel(tgt).float())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="usyd-community/vitpose-plus-large")
    ap.add_argument("--datasets", nargs="+", default=["PA100k", "Market1501", "PETA"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4, help="backbone lr (decoder x10)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    m = pose.ViTPose(a.model)                                             # applies the COCO-expert MoE patch
    bb = m.m.backbone
    mae = MAE(bb, bb.config.hidden_size).cuda()
    ds = Crops(a.datasets)
    dl = DataLoader(ds, a.bs, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True, persistent_workers=True)
    dec_params = [p for n, p in mae.named_parameters() if not n.startswith("bb.")]
    opt = torch.optim.AdamW([{"params": [p for p in bb.parameters() if p.requires_grad], "lr": a.lr},
                             {"params": dec_params, "lr": a.lr * 10}], weight_decay=0.05, betas=(0.9, 0.95))
    total = a.epochs * len(dl); warm = len(dl) // 2; base = [g["lr"] for g in opt.param_groups]
    print(f"{len(ds)} crops from {a.datasets}, {len(dl)} it/epoch", flush=True)
    step, t0 = 0, time.time()
    for ep in range(a.epochs):
        mae.train()
        for x in dl:
            f = step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / (total - warm)))
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * f
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = mae(x.cuda(non_blocking=True))
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(mae.parameters(), 1.0); opt.step(); step += 1
            if step % 100 == 0:
                print(f"ep {ep} step {step}/{total} loss {loss.item():.4f} {step * a.bs / (time.time() - t0):.0f} img/s", flush=True)
        torch.save(bb.state_dict(), out / "backbone.pt")
    print("done", flush=True)


if __name__ == "__main__":
    main()
