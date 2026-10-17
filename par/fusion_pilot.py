"""Fusion pilot: does a per-image x per-attribute gate between the H+ and CLIP query embeddings help on an unseen
dataset, compared with the plain logit mean and with capacity-matched static / concatenation residuals?

All variants share z0 = (z_H + z_C) / 2 (the deployed T1 ensemble: no flip, threshold 0.5, no offset) and add a
residual r learned on frozen features:  z = z0 + r.
  L0  r = 0
  L1  r_a = (sigmoid(t_a) - 1/2) (z_H,a - z_C,a)                  static logit weights per attribute (t_a = 0 at start)
  F0  r_a = w_a . MLP([u_H,a, u_C,a])                              concatenation, no gate
  F1  r_a = w_a . (g_a u_H,a + (1 - g_a) u_C,a),  g_a = sigmoid(c_a) static gate per attribute
  F2  r_a = w_a . (g u_H,a + (1 - g) u_C,a),  g = sigmoid(gate([u_H,a, u_C,a, e_a]))   per image x attribute gate
  FH / FC  r_a = w_a . u_H,a  /  w_a . u_C,a   single-branch controls (added after the first fold: the head also sees
           labels the members never saw, so a gain over L0 may come from the extra labelled images, not from fusion)
  u_H = P_H q_H, u_C = P_C q_C: shared linear projections to d dims, no bias. w_a starts at 0 (r = 0 at start); no
  output bias, e_a only enters the gate.
Loss (all learned variants): the members' own training loss, logit_adj tau 0.32 + DeepMAR + smoothing 0.05 with the
training-split prior of the seen datasets, masked by the unannotated cells; inference uses the raw z.

Data (LODO fold = held-out dataset h; members trained on the TRAIN split of the other two datasets only):
  head training   val split of the two seen datasets (images the members never trained on), 85 % (inner train)
  inner check     the other 15 % (sanity only, nothing is selected on it)
  scoring         val split of h, used once at the end; no epoch / hyper-parameter selection anywhere.
Hyper-parameters are fixed in HP below before any held-out score was seen.

Mechanism checks on the held-out set: F2 with its gate replaced by the per-attribute mean gate of the training rows
(gate_mean) or shuffled across images (gate_shuffle); every learned variant with r computed from another image's
features (feat_shuffle: if the score survives, r is a per-attribute constant, i.e. a hidden offset).

python fusion_pilot.py --feats /dev/shm/feats --fold PETA --bb-seed 1 --head-seeds 1 2 --out ../fusion/PETA_bb1.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from upar.data import attr_names, invalid_mask, load_split
from upar.losses import build_loss
from upar.metrics import challenge_metrics

HP = dict(d=64, d_e=16, gate_hidden=32, f0_hidden=28, drop=0.1, lr=1e-3, wd=0.05, bs=256, epochs=10, warmup=0.05,
          tau=0.32, smooth=0.05, ds_alpha=0.5, inner_frac=0.15)
KEYS = ["Challenge_Avg", "mA", "Instance_F1", "Instance_Prec", "Instance_Rec"]


class Fusion(nn.Module):
    def __init__(self, kind, DH, DC, A, hp):
        super().__init__()
        self.kind, d = kind, hp["d"]
        if kind == "L1":
            self.t = nn.Parameter(torch.zeros(A))
        if kind in ("FH", "FC", "F0", "F1", "F2"):
            self.PH, self.PC = nn.Linear(DH, d, bias=False), nn.Linear(DC, d, bias=False)
            self.w = nn.Parameter(torch.zeros(A, d))
            self.drop = nn.Dropout(hp["drop"])
        if kind == "F0":
            self.mlp = nn.Sequential(nn.Linear(2 * d, hp["f0_hidden"]), nn.GELU(), nn.Linear(hp["f0_hidden"], d, bias=False))
        if kind == "F1":
            self.c = nn.Parameter(torch.zeros(A))
        if kind == "F2":
            self.e = nn.Parameter(torch.randn(A, hp["d_e"]) * 0.02)
            self.gate = nn.Sequential(nn.Linear(2 * d + hp["d_e"], hp["gate_hidden"]), nn.GELU(), nn.Linear(hp["gate_hidden"], 1))
            nn.init.zeros_(self.gate[2].weight); nn.init.zeros_(self.gate[2].bias)   # g = 1/2 at start

    def forward(self, qH, qC, zH, zC, gate_override=None, feat_perm=None):
        """gate_override: (A,) gate used for every image; feat_perm: r of row i computed from row feat_perm[i]."""
        z0 = (zH + zC) / 2
        if self.kind == "L0":
            return z0, None
        if self.kind == "L1":
            dz = zH - zC if feat_perm is None else (zH - zC)[feat_perm]
            return z0 + (torch.sigmoid(self.t) - 0.5) * dz, None
        if feat_perm is not None:
            qH, qC = qH[feat_perm], qC[feat_perm]
        uH, uC = self.drop(self.PH(qH.float())), self.drop(self.PC(qC.float()))
        g = None
        if self.kind in ("FH", "FC"):   # single-branch control: same data and loss, one member's features only
            h = uH if self.kind == "FH" else uC
        elif self.kind == "F0":
            h = self.mlp(torch.cat([uH, uC], -1))
        else:
            if self.kind == "F1":
                g = torch.sigmoid(self.c)[None, :, None].expand(len(uH), -1, 1)
            else:
                e = self.e[None].expand(len(uH), -1, -1)
                g = torch.sigmoid(self.gate(torch.cat([uH, uC, e], -1)))
            if gate_override is not None:
                g = gate_override[None, :, None].expand(len(uH), -1, 1)
            h = g * uH + (1 - g) * uC
        return z0 + (h * self.w).sum(-1), (None if g is None else g[..., 0])


@torch.no_grad()
def predict(m, F, rows, bs=1024, **kw):
    m.eval()
    zs, gs = [], []
    perm = kw.pop("feat_perm", None)
    for i in range(0, len(rows), bs):
        b = rows[i:i + bs]
        if perm is None:
            z, g = m(F["qH"][b], F["qC"][b], F["zH"][b], F["zC"][b], **kw)
        else:   # r from another image of the same evaluation set
            pb = perm[i:i + bs]
            zp, g = m(F["qH"][pb], F["qC"][pb], F["zH"][pb], F["zC"][pb], **kw)
            z = (F["zH"][b] + F["zC"][b]) / 2 + (zp - (F["zH"][pb] + F["zC"][pb]) / 2)
        zs.append(z.float()); gs.append(g)
    return torch.cat(zs).cpu().numpy(), (None if gs[0] is None else torch.cat(gs).cpu().numpy())


def train(kind, seed, F, tr_rows, pw, loss_fn, dims):
    torch.manual_seed(seed)
    m = Fusion(kind, *dims, HP).cuda()
    if kind == "L0":
        return m, 0
    n_par = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=HP["lr"], weight_decay=HP["wd"])
    steps = HP["epochs"] * len(tr_rows) // HP["bs"]
    warm = int(HP["warmup"] * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, steps - warm))))
    tr_t = torch.as_tensor(tr_rows, device="cuda")
    pw_t = torch.as_tensor(pw, device="cuda", dtype=torch.float32)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    m.train()
    for s in range(steps):
        b = tr_t[torch.multinomial(pw_t, HP["bs"], replacement=True, generator=gen)]
        z, _ = m(F["qH"][b], F["qC"][b], F["zH"][b], F["zC"][b])
        loss = loss_fn(z, F["y"][b], F["v"][b])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    return m, n_par


def scores(gt, z, grp):
    m, per = challenge_metrics(gt, 1 / (1 + np.exp(-z)), grp)
    return {k: round(float(m[k]), 5) for k in KEYS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="/dev/shm/feats")
    ap.add_argument("--fold", required=True, choices=["PETA", "Market1501", "PA100k"])
    ap.add_argument("--bb-seed", type=int, default=1)
    ap.add_argument("--head-seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    h = a.fold
    A = attr_names()
    va, tr = load_split("val"), load_split("train")
    inv = invalid_mask(tr)
    gt, grp = va[A].values.astype(np.float32), va.ds.values
    valid = np.ones_like(gt)
    for d, msk in inv.items():
        valid[grp == d] = ~msk
    trs = tr[tr.ds != h]
    tv = np.ones((len(trs), len(A)), np.float32)
    for d, msk in inv.items():
        tv[trs.ds.values == d] = ~msk
    prior = (trs[A].values * tv).sum(0) / tv.sum(0).clip(1)      # = train.py build_data prior of this fold

    tag = {"PETA": "PETA", "Market1501": "Market", "PA100k": "PA100k"}[h]
    pre = {k: Path(a.feats) / f"{k}32_{tag}_s{a.bb_seed}" for k in ("HL", "TL")}
    nz = {k: np.load(str(p) + ".npz") for k, p in pre.items()}
    for k in nz:
        assert str(nz[k]["holdout"]) == h, (k, nz[k]["holdout"])
    F = {"qH": torch.from_numpy(np.load(str(pre["HL"]) + "_q.npy")).cuda(),
         "qC": torch.from_numpy(np.load(str(pre["TL"]) + "_q.npy")).cuda(),
         "zH": torch.from_numpy(nz["HL"]["logits"]).cuda(), "zC": torch.from_numpy(nz["TL"]["logits"]).cuda(),
         "y": torch.from_numpy(gt).cuda(), "v": torch.from_numpy(valid).cuda()}
    dims = (F["qH"].shape[-1], F["qC"].shape[-1], len(A))

    held = np.where(grp == h)[0]
    seen = np.where(grp != h)[0]
    rs = np.random.default_rng(12345)
    inner = np.sort(rs.choice(seen, int(HP["inner_frac"] * len(seen)), replace=False))
    tr_rows = np.setdiff1d(seen, inner)
    names, counts = np.unique(grp[tr_rows], return_counts=True)
    per = {n: c ** HP["ds_alpha"] / c for n, c in zip(names, counts)}
    pw = np.array([per[d] for d in grp[tr_rows]]); pw /= pw.sum()
    loss_fn = build_loss("logit_adj", prior, smooth=HP["smooth"], weighting="deepmar", tau=HP["tau"]).cuda()

    # ---- diagnostics from the members' logits alone (held-out set)
    zH, zC = nz["HL"]["logits"][held], nz["TL"]["logits"][held]
    y, v = gt[held], valid[held].astype(bool)
    dH, dC = zH > 0, zC > 0
    okH, okC = (dH == (y > 0.5)), (dC == (y > 0.5))
    oracle = np.where(okH, zH, np.where(okC, zC, (zH + zC) / 2))          # pick the correct branch when one is correct
    diag = {"H_alone": scores(y, zH, grp[held]), "C_alone": scores(y, zC, grp[held]),
            "L0_mean": scores(y, (zH + zC) / 2, grp[held]), "oracle_select": scores(y, oracle, grp[held]),
            "disagree_rate": float((dH != dC)[v].mean()), "both_wrong_rate": float((~okH & ~okC)[v].mean()),
            "mean_wrong_rate": float((((zH + zC) / 2 > 0) != (y > 0.5))[v].mean())}
    per_attr = {A[k]: {"disagree": float((dH != dC)[:, k][v[:, k]].mean()) if v[:, k].any() else None,
                       "both_wrong": float((~okH & ~okC)[:, k][v[:, k]].mean()) if v[:, k].any() else None,
                       "H_better": float((okH & ~okC)[:, k][v[:, k]].mean()) if v[:, k].any() else None,
                       "C_better": float((~okH & okC)[:, k][v[:, k]].mean()) if v[:, k].any() else None}
                for k in range(len(A))}

    res = {"fold": h, "bb_seed": a.bb_seed, "hp": HP, "n_train": int(len(tr_rows)), "n_inner": int(len(inner)),
           "n_held": int(len(held)), "diag": diag, "per_attr": per_attr, "runs": {}}
    rp = np.random.default_rng(777).permutation(len(held))
    for kind in ["L0", "L1", "FH", "FC", "F0", "F1", "F2"]:
        for seed in ([0] if kind == "L0" else a.head_seeds):
            m, n_par = train(kind, seed, F, tr_rows, pw, loss_fn, dims)
            zh, g = predict(m, F, held)
            zi, _ = predict(m, F, inner)
            r = {"params": n_par, "held": scores(y, zh, grp[held]), "inner": scores(gt[inner], zi, grp[inner]),
                 "pos_per_img": float((zh > 0).sum(1).mean())}
            if kind != "L0":
                z0 = (zH + zC) / 2
                rr = zh - z0
                r["r_abs_mean"] = float(np.abs(rr).mean())
                r["r_const_share"] = float((rr.mean(0) ** 2).sum() / ((rr ** 2).mean(0).sum() + 1e-9))  # share of E[r^2] that is a per-attribute constant
                zf, _ = predict(m, F, held, feat_perm=held[rp])
                r["feat_shuffle"] = scores(y, zf, grp[held])
                r["changed_decisions"] = float(((zh > 0) != (z0 > 0)).mean())
            if kind == "F2":
                _, gtr = predict(m, F, tr_rows)
                gmean = torch.as_tensor(gtr.mean(0), device="cuda")
                zm, _ = predict(m, F, held, gate_override=gmean)
                r["gate_mean"] = scores(y, zm, grp[held])
                r["gate_std_held"] = float(g.std(0).mean())
                # gate of another image, own features
                gsh = torch.as_tensor(g[rp], device="cuda")
                zs = []
                with torch.no_grad():
                    m.eval()
                    for i in range(0, len(held), 1024):
                        b = held[i:i + 1024]
                        uH, uC = m.PH(F["qH"][b].float()), m.PC(F["qC"][b].float())
                        gg = gsh[i:i + 1024][..., None]
                        zz = (F["zH"][b] + F["zC"][b]) / 2 + ((gg * uH + (1 - gg) * uC) * m.w).sum(-1)
                        zs.append(zz.float())
                r["gate_shuffle"] = scores(y, torch.cat(zs).cpu().numpy(), grp[held])
            res["runs"][f"{kind}_h{seed}"] = r
            print(f"{h} bb{a.bb_seed} {kind} h{seed} params {n_par:6d}  held " +
                  " ".join(f"{k[:6]} {r['held'][k]:.4f}" for k in KEYS) + f"  inner {r['inner']['Challenge_Avg']:.4f}" +
                  (f"  featshuf {r['feat_shuffle']['Challenge_Avg']:.4f}" if "feat_shuffle" in r else "") +
                  (f"  gmean {r['gate_mean']['Challenge_Avg']:.4f} gshuf {r['gate_shuffle']['Challenge_Avg']:.4f}"
                   if "gate_mean" in r else ""), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("diag", json.dumps(diag), flush=True)


if __name__ == "__main__":
    main()
