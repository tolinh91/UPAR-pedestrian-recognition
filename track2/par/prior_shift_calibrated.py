import sys, numpy as np, torch
sys.path.insert(0, ".")
from calibrate import fit_offsets
from upar.data import attr_names, load_split
from upar.metrics import challenge_metrics
from prior_shift import em_prior, sig
A = attr_names(); va = load_split("val")
for name in sys.argv[1:]:
    r = f"../runs/remote/{name}"
    s, h = np.load(f"{r}/seen_val_logits_last.npz"), np.load(f"{r}/held_val_logits_last.npz")
    sgt, sds = va[A].values[s["idx"]].astype(float), va.ds.values[s["idx"]]
    gt, ds = va[A].values[h["idx"]], va.ds.values[h["idx"]]
    # per-attribute Platt scaling on seen val, only datasets where the attribute is annotated
    K = len(A); a = np.ones(K); b = np.zeros(K); prior_s = np.full(K, 0.5)
    for k in range(K):
        m = np.zeros(len(sds), bool)
        for d in np.unique(sds):
            if sgt[sds == d, k].any(): m |= sds == d
        if m.sum() < 50: continue
        z = torch.tensor(s["logits"][m, k], dtype=torch.float64); y = torch.tensor(sgt[m, k])
        p = torch.tensor([1.0, 0.0], dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([p], max_iter=100)
        def cl():
            opt.zero_grad(); l = torch.nn.functional.binary_cross_entropy_with_logits(p[0] * z + p[1], y); l.backward(); return l
        opt.step(cl); a[k], b[k] = p.detach().numpy(); prior_s[k] = sgt[m, k].mean()
    prior_s = prior_s.clip(1e-3, 1 - 1e-3)
    zc = a * h["logits"] + b
    prior_t = em_prior(sig(zc), prior_s)
    delta = (np.log(prior_t / (1 - prior_t)) - np.log(prior_s / (1 - prior_s))) / a
    true_t = np.where(gt.any(0), gt.mean(0), np.nan)
    off, _ = fit_offsets(s["logits"], sgt, sds)
    f = lambda sh: challenge_metrics(gt, sig(h["logits"] + sh), ds)[0]["Challenge_Avg"]
    v = gt.any(0)
    print(f"{name:20s} src {f(0.6*off):.4f}  calEM x.5 {f(0.6*off+0.5*delta):.4f}  x1 {f(0.6*off+delta):.4f}  "
          f"| prior error |src-true| {np.abs(prior_s-true_t)[v].mean():.3f}  |EM-true| {np.abs(prior_t-true_t)[v].mean():.3f}", flush=True)
