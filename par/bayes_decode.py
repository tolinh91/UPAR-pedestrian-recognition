"""Bayes decoding for the challenge metric Avg = HM(mA, Instance_F1), applied per image.

Near M ~ F the harmonic mean has equal partial derivatives, so we maximise mA + IF1 to first order:
  mA part (additive over cells): predicting (i, k) positive adds E = [p/pi_k - (1-p)/(1-pi_k)] / (2 K N)
  IF1 part (per image):          E[F1_i(S)] ~ 2 sum_{k in S} p_ik / (|S| + sum_k p_ik)
For a fixed set size m the objective is additive in k: pick the m cells with the largest
  s_ik(m) = [p/pi_k - (1-p)/(1-pi_k)] / (2K) + 2 p_ik / (m + sum_k p_ik),
then keep the best m in 0..K. p must be calibrated: per-attribute Platt scaling fit on source-domain val logits.
pi_k: per-dataset source prior averaged over the source datasets (the metric averages datasets).
"""
import numpy as np


def platt_fit(z, y, valid, iters=50):
    """Per attribute a_k z + b_k by Newton on the logistic loss (valid cells only)."""
    K = z.shape[1]
    ab = np.tile([1.0, 0.0], (K, 1))
    for k in range(K):
        m = valid[:, k] > 0
        x, t = z[m, k], y[m, k]
        a, b = 1.0, 0.0
        for _ in range(iters):
            p = 1 / (1 + np.exp(-(a * x + b)))
            g = np.array([((p - t) * x).sum(), (p - t).sum()])
            w = p * (1 - p) + 1e-9
            H = np.array([[(w * x * x).sum(), (w * x).sum()], [(w * x).sum(), w.sum()]]) + 1e-6 * np.eye(2)
            step = np.linalg.solve(H, g)
            a, b = a - step[0], b - step[1]
            if np.abs(step).max() < 1e-7:
                break
        ab[k] = a, b
    return ab


def platt_apply(z, ab):
    return 1 / (1 + np.exp(-(z * ab[:, 0] + ab[:, 1])))


def bayes_decode(p, pi):
    """p: (N, K) calibrated probabilities, pi: (K,) priors -> (N, K) bool decisions."""
    N, K = p.shape
    pi = np.clip(pi, 1e-4, 1 - 1e-4)
    a = (p / pi - (1 - p) / (1 - pi)) / (2 * K)                 # mA utility per cell
    sp = p.sum(1, keepdims=True)
    best_val = np.full(N, -np.inf)
    best = np.zeros_like(p, dtype=bool)
    order_cache = None
    for m in range(0, K + 1):
        if m == 0:
            val = np.zeros(N)                                      # empty set: F1 = 0, no mA change
            sel = np.zeros_like(p, dtype=bool)
        else:
            s = a + 2 * p / (m + sp)
            idx = np.argpartition(-s, m - 1, axis=1)[:, :m]
            val = np.take_along_axis(s, idx, 1).sum(1)
            sel = np.zeros_like(p, dtype=bool)
            np.put_along_axis(sel, idx, True, 1)
        better = val > best_val
        best_val[better] = val[better]
        best[better] = sel[better]
    return best
