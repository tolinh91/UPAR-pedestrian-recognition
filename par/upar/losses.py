"""Multi-label losses for PAR.

Every loss shares three options handled by BaseLoss:
  smooth     label smoothing: targets become smooth/2 and 1 - smooth/2
  weighting  per-attribute weights from the training prior p (fraction of positives)
               none     w+ = w- = 1
               deepmar  w+ = exp(1 - p),  w- = exp(p)             (mild, ratio <= e)
               sqrt     w+ = sqrt((1 - p) / p), w- = 1            (strong, capped at max_weight)
               inverse  w+ = (1 - p) / p,       w- = 1            (very strong, capped at max_weight)
  masking    cells whose label is not annotated in a dataset (valid = 0) contribute nothing

Subclasses only define elementwise(logits, targets) -> per-cell loss.

  bce        binary cross-entropy
  logit_adj  BCE on logits shifted by tau * log(p / (1 - p)) during training only. At test time the raw
             logit is used, which moves the decision towards balanced accuracy (the mA view).
  focal      focal loss, down-weights easy cells: (1 - p_t)^gamma * BCE
  asl        asymmetric loss (Ridnik et al. 2021): focusing gamma_pos / gamma_neg and probability
             shifting (clip) for negatives, designed for many-negative multi-label problems
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import Registry

LOSSES = Registry("loss")


class BaseLoss(nn.Module):
    def __init__(self, prior, smooth=0.0, weighting="none", max_weight=20.0):
        super().__init__()
        p = torch.as_tensor(prior, dtype=torch.float32).clamp(1e-4, 1 - 1e-4)
        if weighting == "none":
            w_pos, w_neg = torch.ones_like(p), torch.ones_like(p)
        elif weighting == "deepmar":
            w_pos, w_neg = torch.exp(1 - p), torch.exp(p)
        elif weighting == "sqrt":
            w_pos, w_neg = ((1 - p) / p).sqrt().clamp(max=max_weight), torch.ones_like(p)
        elif weighting == "inverse":
            w_pos, w_neg = ((1 - p) / p).clamp(max=max_weight), torch.ones_like(p)
        else:
            raise ValueError(f"unknown weighting '{weighting}'")
        self.register_buffer("prior", p)
        self.register_buffer("w_pos", w_pos)
        self.register_buffer("w_neg", w_neg)
        self.smooth = smooth

    def elementwise(self, logits, targets):
        raise NotImplementedError

    def forward(self, logits, labels, valid):
        """logits, labels, valid: (B, A). labels in {0, 1}; valid = 0 masks unannotated cells."""
        targets = labels * (1 - self.smooth) + self.smooth / 2
        w = torch.where(labels > 0.5, self.w_pos, self.w_neg) * valid
        return (self.elementwise(logits.float(), targets) * w).sum() / valid.sum().clamp(min=1)


@LOSSES.register("bce")
class BCELoss(BaseLoss):
    def elementwise(self, logits, targets):
        return F.binary_cross_entropy_with_logits(logits, targets, reduction="none")


@LOSSES.register("logit_adj")
class LogitAdjustedBCE(BaseLoss):
    def __init__(self, prior, tau=1.0, **kw):
        super().__init__(prior, **kw)
        self.tau = tau

    def elementwise(self, logits, targets):
        shift = self.tau * torch.log(self.prior / (1 - self.prior))
        return F.binary_cross_entropy_with_logits(logits + shift, targets, reduction="none")


@LOSSES.register("focal")
class FocalLoss(BaseLoss):
    def __init__(self, prior, gamma=2.0, **kw):
        super().__init__(prior, **kw)
        self.gamma = gamma

    def elementwise(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        return (1 - p_t).pow(self.gamma) * bce


@LOSSES.register("asl")
class AsymmetricLoss(BaseLoss):
    def __init__(self, prior, gamma_pos=0.0, gamma_neg=4.0, clip=0.05, **kw):
        super().__init__(prior, **kw)
        self.gamma_pos, self.gamma_neg, self.clip = gamma_pos, gamma_neg, clip

    def elementwise(self, logits, targets):
        p = torch.sigmoid(logits)
        p_neg = (p - self.clip).clamp(min=0) if self.clip > 0 else p  # probability shifting
        loss_pos = targets * torch.log(p.clamp(min=1e-8)) * (1 - p).pow(self.gamma_pos)
        loss_neg = (1 - targets) * torch.log((1 - p_neg).clamp(min=1e-8)) * p_neg.pow(self.gamma_neg)
        return -(loss_pos + loss_neg)


def build_loss(name, prior, **kw):
    return LOSSES.build(name, prior, **kw)
