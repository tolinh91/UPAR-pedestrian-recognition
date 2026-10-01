"""Fast checks of losses and heads (no data, no pretrained weights).

cd par && ../.venv/bin/python -m pytest -q tests   (or: ../.venv/bin/python tests/test_components.py)
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from upar.losses import LOSSES, build_loss  # noqa: E402
from upar.models.heads import HEADS  # noqa: E402

torch.manual_seed(0)
B, A, D = 8, 40, 64
PRIOR = np.linspace(0.002, 0.95, A)


def _batch():
    z = torch.randn(B, A, requires_grad=True)
    y = (torch.rand(B, A) < torch.tensor(PRIOR, dtype=torch.float32)).float()
    v = torch.ones(B, A)
    v[:, :3] = 0  # unannotated cells
    return z, y, v


def test_all_losses_finite_and_differentiable():
    for name in LOSSES.names():
        for weighting in ["none", "deepmar", "sqrt", "inverse"]:
            z, y, v = _batch()
            loss = build_loss(name, PRIOR, smooth=0.05, weighting=weighting)(z, y, v)
            assert torch.isfinite(loss), (name, weighting)
            loss.backward()
            assert torch.isfinite(z.grad).all(), (name, weighting)


def test_masked_cells_get_no_gradient():
    for name in LOSSES.names():
        z, y, v = _batch()
        build_loss(name, PRIOR)(z, y, v).backward()
        assert z.grad[:, :3].abs().sum() == 0, name


def test_deepmar_bce_matches_submission_v3_formula():
    z, y, v = _batch()
    new = build_loss("bce", PRIOR, smooth=0.05, weighting="deepmar")(z, y, v)
    p = torch.tensor(PRIOR, dtype=torch.float32)
    ys = y * 0.95 + 0.025
    w = torch.where(y > 0.5, torch.exp(1 - p), torch.exp(p)) * v
    old = (F.binary_cross_entropy_with_logits(z, ys, reduction="none") * w).sum() / v.sum()
    assert torch.allclose(new, old, atol=1e-6)


def test_logit_adjustment_pushes_rare_attributes_up():
    """With the same logits, logit_adj gives rare positives a larger gradient towards higher logits."""
    z = torch.zeros(1, A, requires_grad=True)
    y, v = torch.ones(1, A), torch.ones(1, A)
    build_loss("logit_adj", PRIOR, tau=1.0)(z, y, v).backward()
    assert z.grad[0, 0] < z.grad[0, -1] < 0  # rarest attribute (index 0) gets the strongest push


def test_asl_reduces_easy_negative_loss():
    z = torch.full((1, A), -4.0)  # confident negatives
    y, v = torch.zeros(1, A), torch.ones(1, A)
    assert build_loss("asl", PRIOR)(z, y, v) < build_loss("bce", PRIOR)(z, y, v)


def test_heads_output_shape():
    tokens = torch.randn(2, 5 + 128, D)
    for name in HEADS.names():
        head = HEADS.build(name, D, A, **({"n_heads": 4} if name == "query" else {}))
        assert head(tokens, 5).shape == (2, A), name


def test_linear_head_keeps_v3_parameter_names():
    names = [n for n, _ in HEADS.build("linear", D, A).named_parameters()]
    assert names == ["0.weight", "0.bias", "2.weight", "2.bias"]


def test_distillation_loss():
    from upar.losses import distill_bce
    t = torch.randn(B, A)
    s = t.clone().requires_grad_(True)
    same = distill_bce(s, t)
    other = distill_bce(torch.zeros(B, A), t)
    assert same < other  # student equal to teacher is the minimum
    same.backward()
    assert torch.allclose(s.grad, torch.zeros_like(s.grad), atol=1e-6)  # zero gradient at the optimum
    # T^2 scaling keeps the loss magnitude comparable across temperatures
    z = torch.randn(B, A)
    assert 0.2 < distill_bce(z, t, T=2.0) / distill_bce(z, t, T=1.0) < 5


def test_radio_copy_is_independent():
    """The EMA is a deepcopy: it must compute with its own weights (closures would keep the original's)."""
    import copy
    from upar.models import build_model
    torch.manual_seed(0)
    m = build_model({"backbone": "nvidia/C-RADIOv3-B", "size": [64, 32]}, pretrained=False).eval()
    c = copy.deepcopy(m)
    x = torch.randn(2, 3, 64, 32)
    with torch.no_grad():
        before = c(x)
        for p in m.parameters():
            p.add_(torch.randn_like(p))
        assert torch.equal(c(x), before)
        assert not torch.allclose(m(x), before)
    # train mode uses the eval position embedding (no random CPE crop): equal once dropout / drop path are off
    for mod in c.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0
    c.backbone.patch_generator.pos_dropout = 0.0
    for b in c.backbone.blocks:
        b.drop_path1 = b.drop_path2 = torch.nn.Identity()
    with torch.no_grad():
        assert torch.allclose(c.train()(x), before, atol=1e-5)


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
