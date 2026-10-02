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
        kw = {"query": {"n_heads": 4}, "dual": {"n_heads": 4}, "conv": {"grid": (16, 8)}, "seq": {"n_heads": 4}}.get(name, {})
        head = HEADS.build(name, D, A, **kw)
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


def test_new_options_copy_independent_and_shapes():
    """multi-layer tokens, MixStyle and the dual head: right shapes, EMA deepcopy computes with its own weights,
    MixStyle only acts in train mode."""
    import copy
    from upar.models import build_model
    torch.manual_seed(0)
    cfg = {"backbone": "vit_small_patch16_224", "size": [64, 32], "head": "dual", "head_kw": {"n_heads": 4},
           "multi_layer": 3, "mixstyle": {"block": 2, "p": 1.0, "alpha": 0.1}}
    m = build_model(cfg, pretrained=False).eval()
    x = torch.randn(4, 3, 64, 32)
    with torch.no_grad():
        out = m(x)
        assert out.shape == (4, A)
        c = copy.deepcopy(m)
        assert torch.equal(c(x), out)
        for p in m.parameters():
            p.add_(torch.randn_like(p))
        assert torch.equal(c(x), out)  # the copy does not read the original's block outputs
        assert torch.equal(c(x), c(x))  # eval: MixStyle off, deterministic
        c.train()
        for mod in c.modules():
            if isinstance(mod, torch.nn.Dropout):
                mod.p = 0.0
        for b in c.backbone.blocks:
            b.drop_path1 = b.drop_path2 = torch.nn.Identity()
        assert not torch.allclose(c(x), out)  # train: MixStyle changes the tokens
    names = [n for n, _ in m.named_parameters()]
    assert "neck.layer_w" in names
    groups = m.param_groups(1e-4, 0.05, 0.75, 10.0)
    assert any(g["lr"] == 1e-3 and any(p is m.neck["layer_w"] for p in g["params"]) for g in groups)  # head lr


def test_domain_heads_and_self_attention():
    """Dataset-specific classifiers: training picks the sample's classifier, inference averages the trained ones."""
    tokens = torch.randn(3, 5 + 128, D)
    h = HEADS.build("query", D, A, n_heads=4, n_domains=3, avg_domains=[0, 1], self_attn=True).eval()
    ds = torch.tensor([0, 1, 1])
    with torch.no_grad():
        per = h(tokens, 5, ds)
        own0 = h(tokens, 5, torch.zeros(3, dtype=torch.long))
        own1 = h(tokens, 5, torch.ones(3, dtype=torch.long))
        avg = h(tokens, 5)
    assert torch.allclose(per[0], own0[0]) and torch.allclose(per[1:], own1[1:])
    assert torch.allclose(avg, (own0 + own1) / 2, atol=1e-5)  # linear classifiers: averaging weights = averaging logits
    assert "layers.0.sa.in_proj_weight" in dict(h.named_parameters())


def test_moe_dense_train_equals_sparse_eval():
    """The dense (train) and sparse (eval) MoE paths compute the same function; the head trains end to end."""
    from upar.models.heads import MoEFFN
    torch.manual_seed(0)
    moe = MoEFFN(D, n_experts=8, topk=2, drop=0.0)
    torch.nn.init.normal_(moe.router.weight, std=0.5)
    x = torch.randn(4, 7, D)
    dense = moe.train()(x)
    sparse = moe.eval()(x)
    assert torch.allclose(dense, sparse, atol=1e-5)
    assert 1.0 <= moe.last_aux.item() < 8.0
    for where in ("tokens", "queries", "tokens+queries"):
        h = HEADS.build("query", D, A, n_heads=4, moe_experts=4, moe_where=where).train()
        out = h(torch.randn(2, 5 + 128, D), 5)
        out.sum().backward()
        assert out.shape == (2, A) and h.queries.grad is not None, where


def test_selective_ssm_is_causal_and_sequential_heads_train():
    """Step t of the selective scan never sees steps > t; both sequential heads train end to end."""
    from upar.models.heads import SelectiveSSM
    torch.manual_seed(0)
    ssm = SelectiveSSM(D, d_inner=64, d_state=8)
    torch.nn.init.normal_(ssm.out_proj.weight, std=0.1)
    x = torch.randn(2, 12, D)
    y = ssm(x)
    x2 = x.clone(); x2[:, 7:] += torch.randn_like(x2[:, 7:])
    y2 = ssm(x2)
    assert torch.allclose(y[:, :7], y2[:, :7], atol=1e-5) and not torch.allclose(y[:, 7:], y2[:, 7:])
    for mode in ("mamba", "seqattn"):
        h = HEADS.build("seq", D, A, mode=mode, n_heads=4, d_inner=64, d_state=8).train()
        out = h(torch.randn(2, 5 + 128, D), 5)
        out.sum().backward()
        assert out.shape == (2, A) and h.queries.grad.abs().sum() > 0, mode


def test_reconstruction_decoders():
    """Targets are 4x4 cell means of each patch; both decoders give a finite loss whose gradient reaches their inputs;
    the decoders are separate modules, so the model's state_dict (checkpoint, submission) is unchanged."""
    from upar.models.aux_decoders import AUX_DECODERS, patch_targets
    img = torch.arange(2 * 3 * 64 * 32, dtype=torch.float32).reshape(2, 3, 64, 32)
    t = patch_targets(img, (4, 2))
    assert t.shape == (2, 8, 48)
    assert torch.isclose(t[0, 0, 0], img[0, 0, :4, :4].mean())         # first cell of the first patch, red channel
    toks = torch.randn(2, 8, D, requires_grad=True)
    for name, src in (("mae", toks), ("attr", torch.randn(2, A, D, requires_grad=True))):
        dec = AUX_DECODERS[name](D, (4, 2), width=32, heads=4)
        loss = dec(src, img / img.max())
        loss.backward()
        assert torch.isfinite(loss) and src.grad.abs().sum() > 0, name


def test_query_attention_priors():
    """Body-part prior: hat looks at the top rows, lower-body colour at the bottom, gender is unbiased; both biases
    get gradients; with a zero bias the head equals the plain query head."""
    from upar.models.heads import body_prior
    pr = body_prior((16, 8), A).reshape(A, 16, 8)
    assert pr[39, 0, 0] > pr[39, 15, 0] and pr[25, 15, 0] > pr[25, 0, 0] and pr[3].abs().max() == 0
    torch.manual_seed(0)
    tokens = torch.randn(2, 5 + 128, D)
    h = HEADS.build("query", D, A, n_heads=4, spatial_prior=True, fg_bias=True, grid=(16, 8)).train()
    h(tokens, 5).sum().backward()
    assert h.prior.grad.abs().sum() > 0 and h.fg_w.grad.abs().sum() > 0
    torch.manual_seed(1); plain = HEADS.build("query", D, A, n_heads=4).eval()
    torch.manual_seed(1); biased = HEADS.build("query", D, A, n_heads=4, spatial_prior=True, grid=(16, 8)).eval()
    with torch.no_grad():
        biased.prior.zero_()
        assert torch.allclose(plain(tokens, 5), biased(tokens, 5), atol=1e-5)


def test_efficient_attention_heads_and_truncated_backbone():
    """Linear and sigmoid attention query heads train; a truncated ViT keeps only the first blocks."""
    for kind in ("linear", "sigmoid"):
        h = HEADS.build("query", D, A, n_heads=4, attn_type=kind).train()
        out = h(torch.randn(2, 5 + 128, D), 5)
        out.sum().backward()
        assert out.shape == (2, A) and torch.isfinite(out).all() and h.queries.grad.abs().sum() > 0, kind
    from upar.models import build_model
    m = build_model({"backbone": "vit_small_patch16_224", "size": [64, 32], "keep_blocks": 5}, pretrained=False)
    assert len(m.backbone.blocks) == 5 and m(torch.randn(2, 3, 64, 32)).shape == (2, A)


def test_pca_spotlight_bias():
    """fg_bias="pca" trains and its spotlight follows the token direction that separates two token groups."""
    torch.manual_seed(0)
    h = HEADS.build("query", D, A, n_heads=4, fg_bias="pca").train()
    out = h(torch.randn(2, 5 + 128, D), 5)
    out.sum().backward()
    assert out.shape == (2, A) and torch.isfinite(out).all() and h.fg_w.grad.abs().sum() > 0


def test_noise_adaptation_and_missing_distillation():
    """Anchor dataset: recorded = true. Other datasets: a and b move the recorded probability; distillation on
    missing cells ignores labelled cells."""
    from upar.losses import NoiseAdaptation, distill_missing
    torch.manual_seed(0)
    na = NoiseAdaptation(3, A, anchor=1, init=-1.0)
    z = torch.randn(4, A)
    out = na(z, torch.tensor([1, 1, 0, 2]))
    assert torch.allclose(out[:2], z[:2], atol=1e-4)                 # anchor rows unchanged
    assert not torch.allclose(out[2:], z[2:], atol=1e-3)
    out.sum().backward()
    assert na.a.grad[0].abs().sum() > 0 and na.a.grad[1].abs().sum() == 0
    s, t = torch.randn(2, A, requires_grad=True), torch.randn(2, A)
    v = torch.ones(2, A); v[:, :3] = 0
    distill_missing(s, t, v).backward()
    assert s.grad[:, 3:].abs().sum() == 0 and s.grad[:, :3].abs().sum() > 0


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
