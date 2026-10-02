"""Backbone (timm) + attribute head from the HEADS registry.

    model = build_model({"backbone": "vit_base_patch16_dinov3.lvd1689m", "size": [256, 128],
                         "head": "query", "head_kw": {"n_layers": 1}})

Only torch and timm are imported, so this subpackage is vendored into the submission as is.
"""
import timm
import torch
import torch.nn as nn

from .heads import HEADS
from .radio import create_radio, is_radio

DEFAULT_BACKBONE = "vit_base_patch16_dinov3.lvd1689m"


def _keep_output(module, inputs, output):
    module.kept_output = output


class MixStyleHook:
    """Forward hook on a backbone block (train mode only): mixes the per-image channel mean / std of the patch tokens
    with those of another image of the batch (MixStyle, Zhou et al. 2021). The statistics carry camera / lighting /
    resolution "style"; mixing them across Market, PA100k and PETA makes the later blocks rely less on them."""

    def __init__(self, n_prefix, p=0.5, alpha=0.1, eps=1e-6):
        self.n_prefix, self.p, self.alpha, self.eps = n_prefix, p, alpha, eps

    def __call__(self, module, inputs, x):
        if not module.training:
            return x
        pre, tok = x[:, :self.n_prefix], x[:, self.n_prefix:]
        mu = tok.mean(1, keepdim=True).detach()
        sig = (tok.var(1, keepdim=True) + self.eps).sqrt().detach()
        B = x.shape[0]
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((B, 1, 1)).to(x)
        apply = (torch.rand(B, 1, 1, device=x.device) < self.p).to(x)
        lam = 1 - apply * (1 - lam)  # lam = 1 (no mixing) for the samples that are not selected
        perm = torch.randperm(B, device=x.device)
        mu_mix, sig_mix = lam * mu + (1 - lam) * mu[perm], lam * sig + (1 - lam) * sig[perm]
        return torch.cat([pre, (tok - mu) / sig * sig_mix + mu_mix], 1)


class PARModel(nn.Module):
    def __init__(self, backbone=DEFAULT_BACKBONE, num_attrs=40, img_size=(256, 128), pretrained=True,
                 head="linear", head_kw=None, drop_path=0.1, multi_layer=0, mixstyle=None, keep_blocks=0):
        super().__init__()
        if is_radio(backbone):  # NVIDIA C-RADIO: timm ViT + vendored patch generator, weights from Hugging Face
            self.backbone = create_radio(backbone, tuple(img_size), drop_path, pretrained=pretrained)
        else:
            kw = dict(pretrained=pretrained, num_classes=0, drop_path_rate=drop_path)
            try:  # ViTs take the input size (position embedding), CNNs do not
                self.backbone = timm.create_model(backbone, img_size=tuple(img_size), **kw)
            except TypeError:
                self.backbone = timm.create_model(backbone, **kw)
        if keep_blocks and hasattr(self.backbone, "blocks"):  # truncated backbone: first N blocks, cheaper on CPU
            self.backbone.blocks = self.backbone.blocks[:keep_blocks]
        # CNN (ConvNeXt, ...): feature map (B, C, H, W) -> tokens [global mean ; H*W cells], one prefix token
        # CNNs and SAM's ViT (its neck returns a (B, 256, H, W) map) give feature maps, not tokens
        self.is_cnn = not hasattr(self.backbone, "blocks") or backbone.startswith("samvit")
        if self.is_cnn and (multi_layer > 1 or mixstyle):
            raise ValueError("multi_layer / mixstyle need a ViT backbone")
        self.n_prefix = 1 if self.is_cnn else getattr(self.backbone, "num_prefix_tokens", 1)
        head_kw = dict(head_kw or {})
        if (head == "conv" or head_kw.get("spatial_prior")) and "grid" not in head_kw:  # patch grid of the input size
            ps = getattr(self.backbone, "patch_size", None) or getattr(getattr(self.backbone, "patch_embed", None),
                                                                     "patch_size", 16)
            ps = ps[0] if isinstance(ps, (tuple, list)) else ps
            head_kw["grid"] = (img_size[0] // ps, img_size[1] // ps)
        self.head = HEADS.build(head, self.backbone.num_features, num_attrs, **head_kw)
        # tokens of the last `multi_layer` blocks, mixed with learned softmax weights (mid-level colour / texture cues)
        self.multi_layer = multi_layer
        if multi_layer > 1:
            self.neck = nn.ParameterDict({"layer_w": nn.Parameter(torch.zeros(multi_layer))})
            for blk in self.backbone.blocks[-multi_layer:]:  # output kept on the block itself: safe for the EMA deepcopy
                blk.register_forward_hook(_keep_output)
        if mixstyle:
            blocks = self.backbone.blocks
            pg = getattr(self.backbone, "patch_generator", None)  # C-RADIO: CLS + registers inside the blocks
            n_pre = pg.num_skip if pg is not None else self.n_prefix
            blocks[mixstyle["block"]].register_forward_hook(
                MixStyleHook(n_pre, mixstyle.get("p", 0.5), mixstyle.get("alpha", 0.1)))

    def tokens(self, x):
        if self.is_cnn:
            t = self.backbone.forward_features(x).flatten(2).transpose(1, 2)
            return torch.cat([t.mean(1, keepdim=True), t], 1)
        if self.multi_layer <= 1:
            return self.backbone.forward_features(x)
        last = self.backbone.forward_features(x)
        w = self.neck["layer_w"].softmax(0)
        blocks = self.backbone.blocks[-self.multi_layer:]
        mixed = sum(wi * blk.kept_output for wi, blk in zip(w, blocks))
        for blk in blocks:
            blk.kept_output = None
        mixed = self.backbone.norm(mixed) if hasattr(self.backbone, "norm") else mixed
        if mixed.shape[1] != last.shape[1]:  # C-RADIO drops registers after the blocks: keep its token layout
            mixed = torch.cat([last[:, :1], mixed[:, -(last.shape[1] - 1):]], 1)
        return mixed

    def forward(self, x, ds=None, return_aux=False):
        t = self.tokens(x)
        args = (t, self.n_prefix) if ds is None else (t, self.n_prefix, ds)
        if not return_aux:
            return self.head(*args)
        if getattr(self.head, "supports_q", False):  # training-only decoders read tokens / attribute embeddings
            logits, q = self.head(*args, return_q=True)
        else:
            logits, q = self.head(*args), None
        return logits, t, q

    def freeze_early(self, n_blocks):
        """Freeze everything before block n_blocks (patch embed, cls / register tokens, rope, first blocks)."""
        n = 0
        for name, p in self.backbone.named_parameters():
            if name.startswith("blocks."):
                if int(name.split(".")[1]) >= n_blocks:
                    continue
            elif name.startswith(("norm", "fc_norm", "head", "neck")):  # SAM neck sits after the blocks
                continue
            p.requires_grad_(False)
            n += p.numel()
        return n

    def param_groups(self, lr, weight_decay, layer_decay, head_lr_mult=10.0):
        """Layer-wise lr decay: the last backbone block gets lr, each block below x layer_decay, head x head_lr_mult."""
        cnn_layer = {}
        if self.is_cnn:  # timm's fine-grained parameter grouping (stem, then stage / block order)
            from timm.models import group_parameters
            for gid, names in group_parameters(self.backbone, self.backbone.group_matcher(coarse=False)).items():
                cnn_layer.update({"backbone." + nm: gid for nm in names})
            n = max(cnn_layer.values()) + 1
        else:
            n = len(self.backbone.blocks) + 1
        groups = {}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith(("head.", "neck.")):
                layer, scale = n + 1, head_lr_mult
            elif name in cnn_layer:
                layer = cnn_layer[name]
                scale = layer_decay ** (n - 1 - layer)
            elif name.startswith("backbone.blocks."):
                layer = int(name.split(".")[2]) + 1
                scale = layer_decay ** (n - layer)
            elif name.startswith("backbone.norm") or name.startswith("backbone.fc_norm"):
                layer, scale = n, 1.0
            else:  # patch embed, cls / register tokens, rope
                layer, scale = 0, layer_decay ** n
            no_wd = p.ndim == 1 or "token" in name or "pos_embed" in name or "queries" in name
            wd = 0.0 if no_wd else weight_decay
            g = groups.setdefault((layer, wd), {"params": [], "lr": lr * scale, "weight_decay": wd})
            g["params"].append(p)
        return list(groups.values())


def build_model(cfg, pretrained=True):
    """cfg: dict with backbone, size and optional head / head_kw / num_attrs (train args or submission config)."""
    return PARModel(cfg.get("backbone", DEFAULT_BACKBONE), cfg.get("num_attrs", 40), cfg.get("size", (256, 128)),
                    pretrained=pretrained, head=cfg.get("head", "linear"), head_kw=cfg.get("head_kw") or {},
                    multi_layer=cfg.get("multi_layer", 0), mixstyle=cfg.get("mixstyle"), keep_blocks=cfg.get("keep_blocks", 0))


def cfg_from_checkpoint(ck):
    """Model config of a checkpoint; checkpoints before the registry only stored backbone + size (linear head)."""
    if "model_cfg" in ck:
        return ck["model_cfg"]
    a = ck["args"]
    return {"backbone": a["backbone"], "size": a["size"], "head": a.get("head", "linear"), "head_kw": a.get("head_kw", {})}


__all__ = ["PARModel", "build_model", "cfg_from_checkpoint", "HEADS"]
