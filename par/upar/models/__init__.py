"""Backbone (timm) + attribute head from the HEADS registry.

    model = build_model({"backbone": "vit_base_patch16_dinov3.lvd1689m", "size": [256, 128],
                         "head": "query", "head_kw": {"n_layers": 1}})

Only torch and timm are imported, so this subpackage is vendored into the submission as is.
"""
import timm
import torch.nn as nn

from .heads import HEADS

DEFAULT_BACKBONE = "vit_base_patch16_dinov3.lvd1689m"


class PARModel(nn.Module):
    def __init__(self, backbone=DEFAULT_BACKBONE, num_attrs=40, img_size=(256, 128), pretrained=True,
                 head="linear", head_kw=None, drop_path=0.1):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, img_size=tuple(img_size),
                                          drop_path_rate=drop_path)
        self.n_prefix = getattr(self.backbone, "num_prefix_tokens", 1)
        self.head = HEADS.build(head, self.backbone.num_features, num_attrs, **(head_kw or {}))

    def forward(self, x):
        return self.head(self.backbone.forward_features(x), self.n_prefix)

    def param_groups(self, lr, weight_decay, layer_decay, head_lr_mult=10.0):
        """Layer-wise lr decay: the last backbone block gets lr, each block below x layer_decay, head x head_lr_mult."""
        n = len(self.backbone.blocks) + 1
        groups = {}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("head."):
                layer, scale = n + 1, head_lr_mult
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
                    pretrained=pretrained, head=cfg.get("head", "linear"), head_kw=cfg.get("head_kw") or {})


def cfg_from_checkpoint(ck):
    """Model config of a checkpoint; checkpoints before the registry only stored backbone + size (linear head)."""
    if "model_cfg" in ck:
        return ck["model_cfg"]
    a = ck["args"]
    return {"backbone": a["backbone"], "size": a["size"], "head": a.get("head", "linear"), "head_kw": a.get("head_kw", {})}


__all__ = ["PARModel", "build_model", "cfg_from_checkpoint", "HEADS"]
