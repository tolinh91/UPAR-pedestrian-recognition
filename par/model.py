"""DINOv3 ViT backbone (timm) + attribute head."""
import timm
import torch
import torch.nn as nn


class PARModel(nn.Module):
    def __init__(self, backbone="vit_base_patch16_dinov3.lvd1689m", num_attrs=40, img_size=(256, 128),
                 pretrained=True, drop=0.1, drop_path=0.1):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, img_size=img_size,
                                          drop_path_rate=drop_path)
        d = self.backbone.num_features
        self.n_prefix = getattr(self.backbone, "num_prefix_tokens", 1)
        # CLS token + mean of patch tokens
        self.head = nn.Sequential(nn.LayerNorm(2 * d), nn.Dropout(drop), nn.Linear(2 * d, num_attrs))

    def forward(self, x):
        t = self.backbone.forward_features(x)
        feat = torch.cat([t[:, 0], t[:, self.n_prefix:].mean(1)], 1)
        return self.head(feat)

    def param_groups(self, lr, weight_decay, layer_decay):
        """Layer-wise lr decay: head gets lr*10, deeper blocks get more of lr."""
        blocks = self.backbone.blocks
        n = len(blocks) + 1
        groups = {}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("head"):
                layer, scale = n, 10.0
            elif name.startswith("backbone.blocks."):
                layer = int(name.split(".")[2]) + 1
                scale = layer_decay ** (n - layer)
            elif name.startswith("backbone.norm") or name.startswith("backbone.fc_norm"):
                layer, scale = n, 1.0
            else:  # patch embed, cls/reg tokens, rope
                layer, scale = 0, layer_decay ** n
            wd = 0.0 if p.ndim == 1 or "token" in name or "pos_embed" in name else weight_decay
            key = (layer, wd, name.startswith("head"))
            g = groups.setdefault(key, {"params": [], "lr": lr * scale, "weight_decay": wd})
            g["params"].append(p)
        return list(groups.values())
